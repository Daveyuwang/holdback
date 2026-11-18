"""Real end-to-end check of the proxy on three wire protocols, streaming, two-turn tool loop.

Run the proxy first:  HOLDBACK_DUMP=/tmp/hb_dump.jsonl HOLDBACK_UPSTREAMS=bailian=https://$BAILIAN_BASE_URL/compatible-mode holdback serve
Then:                 uv run python bench/check_proxy.py /tmp/hb_dump.jsonl"""
import json
import os
import re
import sys

import anthropic
from openai import OpenAI

BASE = "http://127.0.0.1:8787"
NAME, EMAIL, PHONE = "Maria Gonzalez", "maria.gonzalez@gmail.com", "+1 415-555-0198"
PROMPT = (f"Customer record: Name: {NAME}, email {EMAIL}, phone {PHONE}. Her refund #R-4821 was approved today. "
          "Send her a short email confirming the refund using the send_email tool, then tell me in one sentence what you did.")
SCHEMA = {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
          "required": ["to", "subject", "body"]}
results = []


def check(name, to, final):
    ok = to == EMAIL
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: tool 'to'={to!r}  final={final[:90]!r}")


def anthropic_case():
    c = anthropic.Anthropic(base_url=f"{BASE}/anthropic", max_retries=0)
    tools = [{"name": "send_email", "description": "Send an email.", "input_schema": SCHEMA}]
    msgs = [{"role": "user", "content": PROMPT}]
    with c.messages.stream(model="claude-haiku-5-5", max_tokens=1024, tools=tools, messages=msgs) as s:
        m1 = s.get_final_message()
    tu = next(b for b in m1.content if b.type == "tool_use")
    msgs += [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in m1.content]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": f"Email sent to {tu.input['to']}."}]}]
    with c.messages.stream(model="claude-haiku-5-5", max_tokens=1024, tools=tools, messages=msgs) as s:
        m2 = s.get_final_message()
    check("anthropic messages -> Claude", tu.input["to"], "".join(b.text for b in m2.content if b.type == "text"))


def chat_case():
    c = OpenAI(base_url=f"{BASE}/deepseek/v1", api_key=os.environ["DEEPSEEK_API_KEY"], max_retries=0)
    tools = [{"type": "function", "function": {"name": "send_email", "description": "Send an email.", "parameters": SCHEMA}}]
    msgs = [{"role": "user", "content": PROMPT}]
    kw = dict(model="deepseek-flash", tools=tools, extra_body={"thinking": {"type": "disabled"}})
    stream = c.chat.completions.create(messages=msgs, stream=True, **kw)
    args, call_id, text = "", None, ""
    for ch in stream:
        d = ch.choices[0].delta if ch.choices else None
        if d and d.tool_calls:
            for tc in d.tool_calls:
                call_id = tc.id or call_id
                args += tc.function.arguments or ""
        if d and d.content:
            text += d.content
    to = json.loads(args)["to"]
    msgs += [{"role": "assistant", "content": text or None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "send_email", "arguments": args}}]},
             {"role": "tool", "tool_call_id": call_id, "content": f"Email sent to {to}."}]
    r2 = c.chat.completions.create(messages=msgs, **kw)
    check("openai chat -> DeepSeek", to, r2.choices[0].message.content or "")


def responses_case():
    c = OpenAI(base_url=f"{BASE}/bailian/v1", api_key=os.environ["BAILIAN_API_KEY"], max_retries=0)
    tools = [{"type": "function", "name": "send_email", "description": "Send an email.", "parameters": SCHEMA}]
    inp = [{"role": "user", "content": PROMPT}]
    kw = dict(model="qwen3.8-flash", tools=tools, extra_body={"enable_thinking": False})
    with c.responses.stream(input=inp, **kw) as s:
        r1 = s.get_final_response()
    call = next(o for o in r1.output if o.type == "function_call")
    to = json.loads(call.arguments)["to"]
    inp += [o.model_dump(exclude_none=True) for o in r1.output]
    inp.append({"type": "function_call_output", "call_id": call.call_id, "output": f"Email sent to {to}."})
    with c.responses.stream(input=inp, **kw) as s:
        r2 = s.get_final_response()
    check("openai responses -> Qwen", to, r2.output_text or "")


for case in (anthropic_case, chat_case, responses_case):
    try:
        case()
    except Exception as e:
        results.append(False)
        print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")

if len(sys.argv) > 1:
    sent = open(sys.argv[1], encoding="utf-8").read()
    leaks = [v for v in (NAME, EMAIL, PHONE, "415-555-0198") if v in sent]
    phs = sorted(set(re.findall(r"\[[A-Z][A-Z_]*_\d+\]", sent)))
    print(f"outgoing requests: {sent.count(chr(10))}; leaked values: {leaks or 'none'}; placeholders seen: {phs}")
print(f"{sum(results)}/{len(results)} passed")
