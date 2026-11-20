"""Task success under redaction: do models still answer when private values are replaced?

Conditions per task: raw | redacted (detector spans -> <REDACTED>) | placeholder (detector spans -> numbered
placeholders, restored in the answer) | refined (holdback's spans, restored) | oracle (only the true private values).
The compared tools in spans.py are conditions too: codex (its mask), presidio (numbered tags, not restored) and
rehydra (its tags, restored).
Answers are kept in bench/results/answers.jsonl with a digest of the prompt they answer, so after a change of
detector or rules only the prompts that changed are sent again.
Needs DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, BAILIAN_API_KEY and BAILIAN_BASE_URL.
Usage: uv run python bench/task_eval.py [--runs 3] [--models deepseek,claude,qwen] [--conds raw,codex,presidio,rehydra,refined]"""
import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic
from openai import OpenAI

from families import generate
from spans import COMPARED, RESULTS, merged, spans

TAGS = {  # condition -> (what replaces the n-th distinct value, whether the answer gets the value back)
    "redacted": (lambda label, n: "<REDACTED>", False),
    "placeholder": (lambda label, n: f"[{label.upper()}_{n}]", True),
    "refined": (lambda label, n: f"[{label.upper()}_{n}]", True),
    "codex": (lambda label, n: "[REDACTED_SECRET]", False),
    "presidio": (lambda label, n: f"<{label}_{n}>", False),
    "rehydra": (lambda label, n: f'<PII type="{label}" id="{n}"/>', True),
}


def replace(ctx, sp, cond):
    tag, restore = TAGS[cond]
    out, last, mapping, seen = [], 0, {}, {}
    for s, e, label in merged(sp):
        value = ctx[s:e]
        key = seen.setdefault(value, tag(label, len(seen) + 1))
        mapping[key] = value
        out += [ctx[last:s], key]
        last = e
    return "".join(out + [ctx[last:]]), mapping if restore else {}


def oracle(ctx, secrets):
    mapping = {}
    for i, s in enumerate(sorted(set(secrets), key=len, reverse=True)):
        if s in ctx:
            ctx = ctx.replace(s, f"[PRIVATE_{i + 1}]")
            mapping[f"[PRIVATE_{i + 1}]"] = s
    return ctx, mapping


def correct(answer, need):
    return all(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", answer) if n.isdigit() else n.lower() in answer.lower() for n in need)


def asker():
    ds = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    qw = OpenAI(base_url=f"https://{os.environ['BAILIAN_BASE_URL']}/compatible-mode/v1", api_key=os.environ["BAILIAN_API_KEY"])
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def ask(model, prompt):
        msgs = [{"role": "user", "content": prompt}]
        if model == "deepseek":
            r = ds.chat.completions.create(model="deepseek-flash", max_tokens=300, messages=msgs, extra_body={"thinking": {"type": "disabled"}})
            return r.choices[0].message.content or ""
        if model == "qwen":
            r = qw.chat.completions.create(model="qwen3.8-flash", max_tokens=300, messages=msgs, extra_body={"enable_thinking": False})
            return r.choices[0].message.content or ""
        r = cl.messages.create(model="claude-haiku-5-5", max_tokens=1024, messages=msgs)  # Haiku thinks first; 1024 leaves room
        return "".join(b.text for b in r.content if b.type == "text")
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--models", default="deepseek,claude,qwen")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--conds", default="raw,redacted,placeholder,refined,oracle")
    a = ap.parse_args()
    models, conds, tasks = a.models.split(","), a.conds.split(","), generate()
    texts = [t.ctx for t in tasks]
    sp = spans(texts, "spans_seed7")
    found = {"redacted": [r["detected"] for r in sp], "placeholder": [r["detected"] for r in sp], "refined": [r["refined"] for r in sp],
             **{c: COMPARED[c](texts, "spans_seed7") for c in conds if c in COMPARED}}
    prompts = {}
    for i, t in enumerate(tasks):
        for c in conds:
            prompts[i, c] = (t.ctx, {}) if c == "raw" else oracle(t.ctx, t.secrets) if c == "oracle" else replace(t.ctx, found[c][i], c)
    sig = {(i, c): hashlib.sha256(f"{sent}\n\n{tasks[i].q}".encode()).hexdigest()[:16] for (i, c), (sent, _) in prompts.items()}

    log = RESULTS / "answers.jsonl"
    done = {}  # answers to the prompts of this run only
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            if sig.get((d["i"], d["cond"])) == d.get("sig"):
                done[d["i"], d["model"], d["cond"], d["run"]] = d
    jobs = [(i, m, c, k) for k in range(a.runs) for m in models for c in conds for i in range(len(tasks)) if (i, m, c, k) not in done]
    ask = asker()

    def work(job):
        i, m, c, k = job
        sent, mapping = prompts[i, c]
        for attempt in range(4):
            try:
                answer = ask(m, f"{sent}\n\n{tasks[i].q}")
                break
            except Exception as e:  # rate limits and transient errors
                answer = f"__ERROR__ {type(e).__name__}"
                time.sleep(2 ** attempt * 2)
        for key, val in mapping.items():
            answer = answer.replace(key, val)
        return {"i": i, "model": m, "cond": c, "run": k, "sig": sig[i, c], "answer": answer[:400], "ok": correct(answer, tasks[i].need)}

    print(f"{len(jobs)} model calls", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, log.open("a") as f:
        for d in ex.map(work, jobs):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            done[d["i"], d["model"], d["cond"], d["run"]] = d

    rate = defaultdict(list)  # (model, cond, needs private value) -> success rate per run
    for m in models:
        for c in conds:
            for k in range(a.runs):
                for private in (False, True):
                    oks = [d["ok"] for (i, mm, cc, kk), d in done.items() if (mm, cc, kk) == (m, c, k) and (tasks[i].cat == "reason_pii") == private]
                    rate[m, c, private].append(100 * sum(oks) / len(oks))
    print(f"errors: {sum(d['answer'].startswith('__ERROR__') for d in done.values())}")
    for private, title in ((False, "tasks that need a non-private value"), (True, "tasks that need the private value itself")):
        print(f"\nsuccess %, {title}" + (f", mean (min-max) over {a.runs} runs" if a.runs > 1 else ""))
        print(f"{'model':9s}" + "".join(f"{c:>16s}" for c in conds))
        for m in models:
            cells = (f"{sum(v) / len(v):.1f}" + (f" ({min(v):.0f}-{max(v):.0f})" if len(v) > 1 else "") for v in (rate[m, c, private] for c in conds))
            print(f"{m:9s}" + "".join(cell.rjust(16) for cell in cells))


if __name__ == "__main__":
    main()
