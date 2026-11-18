//! Where private text lives in each wire protocol, and how values go back into responses and streams.
//!
//! Requests: every string is redacted unless it sits under a structural key, so a field an agent starts
//! using tomorrow is redacted by default instead of leaking. Thinking blocks are never touched in either
//! direction because providers sign them.
use crate::vault::{StreamRestorer, Vault};
use anyhow::Result;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Anthropic,
    Chat,
    Responses,
}

impl Kind {
    pub fn of(path: &str) -> Option<Kind> {
        if path.ends_with("/messages") || path.ends_with("/messages/count_tokens") {
            Some(Kind::Anthropic)
        } else if path.ends_with("/chat/completions") {
            Some(Kind::Chat)
        } else if path.ends_with("/responses") {
            Some(Kind::Responses)
        } else {
            None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Anthropic => "anthropic",
            Kind::Chat => "chat",
            Kind::Responses => "responses",
        }
    }
}

const SKIP_TOP: &[&str] = &["model", "tools", "tool_choice", "metadata", "response_format", "text", "reasoning", "include", "thinking",
    "service_tier", "user", "prompt_cache_key", "stop", "stop_sequences", "stream_options", "output_config"];
const SKIP_KEY: &[&str] = &["id", "call_id", "tool_use_id", "tool_call_id", "type", "role", "name", "signature", "encrypted_content",
    "cache_control", "status", "media_type", "data", "source", "image_url", "file_data", "file_id", "url"];
const SIGNED: &[&str] = &["thinking", "redacted_thinking"];

pub const NOTE: &str = "Holdback, a local privacy layer: some values in this conversation appear as placeholders such as [PRIVATE_EMAIL_2]. Each one stands \
for a real value that is substituted automatically wherever you write it. Use placeholders exactly as written, \
in tool calls, file edits and replies, as if they were the real values. Never try to recover the real values: \
do not ask for them, and do not dump, encode or decode files (od, xxd, hexdump, base64) to see them.";

fn type_of(m: &Map<String, Value>) -> &str {
    m.get("type").and_then(Value::as_str).unwrap_or("")
}

fn walk(v: Value, vault: &mut Vault, key: Option<&str>) -> Result<Value> {
    Ok(match v {
        Value::String(s) => {
            if key == Some("arguments")
                && let Ok(parsed) = serde_json::from_str::<Value>(&s)
            {
                return Ok(Value::String(serde_json::to_string(&walk(parsed, vault, None)?)?));
            }
            Value::String(vault.redact(&s)?)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(|x| walk(x, vault, None)).collect::<Result<_>>()?),
        Value::Object(m) => {
            if SIGNED.contains(&type_of(&m)) {
                return Ok(Value::Object(m));
            }
            let mut out = Map::new();
            for (k, x) in m {
                let x = if SKIP_KEY.contains(&k.as_str()) { x } else { walk(x, vault, Some(&k))? };
                out.insert(k, x);
            }
            Value::Object(out)
        }
        other => other,
    })
}

fn add_note(kind: Kind, body: &mut Map<String, Value>) {
    match kind {
        Kind::Anthropic => match body.get_mut("system") {
            None => {
                body.insert("system".into(), Value::String(NOTE.into()));
            }
            Some(Value::String(s)) => {
                s.push_str("\n\n");
                s.push_str(NOTE);
            }
            Some(Value::Array(a)) => a.push(json!({"type": "text", "text": NOTE})),
            Some(_) => {}
        },
        Kind::Chat => {
            if let Some(Value::Array(msgs)) = body.get_mut("messages") {
                msgs.insert(0, json!({"role": "system", "content": NOTE}));
            }
        }
        Kind::Responses => {
            let joined = match body.get("instructions").and_then(Value::as_str) {
                Some(s) if !s.is_empty() => format!("{s}\n\n{NOTE}"),
                _ => NOTE.to_string(),
            };
            body.insert("instructions".into(), Value::String(joined));
        }
    }
}

pub fn redact_request(kind: Kind, body: Value, vault: &mut Vault) -> Result<Value> {
    let Value::Object(m) = body else { return walk(body, vault, None) };
    let mut out = Map::new();
    for (k, v) in m {
        let v = if SKIP_TOP.contains(&k.as_str()) { v } else { walk(v, vault, Some(&k))? };
        out.insert(k, v);
    }
    add_note(kind, &mut out);
    Ok(Value::Object(out))
}

/// Restore placeholders in any response object, skipping thinking blocks; "arguments" strings are JSON text.
pub fn restore_obj(v: Value, vault: &Vault, key: Option<&str>) -> Value {
    match v {
        Value::String(s) => Value::String(vault.restore(&s, matches!(key, Some("arguments") | Some("partial_json")))),
        Value::Array(a) => Value::Array(a.into_iter().map(|x| restore_obj(x, vault, None)).collect()),
        Value::Object(m) => {
            if matches!(type_of(&m), "thinking" | "redacted_thinking" | "thinking_delta" | "signature_delta") {
                return Value::Object(m);
            }
            Value::Object(
                m.into_iter()
                    .map(|(k, x)| {
                        let keep = matches!(k.as_str(), "signature" | "encrypted_content" | "id" | "call_id" | "tool_use_id");
                        let x = if keep { x } else { restore_obj(x, vault, Some(&k)) };
                        (k, x)
                    })
                    .collect(),
            )
        }
        other => other,
    }
}

/// Rewrites the SSE events of one response. `feed` takes one event and returns the events to send, which may
/// include a synthetic delta carrying text held back at the end of a block.
pub struct StreamRewriter {
    kind: Kind,
    restorers: HashMap<String, StreamRestorer>,
}

type Event = (Option<String>, Value);

impl StreamRewriter {
    pub fn new(kind: Kind) -> Self {
        StreamRewriter { kind, restorers: HashMap::new() }
    }

    fn piece(&mut self, key: String, json_escaped: bool, text: &str, vault: &Vault) -> String {
        self.restorers.entry(key).or_insert_with(|| StreamRestorer::new(json_escaped)).feed(text, vault)
    }

    fn tail(&mut self, key: &str, vault: &Vault) -> String {
        self.restorers.remove(key).map(|mut r| r.flush(vault)).unwrap_or_default()
    }

    pub fn feed(&mut self, event: Option<String>, data: Value, vault: &Vault) -> Vec<Event> {
        match self.kind {
            Kind::Anthropic => self.anthropic(event, data, vault),
            Kind::Chat => self.chat(event, data, vault),
            Kind::Responses => self.responses(event, data, vault),
        }
    }

    fn anthropic(&mut self, event: Option<String>, mut d: Value, vault: &Vault) -> Vec<Event> {
        let t = d["type"].as_str().unwrap_or("").to_string();
        let i = d["index"].as_i64().unwrap_or(0);
        if t == "content_block_delta" {
            let dtype = d["delta"]["type"].as_str().unwrap_or("").to_string();
            if dtype == "text_delta" {
                let s = d["delta"]["text"].as_str().unwrap_or("").to_string();
                d["delta"]["text"] = Value::String(self.piece(format!("{i}/text"), false, &s, vault));
            } else if dtype == "input_json_delta" {
                let s = d["delta"]["partial_json"].as_str().unwrap_or("").to_string();
                d["delta"]["partial_json"] = Value::String(self.piece(format!("{i}/json"), true, &s, vault));
            }
            return vec![(event, d)];
        }
        if t == "content_block_stop" {
            let mut out = vec![];
            for (k, dtype, field) in [("text", "text_delta", "text"), ("json", "input_json_delta", "partial_json")] {
                let tail = self.tail(&format!("{i}/{k}"), vault);
                if !tail.is_empty() {
                    out.push((Some("content_block_delta".into()), json!({"type": "content_block_delta", "index": i, "delta": {"type": dtype, field: tail}})));
                }
            }
            out.push((event, d));
            return out;
        }
        vec![(event, restore_obj(d, vault, None))]
    }

    fn chat(&mut self, event: Option<String>, mut d: Value, vault: &Vault) -> Vec<Event> {
        let mut extra_chunks = vec![];
        let choices = d.get_mut("choices").and_then(Value::as_array_mut).map(std::mem::take).unwrap_or_default();
        let mut new_choices = vec![];
        for mut ch in choices {
            let ci = ch["index"].as_i64().unwrap_or(0);
            if let Some(delta) = ch.get_mut("delta").and_then(Value::as_object_mut) {
                for field in ["content", "reasoning_content"] {
                    if let Some(s) = delta.get(field).and_then(Value::as_str).map(str::to_string) {
                        let r = self.piece(format!("{ci}/{field}"), false, &s, vault);
                        delta.insert(field.into(), Value::String(r));
                    }
                }
                if let Some(Value::Array(calls)) = delta.get_mut("tool_calls") {
                    for tc in calls {
                        let ti = tc["index"].as_i64().unwrap_or(0);
                        if let Some(s) = tc["function"]["arguments"].as_str().map(str::to_string) {
                            tc["function"]["arguments"] = Value::String(self.piece(format!("{ci}/tool/{ti}"), true, &s, vault));
                        }
                    }
                }
            }
            if ch["finish_reason"].is_string() {
                let prefix = format!("{ci}/");
                let keys: Vec<String> = self.restorers.keys().filter(|k| k.starts_with(&prefix)).cloned().collect();
                let mut extra = Map::new();
                for key in keys {
                    let tail = self.tail(&key, vault);
                    if tail.is_empty() {
                        continue;
                    }
                    let rest = &key[prefix.len()..];
                    if let Some(ti) = rest.strip_prefix("tool/") {
                        let calls = extra.entry("tool_calls").or_insert_with(|| json!([]));
                        calls.as_array_mut().unwrap().push(json!({"index": ti.parse::<i64>().unwrap_or(0), "function": {"arguments": tail}}));
                    } else {
                        extra.insert(rest.to_string(), Value::String(tail));
                    }
                }
                if !extra.is_empty() {
                    let mut chunk = d.clone();
                    chunk["choices"] = json!([{"index": ci, "delta": extra, "finish_reason": null}]);
                    extra_chunks.push((event.clone(), chunk));
                }
            }
            new_choices.push(ch);
        }
        if d.get("choices").is_some() {
            d["choices"] = Value::Array(new_choices);
        }
        extra_chunks.push((event, d));
        extra_chunks
    }

    fn responses(&mut self, event: Option<String>, mut d: Value, vault: &Vault) -> Vec<Event> {
        let t = d["type"].as_str().map(str::to_string).or(event.clone()).unwrap_or_default();
        let item = d["item_id"].as_str().unwrap_or("").to_string();
        let ci = d["content_index"].as_i64().unwrap_or(0);
        match t.as_str() {
            "response.output_text.delta" => {
                let s = d["delta"].as_str().unwrap_or("").to_string();
                d["delta"] = Value::String(self.piece(format!("{item}/{ci}/text"), false, &s, vault));
                vec![(event, d)]
            }
            "response.function_call_arguments.delta" => {
                let s = d["delta"].as_str().unwrap_or("").to_string();
                d["delta"] = Value::String(self.piece(format!("{item}/args"), true, &s, vault));
                vec![(event, d)]
            }
            _ => {
                let mut out = vec![];
                let held = match t.as_str() {
                    "response.output_text.done" => Some((format!("{item}/{ci}/text"), "response.output_text.delta")),
                    "response.function_call_arguments.done" => Some((format!("{item}/args"), "response.function_call_arguments.delta")),
                    _ => None,
                };
                if let Some((key, delta_type)) = held {
                    let tail = self.tail(&key, vault);
                    if !tail.is_empty() {
                        out.push((Some(delta_type.into()), json!({"type": delta_type, "item_id": item, "output_index": d["output_index"],
                            "content_index": d["content_index"], "delta": tail, "sequence_number": d["sequence_number"]})));
                    }
                }
                out.push((event, restore_obj(d, vault, None)));
                out
            }
        }
    }
}
