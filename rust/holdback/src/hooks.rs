//! Hook mode: Claude Code hook events in, hook answers out.
//!
//! For a session that cannot be routed through the proxy, such as one in the Claude desktop app. A PostToolUse
//! hook replaces private values in what a tool returned before the model sees it, a PreToolUse hook puts the
//! values back into what a tool is about to receive, and a MessageDisplay hook puts them back into the text
//! shown to the user. Not covered: the system prompt, what the user types, and the output of a failed tool
//! call, which a hook cannot rewrite.
use crate::protocols::NOTE;
use crate::vault::Vault;
use anyhow::Result;
use serde_json::{Value, json};

/// Applies `f` to every string in `v`, except under the keys in `skip`. True when a string changed.
fn map_strings(v: &mut Value, skip: &[&str], f: &mut dyn FnMut(&str) -> Result<String>) -> Result<bool> {
    Ok(match v {
        Value::String(s) => {
            let new = f(s)?;
            let changed = new != *s;
            *s = new;
            changed
        }
        Value::Array(a) => {
            let mut changed = false;
            for x in a {
                changed |= map_strings(x, skip, f)?;
            }
            changed
        }
        Value::Object(m) => {
            let mut changed = false;
            for (k, x) in m {
                if !skip.contains(&k.as_str()) {
                    changed |= map_strings(x, skip, f)?;
                }
            }
            changed
        }
        _ => false,
    })
}

fn specific(event: &str, field: &str, value: Value) -> Value {
    json!({"hookSpecificOutput": {"hookEventName": event, field: value}})
}

/// The answer to one hook event. An empty object leaves the event as it is.
pub fn answer(vault: &mut Vault, event: &Value) -> Result<Value> {
    let name = event["hook_event_name"].as_str().unwrap_or("");
    Ok(match name {
        // ToolSearch returns tool definitions that Claude Code loads by reference; a rewritten result loses them
        "PostToolUse" if event["tool_name"] != "ToolSearch" => {
            let mut out = event["tool_response"].clone();
            if map_strings(&mut out, &["base64"], &mut |s| vault.redact(s))? { specific(name, "updatedToolOutput", out) } else { json!({}) }
        }
        "PreToolUse" => {
            let mut input = event["tool_input"].clone();
            if map_strings(&mut input, &[], &mut |s| Ok(vault.restore(s, false)))? { specific(name, "updatedInput", input) } else { json!({}) }
        }
        "MessageDisplay" => {
            let delta = event["delta"].as_str().unwrap_or("");
            let shown = vault.restore(delta, false);
            if shown != delta { specific(name, "displayContent", json!(shown)) } else { json!({}) }
        }
        // a hook cannot change the output of a failed call; the most it can do is say what was sent
        "PostToolUseFailure" => match vault.count(event["error"].as_str().unwrap_or(""))? {
            0 => json!({}),
            sent => {
                let tool = event["tool_name"].as_str().unwrap_or("tool");
                eprintln!("hook PostToolUseFailure {tool}: {sent} private values sent as is");
                json!({"systemMessage": format!("holdback: the output of a failed {tool} call held {sent} private values and was sent as is. Claude Code does not let a hook change the output of a failed call.")})
            }
        },
        "SessionStart" => {
            let mut out = specific(name, "additionalContext", json!(NOTE));
            out["systemMessage"] = json!("holdback: on, in hook mode. Private values that tools return are replaced before the model sees them. What you type and the system prompt are sent as is.");
            out
        }
        _ => json!({}),
    })
}

/// The answer when the local process cannot be reached: no tool runs, because its output could not be checked.
/// If a tool has already run, its output is withheld. The response keeps its shape, because Claude Code drops
/// a replacement that does not fit the tool's own output format and would send the original.
pub fn unavailable(event: &Value) -> Value {
    let name = event["hook_event_name"].as_str().unwrap_or("");
    match name {
        "PreToolUse" => json!({"hookSpecificOutput": {
            "hookEventName": name,
            "permissionDecision": "deny",
            "permissionDecisionReason": "holdback hook mode is switched on but its local process is not running, so tool output cannot be checked. Run `holdback serve` to see why, or switch hook mode off.",
        }}),
        "PostToolUse" => {
            let mut out = event["tool_response"].clone();
            let _ = map_strings(&mut out, &["type", "base64"], &mut |_| Ok("[withheld: holdback's local process is not running]".into()));
            specific(name, "updatedToolOutput", out)
        }
        "SessionStart" => json!({"systemMessage": "holdback: hook mode is switched on but its local process did not start, so tool calls will be denied. Run `holdback serve` to see why."}),
        _ => json!({}),
    }
}
