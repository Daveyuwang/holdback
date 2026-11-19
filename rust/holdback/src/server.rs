//! Local proxy: /<upstream name>/<provider path> -> upstream base URL + <provider path>.
//! Requests to model endpoints are redacted before they leave; responses and streams are restored before the
//! agent sees them. Everything else passes through. If a request cannot be redacted, it is not sent.
use crate::protocols::{Kind, StreamRewriter, redact_request, restore_obj};
use crate::vault::Vault;
use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::{any, get},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DEFAULT_UPSTREAMS: &[(&str, &str)] = &[
    ("anthropic", "https://api.anthropic.com"),
    ("openai", "https://api.openai.com"),
    ("deepseek", "https://api.deepseek.com"),
    ("deepseek-anthropic", "https://api.deepseek.com/anthropic"),
];
const DROP_REQ: &[&str] = &["host", "content-length", "accept-encoding", "connection"];
const DROP_RESP: &[&str] = &["content-length", "content-encoding", "transfer-encoding", "connection"];

pub fn upstreams() -> HashMap<String, String> {
    let mut ups: HashMap<String, String> = DEFAULT_UPSTREAMS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    if let Ok(extra) = std::env::var("HOLDBACK_UPSTREAMS") {
        for item in extra.split(',').filter(|s| !s.trim().is_empty()) {
            if let Some((name, url)) = item.split_once('=') {
                ups.insert(name.trim().to_string(), url.trim().trim_end_matches('/').to_string());
            }
        }
    }
    ups
}

struct AppState {
    vault: Arc<Mutex<Vault>>,
    upstreams: HashMap<String, String>,
    client: reqwest::Client,
    usage: Mutex<HashMap<String, usize>>,
    recent: Mutex<VecDeque<Value>>,
    dump: Option<String>,
    store: std::path::PathBuf,
}

fn error(kind: Option<Kind>, status: StatusCode, message: &str) -> Response {
    let body = if kind == Some(Kind::Anthropic) {
        json!({"type": "error", "error": {"type": "api_error", "message": message}})
    } else {
        json!({"error": {"message": message, "type": "holdback_error"}})
    };
    (status, [("content-type", "application/json")], body.to_string()).into_response()
}

pub fn app(vault: Vault, store: std::path::PathBuf) -> Router {
    let state = Arc::new(AppState {
        vault: Arc::new(Mutex::new(vault)),
        upstreams: upstreams(),
        client: reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(30)).build().expect("http client"),
        usage: Mutex::new(HashMap::new()),
        recent: Mutex::new(VecDeque::new()),
        dump: std::env::var("HOLDBACK_DUMP").ok(),
        store,
    });
    Router::new()
        .route("/_holdback", get(page))
        .route("/_holdback/status", get(status))
        .route("/_holdback/review", get(review))
        .route("/_holdback/hook", axum::routing::post(hook))
        .fallback(any(proxy))
        .with_state(state)
}

async fn status(State(st): State<Arc<AppState>>) -> Response {
    let (distinct, retired, detector) = {
        let v = st.vault.lock().unwrap();
        (v.distinct_values(), v.retired(), v.detector_name())
    };
    let recent: Vec<Value> = st.recent.lock().unwrap().iter().take(50).cloned().collect();
    axum::Json(json!({"version": crate::setup::VERSION, "detector": detector, "retired_values": retired, "upstreams": st.upstreams, "replaced_by_type": *st.usage.lock().unwrap(), "distinct_values": distinct, "recent": recent}))
        .into_response()
}

/// True for a request addressed to this machine by its local name, which a web page that points its own domain
/// at 127.0.0.1 cannot send. The pages that handle real values answer only these.
fn local(headers: &HeaderMap) -> bool {
    let host = headers.get("host").and_then(|h| h.to_str().ok()).unwrap_or("");
    host.starts_with("127.0.0.1:") || host.starts_with("localhost:")
}

/// Real values, for `holdback review`.
async fn review(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !local(&headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    axum::Json(st.vault.lock().unwrap().review()).into_response()
}

/// One Claude Code hook event in, the hook's answer out (see hooks.rs). From the first event on, the
/// placeholder map is saved, because hook-mode placeholders stay in the conversation.
async fn hook(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if !local(&headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let Ok(event) = serde_json::from_slice::<Value>(&body) else { return StatusCode::BAD_REQUEST.into_response() };
    let (state, t0) = (st.clone(), Instant::now());
    let answered = tokio::task::spawn_blocking(move || {
        let mut v = state.vault.lock().unwrap();
        v.keep_in(state.store.clone())?;
        v.refresh();
        let (before, known) = (v.stats.clone(), v.distinct_values());
        let answer = crate::hooks::answer(&mut v, &event)?;
        v.save()?;
        let what = format!("{} {}", event["hook_event_name"].as_str().unwrap_or(""), event["tool_name"].as_str().unwrap_or("")).trim().to_string();
        anyhow::Ok((answer, what, replaced_since(&v, &before), v.distinct_values() - known))
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r);
    match answered {
        Ok((answer, what, replaced, new_values)) => {
            if !replaced.is_empty() {
                record(&st, "hook", &what, &replaced, new_values, t0.elapsed().as_millis());
            }
            axum::Json(answer).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Replacements by label since `before`.
fn replaced_since(vault: &Vault, before: &HashMap<String, usize>) -> HashMap<String, usize> {
    vault.stats.iter().map(|(l, n)| (l.clone(), n - before.get(l).copied().unwrap_or(0))).filter(|(_, d)| *d > 0).collect()
}

/// Adds one redaction to the log, the totals and the status page.
fn record(st: &AppState, upstream: &str, what: &str, replaced: &HashMap<String, usize>, new_values: usize, ms: u128) {
    let total: usize = replaced.values().sum();
    eprintln!("{upstream} {what}: replaced {total} ({}), {new_values} new, {ms} ms", counts(replaced.iter().map(|(l, n)| (l.as_str(), *n))));
    for (l, n) in replaced {
        *st.usage.lock().unwrap().entry(l.clone()).or_default() += n;
    }
    let mut recent = st.recent.lock().unwrap();
    recent.push_front(json!({"time": clock(), "upstream": upstream, "protocol": what, "replaced": replaced, "new_values": new_values, "ms": ms}));
    recent.truncate(200);
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// "label n, label n" or "none".
fn counts<'a>(items: impl Iterator<Item = (&'a str, usize)>) -> String {
    let parts: Vec<String> = items.map(|(k, v)| format!("{k} {v}")).collect();
    if parts.is_empty() { "none".into() } else { parts.join(", ") }
}

async fn page(State(st): State<Arc<AppState>>) -> Html<String> {
    let distinct = st.vault.lock().unwrap().distinct_values();
    let mut totals: Vec<(String, usize)> = st.usage.lock().unwrap().iter().map(|(k, v)| (k.clone(), *v)).collect();
    totals.sort_by_key(|t| std::cmp::Reverse(t.1));
    let totals = esc(&counts(totals.iter().map(|(k, v)| (k.as_str(), *v))));
    let rows: String = st
        .recent
        .lock()
        .unwrap()
        .iter()
        .map(|r| {
            let replaced = counts(r["replaced"].as_object().into_iter().flatten().map(|(k, v)| (k.as_str(), v.as_u64().unwrap_or(0) as usize)));
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} ms</td></tr>",
                r["time"].as_str().unwrap_or(""),
                esc(r["upstream"].as_str().unwrap_or("")),
                r["protocol"].as_str().unwrap_or(""),
                esc(&replaced),
                r["new_values"],
                r["ms"]
            )
        })
        .collect();
    let rows = if rows.is_empty() { "<tr><td colspan=6>No model requests yet.</td></tr>".to_string() } else { rows };
    Html(format!(
        r#"<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Holdback</title>
<style>body{{font:14px -apple-system,system-ui,sans-serif;margin:32px;color:#222;background:#fff}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:6px 10px;border-bottom:1px solid #e5e5e5}}
th{{color:#666;font-weight:500}}@media(prefers-color-scheme:dark){{body{{background:#161616;color:#ddd}}td,th{{border-color:#333}}}}</style>
<h1>Holdback</h1><p>{distinct} private values kept on this machine. Replaced so far: {totals}.</p>
<p>Original values are never shown here. This page refreshes every 5 seconds.</p>
<table><tr><th>Time</th><th>Upstream</th><th>Protocol</th><th>Replaced in request</th><th>New values</th><th>Added time</th></tr>{rows}</table>"#
    ))
}

/// Redacts one request body. Returns the new body, replacements by label, and how many values are new.
fn redact_body(vault: &Mutex<Vault>, kind: Kind, raw: &[u8]) -> anyhow::Result<(Vec<u8>, HashMap<String, usize>, usize)> {
    let data: Value = serde_json::from_slice(raw)?;
    let mut v = vault.lock().unwrap();
    v.refresh();
    let (before, known) = (v.stats.clone(), v.distinct_values());
    let redacted = redact_request(kind, data, &mut v)?;
    Ok((serde_json::to_vec(&redacted)?, replaced_since(&v, &before), v.distinct_values() - known))
}

/// Appends a redacted request body to the dump file (HOLDBACK_DUMP), one per line, for leak checks.
fn dump(path: &str, body: &[u8]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(body).and_then(|_| f.write_all(b"\n"));
    }
}

/// Wall-clock time of day in UTC, for the status page.
fn clock() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02} UTC", secs / 3600 % 24, secs / 60 % 60, secs % 60)
}

async fn proxy(State(st): State<Arc<AppState>>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let path = uri.path().trim_start_matches('/');
    let (name, rest) = path.split_once('/').unwrap_or((path, ""));
    let Some(base) = st.upstreams.get(name) else {
        let mut known: Vec<&String> = st.upstreams.keys().collect();
        known.sort();
        return error(None, StatusCode::NOT_FOUND, &format!("unknown upstream '{name}'; known: {}", known.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    };
    let url = format!("{base}/{rest}{}", uri.query().map(|q| format!("?{q}")).unwrap_or_default());
    let kind = if method == Method::POST { Kind::of(&format!("/{rest}")) } else { None };

    let mut body = body;
    if let Some(k) = kind.filter(|_| !body.is_empty()) {
        let (vault, raw, t0) = (st.vault.clone(), body.clone(), Instant::now());
        let redacted = tokio::task::spawn_blocking(move || redact_body(&vault, k, &raw)).await.map_err(anyhow::Error::from).and_then(|r| r);
        let (new_body, replaced, new_values) = match redacted {
            Ok(r) => r,
            Err(e) => return error(kind, StatusCode::BAD_GATEWAY, &format!("holdback could not redact this request ({e}); nothing was sent")),
        };
        record(&st, name, k.name(), &replaced, new_values, t0.elapsed().as_millis());
        if let Some(path) = &st.dump {
            dump(path, &new_body);
        }
        body = Bytes::from(new_body);
    }

    let mut req = st.client.request(method, &url).header("accept-encoding", "identity").body(body);
    for (k, v) in headers.iter() {
        if !DROP_REQ.contains(&k.as_str()) {
            req = req.header(k, v);
        }
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return error(kind, StatusCode::BAD_GATEWAY, &format!("holdback could not reach {base} ({e})")),
    };
    let status = resp.status();
    let mut out_headers = HeaderMap::new();
    for (k, v) in resp.headers() {
        if !DROP_RESP.contains(&k.as_str())
            && let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_str().as_bytes()), HeaderValue::from_bytes(v.as_bytes()))
        {
            out_headers.append(k, v);
        }
    }
    let is_sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("text/event-stream"));

    let Some(kind) = kind.filter(|_| status.as_u16() < 400) else {
        let bytes = resp.bytes().await.unwrap_or_default();
        return (status, out_headers, bytes).into_response();
    };
    if !is_sse {
        let bytes = resp.bytes().await.unwrap_or_default();
        let bytes = match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => {
                let restored = restore_obj(v, &st.vault.lock().unwrap(), None);
                Bytes::from(serde_json::to_vec(&restored).unwrap_or_default())
            }
            Err(_) => bytes,
        };
        return (status, out_headers, bytes).into_response();
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(64);
    let vault = st.vault.clone();
    tokio::spawn(async move {
        let mut upstream = resp.bytes_stream();
        let mut rewriter = StreamRewriter::new(kind);
        let (mut buf, mut event, mut data): (Vec<u8>, Option<String>, Vec<String>) = (vec![], None, vec![]);
        while let Some(Ok(chunk)) = upstream.next().await {
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line).trim_end_matches(['\n', '\r']).to_string();
                if let Some(e) = line.strip_prefix("event:") {
                    event = Some(e.trim().to_string());
                } else if let Some(d) = line.strip_prefix("data:") {
                    data.push(d.trim_start().to_string());
                } else if line.is_empty() && !data.is_empty() {
                    let raw = data.join("\n");
                    let out = match serde_json::from_str::<Value>(&raw) {
                        Ok(v) => {
                            let events = rewriter.feed(event.clone(), v, &vault.lock().unwrap());
                            events
                                .into_iter()
                                .map(|(ev, d)| format!("{}data: {}\n\n", ev.map(|e| format!("event: {e}\n")).unwrap_or_default(), d))
                                .collect::<String>()
                        }
                        Err(_) => format!("{}data: {raw}\n\n", event.clone().map(|e| format!("event: {e}\n")).unwrap_or_default()),
                    };
                    if tx.send(Bytes::from(out)).await.is_err() {
                        return;
                    }
                    event = None;
                    data.clear();
                } else if line.is_empty() {
                    event = None;
                }
            }
        }
    });
    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<_, std::io::Error>(b), rx)) });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = out_headers;
    response
}
