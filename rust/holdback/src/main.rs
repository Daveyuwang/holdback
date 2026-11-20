//! holdback: keep personal data and secrets out of requests to cloud models.
mod detect;
mod hooks;
mod native;
mod precision;
mod protocols;
mod server;
mod setup;
mod student;
mod vault;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const USAGE: &str = "holdback: keeps personal data and secrets on this machine

usage:
  holdback serve [--port N]                      start the proxy (default 127.0.0.1:8787); status page at /_holdback
  holdback run claude [--upstream NAME] [args]   start the proxy if needed, then Claude Code pointed at it
  holdback run codex  [--upstream NAME] [args]   same for Codex, through a \"holdback\" model provider
  holdback status [--port N]                     what the running proxy has replaced so far
  holdback review [--port N] [--masked]          each replaced value with its context; prints the real values,
                                                 or with --masked only their shape, which is safe to share
  holdback keep VALUE...                         never replace these values; with no value, show both lists
  holdback replace VALUE...                      always replace these values, such as a name the detector misses
  holdback hook                                  answer one Claude Code hook event from standard input (hook mode)
  holdback setup claude [--proxy]                install the Claude Code plugin and switch holdback on for every
                                                 session; --proxy also routes terminal sessions through the proxy
  holdback doctor                                check the installation and say how to fix what is wrong
  holdback --version

upstreams: anthropic, openai, deepseek, deepseek-anthropic, plus HOLDBACK_UPSTREAMS=\"name=https://host/base,...\"
detector: holdback's distilled model (about 115 MB), downloaded on first start to ~/.cache/holdback;
  HOLDBACK_DETECTOR=privacy-filter runs OpenAI's Privacy Filter instead (about 0.9 GB); HF_ENDPOINT picks a mirror
hook mode: saves its placeholder map to ~/.local/state/holdback; override with HOLDBACK_STATE_DIR";

/// A detector holdback can run, downloaded on first use from Hugging Face (or the mirror in HF_ENDPOINT) at a fixed
/// revision, each file checked against its SHA-256.
pub struct Model {
    /// The value of HOLDBACK_DETECTOR that picks it.
    pub name: &'static str,
    pub title: &'static str,
    repo: &'static str,
    revision: &'static str,
    files: &'static [(&'static str, &'static str)],
    /// The variable that names another directory for it, and its directory under ~/.cache/holdback otherwise.
    dir_var: &'static str,
    cache: &'static str,
    pub size: &'static str,
}

pub static DISTILLED: Model = Model {
    name: "distilled",
    title: "holdback's distilled privacy model",
    repo: "limbo23/holdback-student",
    revision: "f1898f041baaea805f78a73d0de55eb3062f7c61",
    files: &[
        ("config.json", "cbf53393a8839374ce3944ae4ae1c43c0cb481fe24e449a80671ce3fd08c5429"),
        ("model.safetensors", "ceb665f1b44316521792f73c1a677150f5ac96c3b43c34d0716e912f491a5711"),
        ("tokenizer.json", "1e33a5df9f2b9ca818afbbce8dc1799b1d289080e30f3b6332a65bb2a2292f89"),
    ],
    dir_var: "HOLDBACK_STUDENT_DIR",
    cache: "holdback-student",
    size: "about 115 MB",
};

pub static PRIVACY_FILTER: Model = Model {
    name: "privacy-filter",
    title: "OpenAI Privacy Filter (q4)",
    repo: "openai/privacy-filter",
    revision: "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
    files: &[
        ("config.json", "b2b26a4a4a000639ad30b0c264adbefe365bdb567fbd7bb27303b8c438375bd1"),
        ("onnx/model_q4.onnx", "8f7dee8b46d096f052b359375dfba5d983cc4d18c44a783bf548615c472f8dea"),
        ("onnx/model_q4.onnx_data", "f30998e28c71c5374cc7e8b7de8f0f83e981592c0c2d652d2ad4928454dbb496"),
    ],
    dir_var: "HOLDBACK_MODEL_DIR",
    cache: "privacy-filter-q4",
    size: "about 0.9 GB",
};

impl Model {
    pub fn dir(&self) -> PathBuf {
        if let Ok(dir) = std::env::var(self.dir_var) {
            return PathBuf::from(dir);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&home).join(".cache").join("holdback").join(self.cache)
    }

    /// Its files that are not in its directory yet.
    pub fn missing(&self) -> Vec<&'static str> {
        let dir = self.dir();
        self.files.iter().map(|(f, _)| *f).filter(|f| !dir.join(f).exists()).collect()
    }
}

/// The detector HOLDBACK_DETECTOR names: the distilled model unless it says privacy-filter.
pub fn chosen() -> Result<&'static Model> {
    match std::env::var("HOLDBACK_DETECTOR").as_deref() {
        Err(_) | Ok("") | Ok("distilled") => Ok(&DISTILLED),
        Ok("privacy-filter") => Ok(&PRIVACY_FILTER),
        Ok(other) => bail!("HOLDBACK_DETECTOR is {other:?}; use distilled or privacy-filter"),
    }
}

/// Where hook mode saves the placeholder map.
fn store_file() -> PathBuf {
    let dir = std::env::var("HOLDBACK_STATE_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&home).join(".local").join("state").join("holdback")
    });
    dir.join("values.json")
}

/// Where `holdback keep` and `holdback replace` record the user's corrections.
fn lists_file() -> PathBuf {
    let dir = std::env::var("HOLDBACK_CONFIG_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&home).join(".config").join("holdback")
    });
    dir.join("lists.json")
}

/// keep / replace: add values to one list and take them off the other. The running process rereads the file
/// before its next request.
fn correct(list: &str, values: &[String]) -> Result<()> {
    let path = lists_file();
    let mut file: Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
    for key in ["keep", "replace"] {
        let mut items: Vec<String> = file[key].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).filter(|v| !values.contains(v)).collect();
        if key == list {
            items.extend(values.iter().cloned());
        }
        file[key] = json!(items);
    }
    if !values.is_empty() {
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, serde_json::to_string_pretty(&file)?)?;
    }
    println!("never replaced: {}\nalways replaced: {}\n(edit {} to remove an entry)", file["keep"], file["replace"], path.display());
    Ok(())
}

/// Downloads the model's missing files. A file is kept only when its SHA-256 matches.
async fn ensure_model(model: &Model) -> Result<()> {
    let dir = model.dir();
    let host = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let client = reqwest::Client::new();
    for file in model.missing() {
        let target = dir.join(file);
        std::fs::create_dir_all(target.parent().unwrap())?;
        let url = format!("{}/{}/resolve/{}/{file}", host.trim_end_matches('/'), model.repo, model.revision);
        let mut resp = client.get(&url).send().await?.error_for_status().with_context(|| format!("downloading {url}"))?;
        let total = resp.content_length().unwrap_or(0);
        let part = target.with_file_name(format!("{}.part", target.file_name().unwrap().to_string_lossy()));
        let mut out = tokio::fs::File::create(&part).await?;
        let mut hash = aws_lc_rs::digest::Context::new(&aws_lc_rs::digest::SHA256);
        let (mut got, mut shown) = (0u64, 0u64);
        while let Some(chunk) = resp.chunk().await? {
            tokio::io::AsyncWriteExt::write_all(&mut out, &chunk).await?;
            hash.update(&chunk);
            got += chunk.len() as u64;
            if total > 50_000_000 && got * 20 / total > shown {
                shown = got * 20 / total;
                eprintln!("  {file}: {} of {} MB", got / 1_000_000, total / 1_000_000);
            }
        }
        tokio::io::AsyncWriteExt::flush(&mut out).await?;
        let sum: String = hash.finish().as_ref().iter().map(|b| format!("{b:02x}")).collect();
        if model.files.iter().any(|&(f, want)| f == file && want != sum) {
            std::fs::remove_file(&part)?;
            bail!("{file} from {url} does not match its checksum ({got} bytes); nothing was kept, run again to retry");
        }
        std::fs::rename(&part, &target)?;
    }
    Ok(())
}

async fn load_detector() -> Result<detect::Detector> {
    let model = chosen()?;
    let dir = model.dir();
    if !model.missing().is_empty() {
        eprintln!("Downloading {} ({}) to {}", model.title, model.size, dir.display());
        ensure_model(model).await?;
    }
    eprintln!("Loading {}...", model.title);
    let loaded = match model.name == DISTILLED.name {
        true => student::Student::load(&dir).map(|s| detect::Detector::Student(Box::new(s))),
        false => detect::Detector::load(&dir),
    };
    let mut detector = loaded.with_context(|| format!("could not load {} from {}", model.title, dir.display()))?;
    detector.detect("Warm-up: Alice Example, alice@example.com")?;
    Ok(detector)
}

async fn serve(port: u16) -> Result<()> {
    let t0 = Instant::now();
    let mut vault = vault::Vault::new(load_detector().await?);
    vault.correct_from(lists_file());
    let store = store_file();
    if store.exists() {
        vault.keep_in(store.clone()).with_context(|| format!("could not read {}", store.display()))?;
    }
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.with_context(|| format!("port {port} is in use"))?;
    eprintln!("Ready on http://127.0.0.1:{port} in {:.1}s (status page: http://127.0.0.1:{port}/_holdback)", t0.elapsed().as_secs_f32());
    axum::serve(listener, server::app(vault, store)).await?;
    Ok(())
}

/// A page of the running proxy's own interface, or None when no proxy answers.
async fn ask(port: u16, page: &str) -> Option<Value> {
    let r = reqwest::Client::new().get(format!("http://127.0.0.1:{port}/_holdback/{page}")).timeout(Duration::from_secs(1)).send().await.ok()?;
    serde_json::from_slice(&r.bytes().await.ok()?).ok()
}

async fn running(port: u16) -> Option<Value> {
    ask(port, "status").await
}

/// Prints every replaced value so wrong replacements can be found. The values are real unless `masked`, which
/// shows their shape (see vault::shape).
async fn review(port: u16, masked: bool) -> Result<()> {
    let Some(rows) = ask(port, "review").await else { bail!("no proxy on port {port}; start one with `holdback serve`") };
    let rows = rows.as_array().context("unexpected answer from the proxy")?;
    let text = |r: &Value, key: &str| -> String {
        let s = r[key].as_str().unwrap_or("").replace(char::is_control, " ");
        if masked && key != "placeholder" { vault::shape(&s) } else { s }
    };
    let width = |key: &str| rows.iter().map(|r| text(r, key).chars().count()).max().unwrap_or(0).max(key.len());
    let (wp, wv) = (width("placeholder"), width("value"));
    let note = if masked { "Values are masked to their shape." } else { "These are the real values: keep this output away from models and logs." };
    println!("{} values replaced since the proxy started. {note}", rows.len());
    println!("\n{:>8}  {:>7}  {:wp$}  {:wv$}  first seen in", "detected", "matched", "placeholder", "value");
    for r in rows {
        println!("{:>8}  {:>7}  {:wp$}  {:wv$}  {}", r["detected"].as_u64().unwrap_or(0), r["matched"].as_u64().unwrap_or(0), text(r, "placeholder"), text(r, "value"), text(r, "context"));
    }
    Ok(())
}

/// Pull our own options (--port, --upstream) out of the arguments; the rest belongs to the agent.
fn take_options(args: &[String]) -> (u16, Option<String>, Vec<String>) {
    let mut port: u16 = std::env::var("HOLDBACK_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8787);
    let (mut upstream, mut rest) = (None, vec![]);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|p| p.parse().ok()).unwrap_or(port),
            "--upstream" => upstream = it.next().cloned(),
            _ if a.starts_with("--port=") => port = a[7..].parse().unwrap_or(port),
            _ if a.starts_with("--upstream=") => upstream = Some(a[11..].to_string()),
            _ => rest.push(a.clone()),
        }
    }
    if rest.first().map(String::as_str) == Some("--") {
        rest.remove(0);
    }
    (port, upstream, rest)
}

/// Starts `holdback serve` and waits until it answers. `detached` puts it in its own process group, so it
/// outlives the hook or session that started it.
async fn start_proxy(port: u16, detached: bool) -> Result<std::process::Child> {
    let log = std::env::temp_dir().join("holdback.log");
    let file = std::fs::File::create(&log)?;
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args(["serve", "--port", &port.to_string()]).stdin(Stdio::null()).stdout(file.try_clone()?).stderr(file);
    #[cfg(unix)]
    if detached {
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    }
    let mut child = cmd.spawn()?;
    let deadline = Instant::now() + Duration::from_secs(1800);
    while running(port).await.is_none() {
        if let Some(status) = child.try_wait().ok().flatten() {
            bail!("the proxy exited ({status}); see {}", log.display());
        }
        if Instant::now() > deadline {
            bail!("the proxy did not start; see {}", log.display());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(child)
}

/// hook: one Claude Code hook event on standard input, the hook's answer on standard output. Starts the local
/// process when it is not running. If it cannot be reached, tool output is withheld (see hooks::unavailable).
async fn hook(port: u16) -> Result<()> {
    let event = std::io::read_to_string(std::io::stdin())?;
    let ask = || async {
        let r = reqwest::Client::new().post(format!("http://127.0.0.1:{port}/_holdback/hook")).body(event.clone()).send().await.ok()?;
        r.status().is_success().then_some(r.text().await.ok()?)
    };
    let mut answer = ask().await;
    if answer.is_none() && running(port).await.is_none() && start_proxy(port, true).await.is_ok() {
        answer = ask().await;
    }
    println!("{}", answer.unwrap_or_else(|| hooks::unavailable(&serde_json::from_str(&event).unwrap_or(Value::Null)).to_string()));
    Ok(())
}

async fn run(agent: &str, args: &[String]) -> Result<i32> {
    let (port, upstream, rest) = take_options(args);
    let base = format!("http://127.0.0.1:{port}");
    let mut started = None;
    if running(port).await.is_none() {
        eprintln!("holdback: starting the proxy (log: {})", std::env::temp_dir().join("holdback.log").display());
        started = Some(start_proxy(port, false).await?);
    }
    let (cmd, env): (Vec<String>, Vec<(String, String)>) = match agent {
        "claude" => {
            let up = upstream.unwrap_or_else(|| "anthropic".into());
            (std::iter::once("claude".to_string()).chain(rest).collect(), vec![("ANTHROPIC_BASE_URL".into(), format!("{base}/{up}"))])
        }
        "codex" => {
            let up = upstream.unwrap_or_else(|| "openai".into());
            let key_env = std::env::var("HOLDBACK_KEY_ENV").unwrap_or_else(|_| "OPENAI_API_KEY".into());
            let provider = format!("model_providers.holdback={{name=\"holdback\",base_url=\"{base}/{up}/v1\",env_key=\"{key_env}\",wire_api=\"responses\"}}");
            let mut cmd = vec!["codex".to_string(), "-c".into(), "model_provider=holdback".into(), "-c".into(), provider];
            cmd.extend(rest);
            (cmd, vec![])
        }
        other => bail!("unknown agent '{other}' (claude or codex)"),
    };
    eprintln!("holdback: {agent} -> {base} (status: {base}/_holdback)");
    let status = Command::new(&cmd[0]).args(&cmd[1..]).envs(env).status().with_context(|| format!("`{}` is not on PATH", cmd[0]))?;
    if let Some(mut child) = started {
        let _ = child.kill();
    }
    Ok(status.code().unwrap_or(1))
}

/// spans <json>: {"texts": [...]} -> for each text, the detector's spans and the refined spans, as
/// [start, end, label] in character offsets. The benchmark scripts in bench/ read this output.
/// With "sequence": true the texts go through one placeholder map in order, as the requests of one session
/// do, and the output is what gets replaced under each setting of the known-values pass: "all" and "twice".
async fn spans(path: &str) -> Result<()> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if v["sequence"] == true {
        let mut out = vec![json!({}); v["texts"].as_array().context("texts")?.len()];
        for (key, spread) in [("all", vault::Spread::All), ("twice", vault::Spread::Twice)] {
            let mut vault = vault::Vault::new(load_detector().await?);
            vault.spread = spread;
            for (t, row) in v["texts"].as_array().unwrap().iter().zip(out.iter_mut()) {
                let text = t.as_str().context("text")?;
                let chars = |b: usize| text[..b].chars().count();
                let label = |ph: &str| ph[1..ph.rfind('_').unwrap()].to_lowercase();
                row[key] = vault.replacements(text)?.iter().map(|(s, e, ph)| json!([chars(*s), chars(*e), label(ph)])).collect();
            }
        }
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }
    let mut detector = load_detector().await?;
    let mut out = vec![];
    for t in v["texts"].as_array().context("texts")? {
        let text = t.as_str().context("text")?;
        let chars = |b: usize| text[..b].chars().count();
        let as_json = |spans: &[detect::Span]| spans.iter().map(|s| json!([chars(s.start), chars(s.end), s.label])).collect::<Vec<_>>();
        let detected = detector.detect(text)?;
        out.push(json!({"detected": as_json(&detected), "refined": as_json(&precision::refine(text, &detected))}));
    }
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => serve(take_options(&args[1..]).0).await,
        Some("run") if args.len() >= 2 => std::process::exit(run(&args[1], &args[2..]).await?),
        Some("status") => {
            let port = take_options(&args[1..]).0;
            let Some(st) = running(port).await else { bail!("no proxy on port {port}; start one with `holdback serve`") };
            println!("{} private values kept locally; replaced: {}", st["distinct_values"], st["replaced_by_type"]);
            println!("details: http://127.0.0.1:{port}/_holdback");
            Ok(())
        }
        Some("review") => {
            let (port, _, rest) = take_options(&args[1..]);
            review(port, rest.iter().any(|a| a == "--masked")).await
        }
        Some("hook") => hook(take_options(&args[1..]).0).await,
        Some("doctor") => setup::doctor(take_options(&args[1..]).0).await,
        Some("setup") if args.get(1).map(String::as_str) == Some("claude") => {
            let (port, _, rest) = take_options(&args[2..]);
            setup::setup_claude(rest.iter().any(|a| a == "--proxy"), port)
        }
        Some("--version" | "-V") => {
            println!("holdback {}", setup::VERSION);
            Ok(())
        }
        Some(list @ ("keep" | "replace")) => correct(list, &args[1..]),
        Some("spans") if args.len() == 2 => spans(&args[1]).await,
        _ => {
            println!("{USAGE}");
            Ok(())
        }
    }
}
