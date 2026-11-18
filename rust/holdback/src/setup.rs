//! `holdback doctor` and `holdback setup claude`: find out why a session is not protected, and switch
//! protection on without editing files by hand.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The oldest plugin whose hooks this binary answers. Older ones have no hook mode.
const MIN_PLUGIN: &str = "0.3.0";
/// Where Claude Code finds the plugin. HOLDBACK_MARKETPLACE overrides it, for a local checkout.
const MARKETPLACE: &str = "Daveyuwang/holdback";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
}

fn claude_dir() -> PathBuf {
    std::env::var("CLAUDE_CONFIG_DIR").map(PathBuf::from).unwrap_or_else(|_| home().join(".claude"))
}

fn read_json(path: &Path) -> Value {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

/// Standard output of `claude <args>`, or None when the command is missing or fails.
fn claude(args: &[&str]) -> Option<String> {
    let out = Command::new("claude").args(args).env_remove("CLAUDECODE").output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The holdback plugin as Claude Code lists it: one entry per place it is installed.
fn plugins() -> Vec<Value> {
    let listed: Value = claude(&["plugin", "list", "--json"]).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    listed.as_array().into_iter().flatten().filter(|p| p["id"].as_str().is_some_and(|id| id.starts_with("holdback@"))).cloned().collect()
}

fn older(version: &str, than: &str) -> bool {
    let parts = |v: &str| v.split('.').map(|n| n.parse::<u32>().unwrap_or(0)).collect::<Vec<_>>();
    parts(version) < parts(than)
}

/// doctor: what holdback needs, one line each, with the fix where something is wrong. Fails when a line
/// needs fixing, so a script can rely on it.
pub async fn doctor(port: u16) -> Result<()> {
    let mut broken = false;
    let mut say = |mark: &str, text: String| {
        broken |= mark == "FIX";
        println!("{mark:4} {text}");
    };
    let exe = std::env::current_exe()?;
    say("ok", format!("holdback {VERSION} at {}", exe.display()));

    let on_path = std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("holdback").is_file()));
    match on_path {
        true => say("ok", "the holdback command is on PATH".into()),
        false => say("FIX", format!("the holdback command is not on PATH: add {} to PATH in your shell profile", exe.parent().unwrap().display())),
    }

    let model = crate::model_dir();
    match crate::MODEL_FILES.iter().all(|f| model.join(f).exists()) {
        true => say("ok", format!("the privacy model is in {}", model.display())),
        false => say("note", "the privacy model is not downloaded yet; the first start downloads about 0.9 GB".into()),
    }

    let status = crate::ask(port, "status").await;
    match &status {
        Some(st) if st["version"] == VERSION => say("ok", format!("the local process is running on port {port}")),
        Some(st) => say("FIX", format!("the local process on port {port} is version {}, this command is {VERSION}: stop it with `pkill -f \"holdback serve\"`; the next session starts the new one", st["version"].as_str().unwrap_or("older than 0.1.0"))),
        None => say("note", format!("no local process on port {port}; a session or `holdback run` starts it")),
    }

    if claude(&["--version"]).is_none() {
        say("note", "the claude command is not on PATH, so Claude Code was not checked".into());
    } else {
        let here = std::env::current_dir()?;
        let installed = plugins();
        let active: Vec<&Value> = installed.iter().filter(|p| p["enabled"] == true && (p["scope"] == "user" || p["projectPath"].as_str().is_some_and(|d| Path::new(d) == here))).collect();
        match active.first() {
            None if installed.is_empty() => say("FIX", "the Claude Code plugin is not installed: run `holdback setup claude`".into()),
            None => say("note", "the Claude Code plugin is installed for other folders, not for this one".into()),
            Some(p) if older(p["version"].as_str().unwrap_or("0"), MIN_PLUGIN) => {
                say("FIX", format!("the Claude Code plugin is {}, older than {MIN_PLUGIN}: run `claude plugin marketplace update holdback` and `claude plugin update holdback@holdback`", p["version"]))
            }
            Some(p) => say("ok", format!("the Claude Code plugin {} is on for this folder ({} scope)", p["version"].as_str().unwrap_or("?"), p["scope"].as_str().unwrap_or("?"))),
        }
        // settings env for a session in this folder: user, then project, then local, later ones winning
        let mut env = serde_json::Map::new();
        for file in [claude_dir().join("settings.json"), here.join(".claude/settings.json"), here.join(".claude/settings.local.json")] {
            env.extend(read_json(&file)["env"].as_object().cloned().unwrap_or_default());
        }
        let hooks = !active.is_empty() && env.get("HOLDBACK_HOOKS").and_then(Value::as_str) == Some("1");
        let routed = env.get("ANTHROPIC_BASE_URL").and_then(Value::as_str).is_some_and(|u| u.starts_with(&format!("http://127.0.0.1:{port}/")));
        let unprotected = "not protected: run `holdback setup claude`";
        say(if hooks { "ok" } else { "FIX" }, format!("a Claude desktop app session in this folder: {}", if hooks { "hook mode" } else { unprotected }));
        let terminal = if routed { "proxy mode" } else if hooks { "hook mode" } else { unprotected };
        say(if routed || hooks { "ok" } else { "FIX" }, format!("a terminal session in this folder: {terminal} (`holdback run claude` always uses proxy mode)"));
    }

    let saved = read_json(&crate::store_file())["values"].as_object().map_or(0, |m| m.len());
    let lists = read_json(&crate::lists_file());
    let count = |key: &str| lists[key].as_array().map_or(0, |a| a.len());
    let retired = status.as_ref().and_then(|st| st["retired_values"].as_u64()).map_or(String::new(), |n| format!(", {n} of them no longer replaced under today's rules"));
    say("ok", format!("hook mode has saved {saved} values{retired}; you keep {} and always replace {}", count("keep"), count("replace")));
    if broken {
        bail!("some lines above need fixing");
    }
    Ok(())
}

/// setup claude: install the plugin for every project and switch holdback on in Claude Code's user settings.
/// Hook mode protects all sessions, the desktop app included; `proxy` also routes terminal sessions through
/// the proxy, which covers more.
pub fn setup_claude(proxy: bool, port: u16) -> Result<()> {
    if claude(&["--version"]).is_none() {
        bail!("the claude command is not on PATH; install Claude Code first");
    }
    if !plugins().iter().any(|p| p["scope"] == "user") {
        let source = std::env::var("HOLDBACK_MARKETPLACE").unwrap_or_else(|_| MARKETPLACE.into());
        let _ = claude(&["plugin", "marketplace", "add", &source]); // fails harmlessly when it is already added
        claude(&["plugin", "install", "holdback@holdback"]).context("`claude plugin install holdback@holdback` failed; run it yourself to see why")?;
        println!("installed the Claude Code plugin for every project");
    }
    let path = claude_dir().join("settings.json");
    let mut settings = match read_json(&path) {
        Value::Null if path.exists() => bail!("{} is not valid JSON; fix it and run this again", path.display()),
        Value::Null => json!({}),
        other => other,
    };
    let backup = path.with_extension("json.before-holdback");
    if path.exists() && !backup.exists() {
        std::fs::copy(&path, &backup)?;
    }
    settings["env"]["HOLDBACK_HOOKS"] = json!("1");
    if proxy {
        settings["env"]["ANTHROPIC_BASE_URL"] = json!(format!("http://127.0.0.1:{port}/anthropic"));
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_string_pretty(&settings)? + "\n")?;
    println!("switched on in {}: hook mode for every session{}", path.display(), if proxy { ", proxy mode for terminal sessions" } else { "" });
    println!("new sessions are protected; `holdback doctor` shows the state");
    println!("to undo: remove HOLDBACK_HOOKS{} from the env block of that file, or restore {}", if proxy { " and ANTHROPIC_BASE_URL" } else { "" }, backup.display());
    Ok(())
}
