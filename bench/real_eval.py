"""Real agent sessions on real repositories, with hook mode on and off.

Claude Code runs headless on copies of public crates that cargo has already fetched. Every task has a check a
program can verify. Each run starts from a fresh copy and, with hook mode on, a fresh saved map. Requests go to
the provider through a recording pass-through, so the check reads what was sent. Each copy gets a `.env` with
made-up secrets; everything else in it is public, maintainers' names and emails included.

Reports, per mode: tasks passed, time, tool calls and their size; with hook mode on, the planted values found in
recorded requests, the values saved, and how many of them are wrong. A saved value is right when it is a
planted secret or part of a maintainer's entry, wrong when it is a file name, a path, an identifier or an id,
and unjudged otherwise; unjudged values are listed as shapes.

Needs the `claude` command, logged in, a built holdback binary, and `cargo build` to have fetched the crates.
Usage: uv run python bench/real_eval.py [--model claude-haiku-5-5] [--jobs 3] [--only-on] [--show]"""
import argparse
import glob
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from check_agents import DB_PW, ENV, STRIPE
from spans import BINARY

PLUGIN = Path(__file__).parent.parent / "plugins/claude-code"
CRATES = ["bytes", "memmap2", "anyhow", "fancy-regex", "gemm", "tiktoken-rs", "bit-set", "itoa", "memchr"]  # small, with an authors field
SENT: dict[str, list[str]] = {}  # run id -> request bodies


class Recorder(BaseHTTPRequestHandler):
    """Passes requests on to the provider and keeps their bodies, by the run id that leads the path."""
    def do_POST(self):
        _, run, rest = self.path.split("/", 2)
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        SENT.setdefault(run, []).append(body.decode("utf-8", "replace"))
        skip = ("host", "accept-encoding", "connection", "content-length")
        conn = http.client.HTTPSConnection("api.anthropic.com", timeout=600)
        conn.request(self.command, "/" + rest, body=body, headers={k: v for k, v in self.headers.items() if k.lower() not in skip})
        resp = conn.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                self.send_header(k, v)
        self.end_headers()
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
    do_GET = do_POST

    def log_message(self, *args):
        pass


def tasks(src: Path) -> tuple[dict, list[str]]:
    """name -> (prompt, check(copy, answer)), and the crate's authors. The expected values come from the crate itself."""
    manifest = (src / "Cargo.toml").read_text()
    symbol = next(m.group(1) for f in sorted(src.glob("src/**/*.rs")) for m in [re.search(r"fn ([a-z_]{6,})[<(]", f.read_text())] if m)
    authors = re.findall(r'"([^"]+)"', re.search(r"authors\s*=\s*\[(.*?)\]", manifest, re.S).group(1))
    license_ = re.search(r'license\s*=\s*"([^"]+)"', manifest).group(1)
    where = Path(subprocess.run(f"grep -rl 'fn {symbol}' src | head -1", shell=True, cwd=src, capture_output=True, text=True).stdout.strip()).name
    unsafe = subprocess.run("grep -r unsafe src | wc -l", shell=True, cwd=src, capture_output=True, text=True).stdout.strip()
    built = subprocess.run("cargo check --quiet --offline", shell=True, cwd=src, capture_output=True).returncode == 0
    last = lambda answer: answer.strip().splitlines()[-1] if answer.strip() else ""
    file = lambda d, name: (d / name).read_text() if (d / name).exists() else ""
    return {
        "explain": (f"Find the file where the function `{symbol}` is implemented. Say in two lines what the function does, and name the file.",
                    lambda d, a: where in a),
        "maintainers": ("Read Cargo.toml. Write a file MAINTAINERS.txt with one line per entry of the `authors` field, exactly as written there, "
                        "and a last line `license: <the license field>`.",
                        lambda d, a: all(x in file(d, "MAINTAINERS.txt") for x in authors) and license_ in file(d, "MAINTAINERS.txt")),
        "build": ("Run `cargo check --quiet --offline`. Reply with exactly PASS or FAIL on the last line of your answer.",
                  lambda d, a: last(a).strip("`*. ") == ("PASS" if built else "FAIL")),
        "count": ("Run `grep -r unsafe src | wc -l`. Reply with only the number on the last line of your answer.",
                  lambda d, a: re.sub(r"\D", "", last(a)) == unsafe),
        "env": ("The database actually listens on port 5433. Fix DATABASE_URL in .env so the app can connect. Change nothing else.",
                lambda d, a: file(d, ".env") == ENV.replace(":5432/", ":5433/")),
    }, authors


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(crate: str, src: Path, name: str, prompt: str, check, hooks: bool, a, recorder_port: int) -> dict:
    run_id = f"{crate}-{name}-{'on' if hooks else 'off'}"
    work = Path(tempfile.mkdtemp(prefix="hb-real-"))
    copy, port = work / "repo", free_port()
    shutil.copytree(src, copy)
    (copy / ".env").write_text(ENV)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_", "HOLDBACK_"))}
    env |= {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{recorder_port}/{run_id}", "HOLDBACK_PORT": str(port), "HOLDBACK_STATE_DIR": str(work / "state"),
            "HOLDBACK_CONFIG_DIR": str(work / "config"), "PATH": f"{Path(BINARY).parent}{os.pathsep}{os.environ['PATH']}"}
    if hooks:
        env["HOLDBACK_HOOKS"] = "1"
    cmd = ["claude", "-p", prompt, "--plugin-dir", str(PLUGIN), "--model", a.model, "--allowedTools", "Read,Write,Edit,Bash,Grep,Glob",
           "--output-format", "stream-json", "--verbose"]
    t0 = time.time()
    out = subprocess.run(cmd, cwd=copy, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900).stdout
    seconds = time.time() - t0
    subprocess.run(["pkill", "-f", f"holdback serve --port {port}"])
    answer, calls, size = "", 0, 0
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        blocks = event.get("message", {}).get("content") if event.get("type") in ("assistant", "user") else None
        for block in blocks if isinstance(blocks, list) else []:
            calls += block.get("type") == "tool_use"
            if block.get("type") == "tool_result":
                content = block.get("content")
                size += len(content) if isinstance(content, str) else sum(len(x.get("text", "")) for x in content if isinstance(x, dict))
        if event.get("type") == "result":
            answer = str(event.get("result"))
    try:
        ok = bool(check(copy, answer))
    except OSError:
        ok = False
    saved = json.loads((work / "state/values.json").read_text())["values"] if (work / "state/values.json").exists() else {}
    sent = "\n".join(SENT.get(run_id, []))
    shutil.rmtree(work, ignore_errors=True)
    return {"id": run_id, "hooks": hooks, "ok": ok, "seconds": seconds, "calls": calls, "size": size, "saved": saved,
            "leaked": [s for s in (DB_PW, STRIPE) if s in sent], "answer": answer[-300:]}


def judge(value: str, authors: list[str]) -> str:
    """right, wrong or unjudged, for a value hook mode saved while working on a public crate."""
    if value in (DB_PW, STRIPE) or DB_PW in value or any(value in entry for entry in authors):
        return "right"
    if re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", value):
        return "right"
    looks_like_code = [r"[\w.-]+\.(rs|toml|md|lock|txt|orig|yml|json)", r".*[/\\].*", r"[a-z0-9]+([-_.][a-z0-9]+)+", r"[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z0-9_]+)+",
                       r"[0-9a-f]{12,}", r"([a-z]+_)?[0-9a-f]{8}(-[0-9a-f]{4,})+", r"[A-Z][A-Z0-9_]{2,}", r"[a-z]+[A-Z]\w*", r"\w+<.*>", r"\w+\(.*\)"]
    return "wrong" if any(re.fullmatch(p, value) for p in looks_like_code) else "unjudged"


def shape(text: str) -> str:
    return re.sub(r"[0-9]", "9", re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only-on", action="store_true", help="skip the runs with hook mode off")
    ap.add_argument("--show", action="store_true", help="list wrong and unjudged values as written; all of them are public")
    a = ap.parse_args()
    recorder = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=recorder.serve_forever, daemon=True).start()
    jobs, all_authors = [], []
    for crate in CRATES:
        found = sorted(glob.glob(f"{Path.home()}/.cargo/registry/src/*/{crate}-[0-9]*"))
        if not found:
            print(f"skipped {crate}: not in cargo's registry")
            continue
        todo, authors = tasks(Path(found[-1]))
        all_authors += authors
        jobs += [(crate, Path(found[-1]), name, prompt, check, hooks) for name, (prompt, check) in todo.items() for hooks in ((True,) if a.only_on else (True, False))]
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda j: run(*j, a, recorder.server_port), jobs))

    for hooks in (True, False):
        rows = [r for r in results if r["hooks"] == hooks]
        if not rows:
            continue
        failed = ", ".join(r["id"] for r in rows if not r["ok"]) or "none"
        print(f"hook mode {'on ' if hooks else 'off'}: {sum(r['ok'] for r in rows)}/{len(rows)} tasks pass; {sum(r['seconds'] for r in rows) / len(rows):.0f} s per task; "
              f"{sum(r['calls'] for r in rows)} tool calls returning {sum(r['size'] for r in rows) // 1000} KB; failed: {failed}")
    for r in results:
        if not r["ok"]:
            print(f"  failed {r['id']}: the agent said: {r['answer'][-200:]!r}")
    on = [r for r in results if r["hooks"]]
    verdicts = Counter()
    listed = []
    for r in on:
        for placeholder, value in r["saved"].items():
            verdict = judge(value, all_authors)
            verdicts[verdict] += 1
            if verdict != "right":
                listed.append(f"  {verdict:8s} {r['id']:24s} {placeholder.rsplit('_', 1)[0][1:]:16s} {value if a.show else shape(value)!r}")
    total = sum(verdicts.values())
    print(f"with hook mode on: planted values in recorded requests: {sum(len(r['leaked']) for r in on)}; values saved {total}: "
          f"right {verdicts['right']}, wrong {verdicts['wrong']} ({verdicts['wrong'] / max(1, total):.0%}), unjudged {verdicts['unjudged']}")
    print("\n".join(sorted(listed)))
    print(f"model {a.model}")


if __name__ == "__main__":
    main()
