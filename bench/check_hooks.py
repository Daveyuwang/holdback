"""Hook mode with a real Claude Code session: private values that tools return must not reach the provider.

Claude Code runs headless with the plugin in hook mode on a small repo with planted values. Its requests go to
the provider through a recording pass-through, so the check reads exactly what was sent. Values are synthetic.

Needs the `claude` command, logged in, and a built holdback binary. Usage:
  uv run python bench/check_hooks.py [--model claude-haiku-5-5] [--desktop]
--desktop sets the entrypoint the Claude desktop app uses.

Three checks: the task succeeds and no planted value is sent; after the local process restarts, a resumed
session still gets the real values in its tool calls; with a holdback command that cannot answer, no tool runs."""
import argparse
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from check_agents import PEOPLE, SECRETS, STRIPE, workspace
from spans import BINARY

PLUGIN = Path(__file__).parent.parent / "plugins/claude-code"
SENT = []  # every request body Claude Code sent to the provider
LINES = [f"Hello {n} <{e}>" for n, e in PEOPLE] + [f"STRIPE={STRIPE}"]
WRITE = ("Read customers.csv and .env. Create welcome.txt with one line per customer, formatted as: Hello NAME <EMAIL>, "
         "and a last line: STRIPE=<the value of STRIPE_SECRET_KEY>.")
COPY = "Create copy.txt with exactly the content you wrote to welcome.txt. Use the Write tool and what you already know; read no file."
READ = "Read .env and tell me the database host."


class Recorder(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        SENT.append(body.decode("utf-8", "replace"))
        skip = ("host", "accept-encoding", "connection", "content-length")
        conn = http.client.HTTPSConnection("api.anthropic.com", timeout=600)
        conn.request(self.command, self.path, body=body, headers={k: v for k, v in self.headers.items() if k.lower() not in skip})
        resp = conn.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                self.send_header(k, v)
        self.end_headers()  # HTTP/1.0: the body ends when the connection closes, which suits a stream
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
    do_GET = do_POST

    def log_message(self, *args):
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def claude(d: Path, prompt: str, env: dict, model: str, resume: str | None = None) -> dict:
    cmd = ["claude", "-p", prompt, "--plugin-dir", str(PLUGIN), "--model", model, "--allowedTools", "Read,Write,Edit,Bash", "--output-format", "json"]
    p = subprocess.run(cmd + (["--resume", resume] if resume else []), cwd=d, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"result": (p.stdout or p.stderr)[-400:]}


def report(name: str, ok: bool, start: int, said: str) -> bool:
    sent = "\n".join(SENT[start:])
    leaks = [s for s in SECRETS if s in sent]
    print(f"{'PASS' if ok and not leaks else 'FAIL'}  {name}: check {'ok' if ok else 'FAILED'}; requests {len(SENT) - start}; leaked {leaks or 'none'}")
    if not ok:
        print("   agent said:", said[-400:].replace("\n", " "))
    return ok and not leaks


def has_lines(path: Path) -> bool:
    return path.exists() and all(line in path.read_text() for line in LINES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--desktop", action="store_true")
    a = ap.parse_args()
    recorder = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=recorder.serve_forever, daemon=True).start()
    port, state = free_port(), tempfile.mkdtemp(prefix="hb-state-")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}
    env |= {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{recorder.server_port}", "HOLDBACK_HOOKS": "1", "HOLDBACK_PORT": str(port),
            "HOLDBACK_STATE_DIR": state, "PATH": f"{Path(BINARY).parent}{os.pathsep}{os.environ['PATH']}"}
    if a.desktop:
        env["CLAUDE_CODE_ENTRYPOINT"] = "claude-desktop"

    def stop_local():
        subprocess.run(["pkill", "-f", f"holdback serve --port {port}"])
        time.sleep(0.5)

    d, t0 = workspace({}), time.time()
    out = claude(d, WRITE, env, a.model)
    ok_all = report("values in a file the agent writes", has_lines(d / "welcome.txt"), 0, str(out.get("result")))

    stop_local()
    start = len(SENT)
    out = claude(d, COPY, env, a.model, resume=out.get("session_id"))
    ok_all &= report("resumed after the local process restarted", has_lines(d / "copy.txt"), start, str(out.get("result")))
    stop_local()

    # a holdback command that prints help text, as a build from before hook mode does
    old = Path(tempfile.mkdtemp(prefix="hb-old-"))
    (old / "holdback").write_text("#!/bin/sh\necho 'usage: holdback serve'\n")
    (old / "holdback").chmod(0o755)
    d2, start = workspace({}), len(SENT)
    out = claude(d2, READ, env | {"PATH": f"{old}{os.pathsep}{os.environ['PATH']}"}, a.model)
    ok_all &= report("no tool runs when holdback cannot answer", "db.internal.acme.io" not in str(out.get("result")), start, str(out.get("result")))

    print(f"model {a.model}{', desktop entrypoint' if a.desktop else ''}; {time.time() - t0:.0f} s")
    for path in (d, d2, old, state):
        shutil.rmtree(path, ignore_errors=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
