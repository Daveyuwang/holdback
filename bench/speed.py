"""The proxy's own cost: time from start until it answers, memory when idle and after long inputs, and the time
redaction adds to a request. Requests go to an upstream that refuses connections, so each one measures redaction
and nothing else. The proxy gets empty state and settings folders, so this machine's saved values do not count.
Set HOLDBACK_DETECTOR=privacy-filter to measure the other detector. Memory is the process footprint on macOS and the
resident size elsewhere. Run it on an idle machine.

Usage: uv run python bench/speed.py"""
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from check_hooks import free_port
from spans import BINARY

REPO = Path(__file__).parent.parent
DETECTOR = ("HOLDBACK_DETECTOR", "HOLDBACK_STUDENT_DIR", "HOLDBACK_MODEL_DIR")


def memory(pid: int) -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
        return " ".join(re.search(r"phys_footprint:\s*([\d.]+\s*[KMG]B)", out).group(1).split())
    kb = int(re.search(r"VmRSS:\s*(\d+)", Path(f"/proc/{pid}/status").read_text()).group(1))
    return f"{kb // 1024} MB"


def start(env: dict, port: int, log: Path) -> tuple[subprocess.Popen, float]:
    t0 = time.time()
    proc = subprocess.Popen([BINARY, "serve", "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=log.open("w"))
    while proc.poll() is None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_holdback/status", timeout=1)
            return proc, time.time() - t0
        except OSError:
            time.sleep(0.005)
    sys.exit(f"holdback serve exited:\n{log.read_text()}")


def send(port: int, text: str) -> float:
    body = json.dumps({"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/dead/v1/messages", body, {"content-type": "application/json", "x-api-key": "k"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError:
        pass
    return time.time() - t0


def main():
    work = Path(tempfile.mkdtemp(prefix="hb-speed-"))
    env = {k: v for k, v in os.environ.items() if k in DETECTOR or not k.startswith("HOLDBACK_")}
    env |= {"HOLDBACK_UPSTREAMS": "dead=http://127.0.0.1:9", "HOLDBACK_STATE_DIR": str(work / "state"), "HOLDBACK_CONFIG_DIR": str(work / "config")}
    port, times, proc = free_port(), [], None
    try:
        for _ in range(3):
            if proc:
                proc.kill()
                proc.wait()
            proc, took = start(env, port, work / "serve.log")
            times.append(took)
        print(f"start until the status page answers: {statistics.median(times):.2f} s (median of 3)")
        time.sleep(1)
        print(f"memory when idle: {memory(proc.pid)}")
        names = ["Maria Gonzalez", "Daniel Okafor", "Priya Raman", "Hannah Becker", "Oliver Clarke", "Sofia Rossi", "Lucas Novak", "Aisha Dubois"]
        requests = [f"Please look at deploy.log. {n} ({n.split()[0].lower()}@example.com) saw the error at 14:{i:02d} on port 54{i:02d} after "
                    f"commit 3f9a{i:03d}c. " + "The service restarted twice and the queue grew to 1,200 jobs. " * 12 for i, n in enumerate(names * 3)]
        added = statistics.median(send(port, r) for r in requests)
        print(f"a request of {len(requests[0])} characters, new text each time: {added * 1000:.0f} ms (median of {len(requests)})")
        source = "".join(f.read_text() for f in sorted((REPO / "rust/holdback/src").glob("*.rs")))
        print(f"48,000 characters of source, first time: {send(port, source[:48_000]):.2f} s")
        file13 = source[60_000:73_000]
        send(port, file13)
        print(f"13 KB of source after a one-line edit: {send(port, file13.replace(chr(10), chr(10) + '// edited' + chr(10), 1)) * 1000:.0f} ms")
        print(f"memory after these inputs: {memory(proc.pid)}")
    finally:
        if proc:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
