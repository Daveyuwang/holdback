"""False replacements on developer output: text with nothing private in it, where every replacement is a mistake.

The corpus is this repository's source code, its lockfiles and manifests, and the output of common commands,
with the home directory masked. Files that hold made-up names or keys for other checks are left out. Made-up
private values are planted at known places, so the same run shows that fixing false replacements did not start
leaking real ones.

Usage: uv run python bench/dev_eval.py [--show] [--held-out]
--show lists each false replacement. The source and lockfile parts are the same on every machine; the command
output depends on the machine and is reported separately.

The rules were tuned on this corpus. --held-out runs a second one they were not tuned on: the source of three
crates from cargo's registry, an npm lockfile, and other commands. It needs `cargo build` to have fetched the
crates and `npm install` in bench/rehydra for the lockfile; missing parts are skipped."""
import argparse
import glob
import subprocess
from collections import Counter
from pathlib import Path

from check_agents import STRIPE
from spans import spans

ROOT = Path(__file__).parent.parent
SOURCE = ["rust/holdback/src/native.rs", "rust/holdback/src/detect.rs", "rust/holdback/src/server.rs", "rust/holdback/src/protocols.rs",
          "rust/holdback/src/main.rs", "rust/holdback/src/setup.rs", "rust/holdback/src/hooks.rs", "rust/holdback/src/vault.rs", "install.sh",
          "plugins/claude-code/hooks/hook.sh", "plugins/claude-code/hooks/session_start.sh", "bench/spans.py", "bench/detect_eval.py"]
LOCKS = ["rust/holdback/Cargo.lock", "uv.lock", "rust/holdback/Cargo.toml", "pyproject.toml"]
PINNED = "9e6c174"  # git commands read history up to this commit, so the corpus does not change with every new commit
COMMANDS = [f"git log --oneline -60 {PINNED}", f"git log --stat -5 --format='%h %s' {PINNED}", "cargo tree --manifest-path rust/holdback/Cargo.toml | head -150",
            "ls -la /usr/bin | head -120", "rustc --print target-list | head -120", f"git ls-tree -r --name-only {PINNED} | head -80"]
PLANTED = [("Maria Gonzalez", "maria.gonzalez@gmail.com", STRIPE), ("Daniel Okafor", "dokafor88@outlook.com", "ghp_R4nd0mEXAMPLEt0k3nR4nd0mEXAMPLEt0k3n"),
           ("Priya Raman", "priya.raman@proton.me", "S3cr3t-Pa55w0rd!")]
HELD_OUT_CRATES = ["memmap2", "bytes", "anyhow"]
HELD_OUT_COMMANDS = ["cargo --list", "df -h", "git branch -a", "ls -la /usr/lib | head -80", "env | cut -d= -f1 | sort | head -80", "uv tree 2>/dev/null | head -120",
                     "ps -axo pid,comm | head -80"]
CHUNK, LIMIT = 4000, 50_000  # a tool result is rarely larger than a chunk; a lockfile is cut at the limit


def corpus(held_out: bool = False) -> list[tuple[str, str, list[str]]]:
    """(kind, text, planted values in it) per chunk."""
    home = str(Path.home())
    run = lambda c: subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True).stdout
    if held_out:
        files = [f for crate in HELD_OUT_CRATES for f in sorted(glob.glob(f"{home}/.cargo/registry/src/*/{crate}-[0-9]*/src/*.rs"))[:4]]
        parts = [("source", Path(f).read_text()[:LIMIT]) for f in files]
        lock = ROOT / "bench/rehydra/package-lock.json"
        parts += [("lockfile", lock.read_text()[:LIMIT])] if lock.exists() else []
        parts += [("command output", run(c)) for c in HELD_OUT_COMMANDS]
    else:
        parts = [("source", (ROOT / f).read_text()) for f in SOURCE] + [("lockfile", (ROOT / f).read_text()[:LIMIT]) for f in LOCKS]
        parts += [("command output", run(c)) for c in COMMANDS]
    chunks = [(kind, text[i:i + CHUNK].replace(home, "/srv/work")) for kind, text in parts for i in range(0, len(text), CHUNK)]
    out = []
    for n, (kind, text) in enumerate(chunks):
        planted = []
        if n % 6 == 0:  # one chunk in six gets a line with a name and an email, and a line with a key
            name, email, key = PLANTED[n // 6 % len(PLANTED)]
            cut = text.find("\n", len(text) // 2) + 1
            text = f"{text[:cut]}# reviewed by {name} <{email}>\nAPI_TOKEN={key}\n{text[cut:]}"
            planted = [name, email, key]
        out.append((kind, text, planted))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--held-out", action="store_true")
    a = ap.parse_args()
    chunks = corpus(a.held_out)
    per = spans([text for _, text, _ in chunks], "dev_held_out_spans" if a.held_out else "dev_spans")
    for title, key in (("detector only", "detected"), ("detector + rules", "refined")):
        false, size, labels, leaked, planted_n = Counter(), Counter(), Counter(), 0, 0
        for (kind, text, planted), row in zip(chunks, per):
            at = [(text.find(v), text.find(v) + len(v)) for v in planted]
            size[kind] += len(text)
            planted_n += len(planted)
            leaked += sum(not any(s < e2 and s2 < e for s, e, _ in row[key]) for s2, e2 in at)
            for s, e, label in row[key]:
                if not any(s < e2 and s2 < e for s2, e2 in at):
                    false[kind] += 1
                    labels[label] += 1
                    if a.show and key == "refined":
                        print(f"  {kind:14s} {label:16s} {text[s:e]!r}")
        rate = {k: f"{false[k] / (size[k] / 100_000):.0f}" for k in size}
        total = sum(false.values()) / (sum(size.values()) / 100_000)
        print(f"{title:18s} false replacements per 100 KB: {total:.0f} ({', '.join(f'{k} {v}' for k, v in rate.items())})   "
              f"planted values leaked {leaked}/{planted_n}   by label {dict(labels.most_common())}")
    print(f"corpus: {', '.join(f'{k} {v // 1000} KB' for k, v in size.items())}")


if __name__ == "__main__":
    main()
