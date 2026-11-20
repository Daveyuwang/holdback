"""False replacements on developer output: text with nothing private in it, where every replacement is a mistake.

The corpus is this repository's source code, its lockfiles and manifests, and the output of common commands,
with the home directory masked. Files that hold made-up names or keys for other checks are left out. Made-up
private values are planted at known places, so the same run shows that fixing false replacements did not start
leaking real ones.

Usage: uv run python bench/dev_eval.py [--show] [--held-out | --authors]
--show lists each false replacement. The source and lockfile parts are the same on every machine; the command
output depends on the machine and is reported separately.

The rules were tuned on this corpus. --held-out runs a second one they were not tuned on: the source of three
crates from cargo's registry, an npm lockfile, and other commands. It needs `cargo build` to have fetched the
crates and `npm install` in bench/rehydra for the lockfile; missing parts are skipped.

--authors measures the opposite mistake: how many of the maintainers' names and emails in the authors lists of the
crates bench/real_eval.py works on holdback replaces.

distill/data.py reads two things from here: people(), which labels real people in its developer text, and the crate
lists, whose crates it leaves out. A change to either changes the next model's training data."""
import argparse
import glob
import re
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
CRATES = ["bytes", "memmap2", "anyhow", "fancy-regex", "gemm", "tiktoken-rs", "bit-set", "itoa", "memchr"]  # bench/real_eval.py's: small, with an authors field
HELD_OUT_CRATES = ["memmap2", "bytes", "anyhow"]
HELD_OUT_COMMANDS = ["cargo --list", "df -h", "git branch -a", "ls -la /usr/lib | head -80", "env | cut -d= -f1 | sort | head -80", "uv tree 2>/dev/null | head -120",
                     "ps -axo pid,comm | head -80"]
CHUNK, LIMIT = 4000, 50_000  # a tool result is rarely larger than a chunk; a lockfile is cut at the limit


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
SERVICE = re.compile(r"(?i)(no-?reply|do-?not-?reply)@")
NOT_A_HOST = re.compile(r"\d+(\.\d+)+\.|\d+x\.")  # `lib@1.2.0.tgz`, `icon@2x.png`
WORDS = r"[^\W\d_][\w'’.-]*(?: ['’]?[^\W\d_][\w'’.-]*){0,3}"  # one to four words, as in `Jane 'jd' Doe`
PAIR = re.compile(rf"({WORDS}) <({EMAIL.pattern})>")
FIELD = re.compile(r"""(?mi)^[\s#/*;!-]*(?:@(?:author|maintainer)\b|["']?(?:authors?|maintainers?|__author__|__maintainer__)["']?\s*[=:])"""
                   r"""[ \t]*(\[[^\]]*\]|"[^"\n]*"|'[^'\n]*'|[^\n]+)""")
QUOTED = re.compile(r""""([^"\n]+)"|'([^'\n]+)'""")
COPYRIGHT = re.compile(rf"(?i)copyright\b[\s:=\"']*(?:\(c\)|©)?(?:\s*\d{{4}}(?:\s*[-–,]\s*(?:\d{{4}}|present))*)?,?\s+(?:by\s+)?({WORDS})")
NOT_PEOPLE = {"the", "authors", "author", "contributors", "developers", "maintainers", "project", "team", "foundation", "inc",
              "llc", "ltd", "limited", "corp", "corporation", "company", "gmbh", "ag", "ab", "as", "oy", "sa", "bv", "nv", "plc",
              "org", "organization", "community", "group", "software", "systems", "labs", "networks", "technologies", "vpn",
              "university", "institute", "association", "others", "admin", "support", "info", "contact", "security", "help",
              "sales", "webmaster", "root", "postmaster", "bot", "owner", "owners", "holder", "holders", "or", "license",
              "rights", "reserved", "notice", "hello", "hi", "dear", "thanks"}
PARTICLES = {"van", "von", "der", "den", "de", "da", "di", "du", "del", "della", "la", "le", "dos", "das", "bin", "al", "el", "zu", "ter"}


def name_word(w: str) -> bool:
    """A word that can be part of a name: a letter first that is not lowercase (so a CJK name counts), a particle
    such as `van` or `d'Arc`, or a quoted nickname."""
    return (w[0].isalpha() and not w[0].islower()) or w.lower() in PARTICLES or w.startswith(("d'", "d’", "'", "’"))


def is_email(text: str, m: re.Match) -> bool:
    """Whether a match of EMAIL is someone's address, not `user:password@host`, `git@host:path`, a file such as
    `lib@1.2.0.tgz`, code such as `x@self.weight.T` or a no-reply address."""
    before, host = text[:m.start()], m.group().split("@", 1)[1]
    if before.endswith((":", "/")) and not before.lower().endswith(("mailto:", "email:")):
        return False
    tld = host.rsplit(".", 1)[1]
    return text[m.end():m.end() + 1] != ":" and not NOT_A_HOST.match(host) and len(tld) >= 2 and tld.isalpha() and not SERVICE.match(m.group())


def people(text: str) -> list[list]:
    """Real people's names and emails in developer text, as [start, end, label]: every email but no-reply addresses,
    the name in `Name <email>`, person names in author and maintainer fields (quoted, unquoted or in a comment), and
    copyright holders. Names of organizations and project teams are left out, and so is a name without an email
    unless it has two to four capitalized words."""
    out = []

    def name(start: int, value: str, alone: bool):
        words = value.split(" ")
        if not all(words):  # irregular spacing: not a name
            return
        if alone:  # the leading run, to a sentence end: `Jane Doe and contributors`, `Jane Doe. All rights` -> `Jane Doe`
            run = 0
            for w in words:
                if not name_word(w):
                    break
                run += 1
                if run > 1 and w.endswith(".") and len(w) > 2:
                    break
            words = words[:run]
        elif len(words) > 1:  # the trailing run without lead-ins: `reviewed by Jane Doe`, `Author Jane Doe` -> `Jane Doe`
            first = len(words)
            while first and name_word(words[first - 1]):
                first -= 1
            while first < len(words) and words[first].lower() in NOT_PEOPLE:
                first += 1
            start += sum(len(w) + 1 for w in words[:first])
            words = words[first:]
        value = " ".join(words).rstrip(".,")  # a single word before an email stays: a handle such as `jdoe <...>`
        if not value or any(w.lower().strip(".,") in NOT_PEOPLE for w in words):
            return
        if alone and not (2 <= len(words) <= 4 and value[0].isupper() and not value.isupper()):  # all capitals: license text
            return
        if not any(s < start + len(value) and start < e for s, e, _ in out):
            out.append([start, start + len(value), "private_person"])

    for m in PAIR.finditer(text):
        if not SERVICE.match(m.group(2)):
            name(m.start(1), m.group(1), alone=False)
    for m in FIELD.finditer(text):
        value, at = m.group(1), m.start(1)
        items = [(at + q.start(q.lastindex), q.group(q.lastindex)) for q in QUOTED.finditer(value)] if value[0] in "[\"'" else [(at, value)]
        for start, item in items:
            for part in re.finditer(r"[^,;&]+", item):  # `Jane Doe, John Roe`
                piece = part.group()
                if piece.strip():
                    name(start + part.start() + len(piece) - len(piece.lstrip()), piece.strip(), alone=True)
    for m in COPYRIGHT.finditer(text):
        name(m.start(1), m.group(1), alone=True)
    for m in EMAIL.finditer(text):
        if is_email(text, m):
            out.append([m.start(), m.end(), "private_email"])
    return sorted(out)


def source(crate: str) -> Path | None:
    """The newest version of a crate in cargo's registry."""
    found = glob.glob(f"{Path.home()}/.cargo/registry/src/*/{crate}-[0-9]*")
    version = lambda p: [(0, int(x)) if x.isdigit() else (1, x) for x in re.split(r"[.+-]", Path(p).name[len(crate) + 1:])]
    return Path(max(found, key=version)) if found else None


def maintainers(manifest: str) -> list[list]:
    """The people in a Cargo.toml's authors list, as [start, end, label]: each entry's name and email, without team
    entries such as `The Contributors` and no-reply addresses. Unlike people(), it takes every such entry, a lone
    first name or handle included."""
    field = re.search(r"authors\s*=\s*\[(.*?)\]", manifest, re.S)
    out = []
    for entry in re.finditer(r'"([^"]+)"', field.group(1) if field else ""):
        base = field.start(1) + entry.start(1)
        m = re.fullmatch(r"\s*(.*?)\s*(?:<\s*(.*?)\s*>)?\s*", entry.group(1))
        if m.group(1) and not any(w.lower() in NOT_PEOPLE for w in m.group(1).split()):
            out.append([base + m.start(1), base + m.end(1), "private_person"])
        if m.group(2) and not SERVICE.match(m.group(2)):
            out.append([base + m.start(2), base + m.end(2), "private_email"])
    return out


def authors():
    """--authors: the maintainers' names and emails replaced in full in the manifests of bench/real_eval.py's crates."""
    texts = [(s / "Cargo.toml").read_text() for s in map(source, CRATES) if s]
    per = spans(texts, "authors_spans")
    for title, key in (("detector only", "detected"), ("detector + rules", "refined")):
        found, total = Counter(), Counter()
        for text, row in zip(texts, per):
            for s, e, label in maintainers(text):
                total[label] += 1
                found[label] += all(any(s2 <= i < e2 for s2, e2, _ in row[key]) for i in range(s, e))
        print(f"{title:18s} names replaced {found['private_person']}/{total['private_person']}   "
              f"emails replaced {found['private_email']}/{total['private_email']}   in the authors lists of {len(texts)} crates")


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
    ap.add_argument("--authors", action="store_true")
    a = ap.parse_args()
    if a.authors:
        return authors()
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
