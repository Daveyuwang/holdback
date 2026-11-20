"""Training data for holdback's distilled detector, from public sources at fixed versions. None of it overlaps the
rows holdback's benchmarks read.

  natural      Nemotron-PII and PII Bench ZH rows with their dataset labels and the Privacy Filter's spans, and
               developer text with planted values -> data/natural.jsonl
  constructed  rows with exact labels: hash-heavy lockfile text, code where values of the same shape are secrets or
               not by their names, agent output, secrets as source code writes them, and Chinese documentation with
               planted values -> data/constructed.jsonl

A row is {"source", "text", "gold": [[start, end, category]], "teacher": [...]} with character offsets. The sources:
the datasets and repositories in SOURCES at fixed revisions, the crates of rust/holdback/Cargo.lock (`cargo vendor`)
and the Python packages of this environment (uv.lock). The Privacy Filter's spans come from the release binary
(`cargo build --release` in rust/holdback), which downloads its model on first use.

Usage: uv run --group distill python distill/data.py natural | constructed"""
import ast
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import sysconfig
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
REPO = HERE.parent
BINARY = os.environ.get("HOLDBACK_BIN", str(REPO / "rust/holdback/target/release/holdback"))
SOURCES = {
    # (name in the rows, dataset, revision, split, first row, rows). The benchmarks read Nemotron-PII's test split
    # and rows 0 to 399 of PII Bench ZH.
    "datasets": [("nemotron", "nvidia/Nemotron-PII", "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed", "train", 0, 16000),
                 ("zh", "asgdaudahu/pii-bench-zh", "69a37af094940c4b999599011814283dee9cf123", "train", 1000, 6000)],
    # git object listings for the lockfile rows
    "repos": {"fd": ("https://github.com/sharkdp/fd", "b80cd53e65615060a62a5ea349a0d78b65d7029e"),
              "ripgrep": ("https://github.com/BurntSushi/ripgrep", "fdf7d95485ec69c599418b7e7154baa4518fed2c"),
              "serde": ("https://github.com/serde-rs/serde", "e306b0c7c40ad21d5bdf52bcd20da513904682ae")},
    # Chinese documentation of MIT-licensed projects, only the files ZH_DOCS_FILES matches
    "zh_docs": {"ant-design": ("https://github.com/ant-design/ant-design", "ee8db7734e6a4d2387a55324e7e35eecb5535e72"),
                "hooks": ("https://github.com/alibaba/hooks", "a0c33c21caefda40753fe4b3cbb3b1efbabd1a79"),
                "umi": ("https://github.com/umijs/umi", "6d47620e2063b117c17c0a5f3ac5a79cb2b133fd"),
                "vant": ("https://github.com/youzan/vant", "7cc499279866e74b117593abf612567d5a2eb28d")},
}
ZH_DOCS_FILES = ["*.zh-CN.md", "docs/docs/**/*.md", "!*.en-US.md"]
SKIP = {"date", "time", "date_time", "occupation", "company_name", "url", "country", "county", "state", "race_ethnicity", "religious_belief",
        "employment_status", "gender", "age", "education_level", "political_view", "blood_type", "sexuality", "language", "organization",
        "title", "sex"}
EVAL_CRATES = {"memmap2", "bytes", "anyhow", "fancy-regex", "gemm", "tiktoken-rs", "bit-set", "itoa", "memchr"}  # dev_eval and real_eval read these
CHUNK = 4000
DEV_ROWS = 2000


def checkout(url: str, commit: str, path: Path, files: list[str] | None = None) -> Path:
    """One commit of a repository, without history; with `files`, only the files those patterns match."""
    if not path.exists():
        tmp = path.with_name(path.name + ".partial")
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        git = lambda *a: subprocess.run(["git", "-C", str(tmp), *a], check=True, capture_output=True)
        tmp.mkdir(parents=True)
        git("init", "-q")
        git("remote", "add", "origin", url)
        if files:
            git("sparse-checkout", "set", "--no-cone", *files)
        git("fetch", "-q", "--depth", "1", "origin", commit)
        git("checkout", "-q", "FETCH_HEAD")
        tmp.rename(path)
    return path


def crates() -> list[Path]:
    """The published source of every crate in rust/holdback/Cargo.lock, except the ones holdback's benchmarks read."""
    vendor = DATA / "crates"
    if not vendor.exists():
        tmp = DATA / "crates.partial"
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        subprocess.run(["cargo", "vendor", "--locked", "--versioned-dirs", "--manifest-path", str(REPO / "rust/holdback/Cargo.toml"), str(tmp)],
                       check=True, capture_output=True)
        tmp.rename(vendor)
    return [d for d in sorted(vendor.iterdir()) if d.is_dir() and d.name.rsplit("-", 1)[0] not in EVAL_CRATES]


def spans(texts: list[str]) -> list[dict]:
    """The Privacy Filter's spans for each text: "detected" alone, "refined" after holdback's rules."""
    env = {k: v for k, v in os.environ.items() if k != "HOLDBACK_STUDENT_DIR"} | {"HOLDBACK_DETECTOR": "privacy-filter"}  # the teacher
    with tempfile.TemporaryDirectory() as d:
        batch = Path(d) / "batch.json"
        batch.write_text(json.dumps({"texts": texts}))
        return json.loads(subprocess.run([BINARY, "spans", str(batch)], env=env, capture_output=True, text=True, check=True).stdout)


def public_rows():
    from datasets import load_dataset
    seen = Counter()
    for name, dataset, revision, split, first, n in SOURCES["datasets"]:
        key, labels = ("type", "entities") if name == "zh" else ("label", "spans")
        for r in load_dataset(dataset, split=split, revision=revision, streaming=True).skip(first).take(n):
            gold = []
            for g in ast.literal_eval(r[labels]) if isinstance(r[labels], str) else r[labels]:
                seen[g[key]] += 1
                if cat := category(g[key]):
                    gold.append([g["start"], g["end"], cat])
            if r["text"].strip():
                yield {"source": name, "text": r["text"], "gold": gold}
    print("dataset labels:", dict(seen.most_common()), file=sys.stderr)


def category(kind: str) -> str | None:
    """The detector's category for a dataset's label, or None for attributes the benchmarks do not count."""
    k = kind.lower()
    if k in SKIP:
        return None
    for words, cat in [(("email",), "private_email"), (("phone", "tel", "fax"), "private_phone"), (("birth", "dob"), "private_date"),
                       (("password", "pin", "cvv", "api_key", "token", "secret"), "secret"),
                       (("address", "street", "city", "zip", "postcode", "building", "coordinate"), "private_address"),
                       (("ip", "mac", "url"), "private_url"), (("name", "person"), "private_person")]:
        if any(w in k for w in words):
            return cat
    return "account_number"


def planted(fake, rng):
    """A few lines holding made-up private values, and those values with their categories."""
    name, email, phone = fake.name(), fake.email(), fake.phone_number()
    key = rng.choice(["ghp_", "sk-", "xoxb-", "AKIA", "glpat-", ""]) + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789", k=rng.randint(20, 40)))
    password = fake.password(length=rng.randint(10, 18))
    host = fake.domain_name()
    lines = rng.sample([
        (f"// maintainer: {name} ({email})", [(name, "private_person"), (email, "private_email")]),
        (f"Author: {name} <{email}>", [(name, "private_person"), (email, "private_email")]),
        (f'  "contact": {{"name": "{name}", "phone": "{phone}"}},', [(name, "private_person"), (phone, "private_phone")]),
        (f"export SERVICE_KEY={key}", [(key, "secret")]),
        (f'auth_token = "{key}"', [(key, "secret")]),
        (f"DATABASE_URL=postgres://svc:{password}@db.{host}:5432/main", [(password, "secret")]),
        (f"password: {password}", [(password, "secret")]),
        (f"# ask {name} before changing this", [(name, "private_person")]),
        (f"Signed-off-by: {name} <{email}>", [(name, "private_person"), (email, "private_email")]),
    ], k=rng.randint(1, 3))
    return lines


def dev_rows():
    from faker import Faker
    rng = random.Random(5)
    fakes = [Faker(loc) for loc in ("en_US", "en_GB", "de_DE", "es_ES", "fr_FR", "zh_CN", "ja_JP", "pt_BR", "it_IT", "en_IN")]
    for f in fakes:
        f.seed_instance(rng.randint(0, 10**9))
    files = []
    for d in crates():
        files += sorted(glob.glob(f"{d}/src/**/*.rs", recursive=True))[:3] + glob.glob(f"{d}/Cargo.toml") + glob.glob(f"{d}/README.md")[:1]
    site = sorted(glob.glob(f"{sysconfig.get_paths()['purelib']}/*/*.py"))
    rng.shuffle(site)
    files += site[:400]
    parts = []
    for f in files:
        try:
            parts.append(Path(f).read_text()[:5 * CHUNK])
        except (UnicodeDecodeError, OSError):
            pass
    chunks = [p[i:i + CHUNK] for p in parts for i in range(0, len(p), CHUNK)]
    chunks = [c for c in chunks if len(c) > 200]
    rng.shuffle(chunks)
    for n, text in enumerate(chunks[:DEV_ROWS]):
        gold = []
        if n % 3 == 0:
            lines = planted(rng.choice(fakes), rng)
            cut = text.find("\n", rng.randint(0, len(text) - 1)) + 1
            block = "".join(line + "\n" for line, _ in lines)
            text = text[:cut] + block + text[cut:]
            at = cut
            for line, values in lines:
                for value, cat in values:
                    s = text.index(value, at)
                    gold.append([s, s + len(value), cat])
                at += len(line) + 1
        yield {"source": "dev", "text": text, "gold": gold}


def hex_key(rng):
    """A made-up secret in hex, as many services issue them. Lockfile checksums look the same, so only the
    name it is assigned to tells them apart."""
    return "".join(rng.choices("0123456789abcdef", k=rng.choice([32, 40, 64])))


def checksums(d: Path, pattern: str, algorithm: str) -> str:
    """What `shasum -a 256` or `md5 -r` prints for the first 40 files under `d` that match `pattern`."""
    files = sorted(p.relative_to(d) for p in d.rglob(pattern) if p.is_file())[:40]
    sep = "  " if algorithm == "sha256" else " "
    return "".join(f"{hashlib.new(algorithm, (d / f).read_bytes()).hexdigest()}{sep}./{f}\n" for f in files)


def lock_rows():
    """Hash-heavy developer text with nothing private in it: other projects' Cargo.lock files, file checksums and git
    object listings. Only the planted values are labeled, so every checksum is a negative. npm lockfiles stay out:
    the held-out developer corpus has one, so it checks whether this carries over."""
    from faker import Faker
    rng = random.Random(9)
    fakes = [Faker(loc) for loc in ("en_US", "en_GB", "de_DE", "es_ES", "fr_FR", "zh_CN", "ja_JP", "pt_BR", "it_IT", "en_IN")]
    for f in fakes:
        f.seed_instance(rng.randint(0, 10**9))
    sources = crates()
    git = lambda repo, *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
    parts = [(d / "Cargo.lock").read_text()[:2 * CHUNK] for d in sources if (d / "Cargo.lock").exists()]
    for d in rng.sample(sources, 40):
        parts.append(checksums(d, "*.rs", "sha256"))
        parts.append(checksums(d, "*", "md5"))
    for name, (url, commit) in sorted(SOURCES["repos"].items()):
        repo = checkout(url, commit, DATA / "repos" / name)
        parts.append("".join(git(repo, "ls-files", "-s").splitlines(keepends=True)[:300]))
        parts.append("".join(git(repo, "rev-list", "--objects", "HEAD").splitlines(keepends=True)[:300]))
    chunks = [p[i:i + CHUNK] for p in parts for i in range(0, len(p), CHUNK)]
    chunks = [c for c in chunks if len(c) > 200]
    rng.shuffle(chunks)
    for n, text in enumerate(chunks[:800]):
        gold = []
        if n % 3 == 0:
            lines = planted(rng.choice(fakes), rng)
            if rng.random() < 0.5:  # a hex secret under a secret's name, so the name decides, not the characters
                key = hex_key(rng)
                lines.append((rng.choice([f"SECRET_KEY={key}", f'api_secret = "{key}"', f"  client_secret: {key}"]), [(key, "secret")]))
            cut = text.find("\n", rng.randint(0, len(text) - 1)) + 1
            text = text[:cut] + "".join(line + "\n" for line, _ in lines) + text[cut:]
            at = cut
            for line, values in lines:
                for value, cat in values:
                    s = text.index(value, at)
                    gold.append([s, s + len(value), cat])
                at += len(line) + 1
        yield {"source": "lock", "text": text, "gold": gold, "teacher": []}


SECRET_NAMES = ["key", "secret", "token", "password", "passwd", "pwd", "api_key", "apikey", "api_secret", "access_key", "secret_key",
                "private_key", "client_secret", "auth_token", "access_token", "refresh_token", "session_secret", "signing_key",
                "encryption_key", "master_key", "app_secret", "consumer_secret", "webhook_secret", "db_password", "jwt_secret", "hmac_key",
                "bot_token", "subscription_key", "license_key", "private_token", "auth_key", "credential"]
PLAIN_NAMES = ["checksum", "sha", "sha1", "sha256", "md5", "hash", "digest", "etag", "commit", "commit_sha", "revision", "rev", "git_sha",
               "build_id", "object_id", "blob", "content_hash", "cache_key", "trace_id", "request_id", "span_id", "correlation_id", "uuid",
               "id", "guid", "image_digest", "file_hash", "parent", "tree", "sort_key", "partition_key", "run_id"]
FORMATS = {  # file kind -> assignment line templates; {n} is the name in that file's style, {v} the value
    "env": ["{N}={v}", "export {N}={v}", '{N}="{v}"'],
    "python": ['{n} = "{v}"', '    {n}="{v}",', '    "{n}": "{v}",', 'os.environ["{N}"] = "{v}"'],
    "js": ['const {c} = "{v}";', "  {c}: '{v}',", 'process.env.{N} = "{v}";'],
    "java": ['private static final String {N} = "{v}";', '    String {c} = "{v}";'],
    "go": ['{c} := "{v}"', '\t{C}: "{v}",'],
    "yaml": ["{n}: {v}", '  {n}: "{v}"', "    {k}: {v}"],
    "json": ['  "{n}": "{v}",', '    "{c}": "{v}"'],
    "toml": ['{n} = "{v}"', "{n}={v}"],
    "xml": ["<{c}>{v}</{c}>", '  <add key="{c}" value="{v}" />'],
    "php": ["${c} = '{v}';", "    '{n}' => '{v}',"],
}


def styles(name: str) -> dict:
    parts = name.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return {"n": name, "N": name.upper(), "c": camel, "C": camel[0].upper() + camel[1:], "k": name.replace("_", "-")}


def value(rng, secret: bool) -> str:
    """The same shapes for both kinds of names, so only the name tells a secret from a checksum or an ID."""
    kind = rng.random()
    if kind < 0.45:
        return "".join(rng.choices("0123456789abcdef", k=rng.choice([16, 24, 32, 40, 48, 64, 128])))
    if kind < 0.65:
        import base64
        return base64.b64encode(rng.randbytes(rng.choice([15, 24, 32, 48]))).decode()
    if kind < 0.8:
        return "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=rng.randint(20, 48)))
    if kind < 0.9:
        import uuid
        return str(uuid.UUID(bytes=rng.randbytes(16)))
    if secret:  # shapes that only secrets have
        return rng.choice(["sk-", "ghp_", "xoxb-", "glpat-", "AKIA", "AIza"]) + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789", k=rng.randint(20, 40)))
    return f"{rng.randint(0, 9)}.{rng.randint(0, 40)}.{rng.randint(0, 20)}"


def contrast_rows(n: int):
    """Code and config files where values of the same shapes sit under secret names (labeled) and under checksum, ID
    and hash names (not labeled), between lines of real source code."""
    rng = random.Random(13)
    files = [f for d in crates() for f in sorted(glob.glob(f"{d}/src/*.rs"))[:2]]
    filler = []
    for f in rng.sample(files, min(300, len(files))):
        lines = [x for x in Path(f).read_text(errors="ignore").splitlines() if x.strip()]
        if len(lines) > 12:
            at = rng.randint(0, len(lines) - 6)
            filler.append(lines[at:at + rng.randint(2, 6)])
    for _ in range(n):
        kind = rng.choice(list(FORMATS))
        out, gold = [], []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.6:
                out += rng.choice(filler)
            for _ in range(rng.randint(2, 6)):
                secret = rng.random() < 0.5
                name = rng.choice(SECRET_NAMES if secret else PLAIN_NAMES)
                if rng.random() < 0.3:  # a prefix such as stripe_, github_, aws_
                    name = rng.choice(["stripe", "github", "aws", "app", "prod", "service", "redis", "s3", "oauth", "build", "repo", "cdn"]) + "_" + name
                v = value(rng, secret)
                line = rng.choice(FORMATS[kind]).format(v=v, **styles(name))
                if secret:
                    start = sum(len(x) + 1 for x in out) + line.index(v)
                    gold.append([start, start + len(v), "secret"])
                out.append(line)
        yield {"source": "contrast", "text": "\n".join(out) + "\n", "gold": gold, "teacher": []}


class Doc:
    """Text built piece by piece, with the span of every labeled piece."""

    def __init__(self):
        self.parts, self.spans, self.size = [], [], 0

    def add(self, s: str, label: str | None = None):
        if label and s.strip():
            lead = len(s) - len(s.lstrip())
            self.spans.append([self.size + lead, self.size + lead + len(s.strip()), label])
        self.parts.append(s)
        self.size += len(s)

    def text(self) -> str:
        return "".join(self.parts)


def agent_rows(n: int):
    """What coding agents read in logs, cluster and CI output, exports and configs, with exact labels by a fixed
    policy: people's names, personal user names, emails, phones, cities and public client IPs are private, as are
    secrets; pod names, request and trace IDs, UUIDs, hashes, digests, private-range IPs, ports and times are not."""
    from faker import Faker
    rng = random.Random(21)
    fakes = [Faker(loc) for loc in ("en_US", "en_GB", "de_DE", "fr_FR", "es_ES", "it_IT", "pt_BR", "nl_NL", "en_IN", "zh_CN", "ja_JP", "ko_KR")]
    for f in fakes:
        f.seed_instance(rng.randint(0, 10**9))
    loose = Faker()  # passwords and card numbers; seeded apart from rng, so every other value stays as it was
    loose.seed_instance(21)
    hexs = lambda k: "".join(rng.choices("0123456789abcdef", k=k))
    alnum = lambda k: "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=k))
    system_users = ["root", "admin", "ubuntu", "git", "deploy", "postgres", "oracle", "test", "www-data", "ec2-user", "jenkins", "nobody"]

    def person():
        f = rng.choice(fakes)
        first, last = f.first_name(), f.last_name()
        ascii_ok = (first + last).isascii()
        user = rng.choice([f"{first[0]}{last}", f"{first}.{last}", f"{first}{rng.randint(1, 99)}"]).lower() if ascii_ok else f"user{rng.randint(100, 999)}"
        email = f"{user if ascii_ok else alnum(8)}@{rng.choice(['example.com', 'example.org', 'mail.example.net', f.domain_name()])}"
        return f, f.name(), user, email, ascii_ok

    def public_ip():
        while True:
            a = rng.randint(1, 223)
            if a not in (10, 127, 169, 172, 192):
                return f"{a}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"

    def private_ip():
        return rng.choice([f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}", f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                           f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"])

    def ts():
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"

    def secret():
        return rng.choice([lambda: "sk-" + alnum(40), lambda: "ghp_" + alnum(36), lambda: hexs(rng.choice([32, 40, 64])), lambda: "xoxb-" + alnum(30),
                           lambda: loose.password(length=rng.randint(12, 20))])()

    def access_log(d):
        for _ in range(rng.randint(4, 12)):
            d.add(public_ip(), "private_url")
            d.add(" - ")
            if rng.random() < 0.3:
                d.add(person()[2], "private_person")
            else:
                d.add("-")
            path = rng.choice(["/api/v2/orders", "/login", "/static/app.js", "/healthz", "/api/users/me", "/checkout", "/search"])
            d.add(f' [{rng.randint(1, 28):02d}/Nov/2026:{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} +0000] "{rng.choice(["GET", "POST", "PUT"])} {path}')
            if rng.random() < 0.15:
                d.add("?email=")
                d.add(person()[3], "private_email")
            d.add(f' HTTP/1.1" {rng.choice([200, 201, 302, 401, 404, 500, 502])} {rng.randint(100, 90000)} "-" "{rng.choice(["Mozilla/5.0 (Macintosh)", "curl/8.7.1", "python-requests/2.32.3", "Go-http-client/2.0"])}"\n')

    def auth_log(d):
        host = rng.choice(["web-1", "bastion", "api-2", "ci-runner"])
        for _ in range(rng.randint(4, 10)):
            d.add(f"Nov {rng.randint(1, 28):2d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} {host} sshd[{rng.randint(1000, 99999)}]: ")
            kind = rng.random()
            if kind < 0.4:
                d.add("Accepted publickey for ")
                d.add(person()[2], "private_person")
                d.add(" from ")
                d.add(public_ip(), "private_url")
                d.add(f" port {rng.randint(1024, 65535)} ssh2: ED25519 SHA256:{alnum(43)}\n")
            elif kind < 0.7:
                d.add(f"Invalid user {rng.choice(system_users)} from ")
                d.add(public_ip(), "private_url")
                d.add(f" port {rng.randint(1024, 65535)}\n")
            else:
                d.add(f"Connection closed by {private_ip()} port {rng.randint(1024, 65535)} [preauth]\n")

    def json_log(d):
        for _ in range(rng.randint(3, 8)):
            _, name, user, email, _ = person()
            d.add('{"ts": "' + ts() + '", "level": "' + rng.choice(["info", "warn", "error"]) + '", "msg": "' + rng.choice(["user signed in", "payment failed", "profile updated", "token refreshed"]) + '", ')
            d.add(f'"user_id": "usr_{alnum(12)}", "email": "')
            d.add(email, "private_email")
            if rng.random() < 0.5:
                d.add('", "name": "')
                d.add(name, "private_person")
            d.add('", "ip": "')
            ip = public_ip() if rng.random() < 0.6 else private_ip()
            d.add(ip, "private_url" if not ip.startswith(("10.", "192.168.", "172.")) else None)
            d.add(f'", "request_id": "{uuid_str()}", "trace_id": "{hexs(32)}", "duration_ms": {rng.randint(1, 3000)}}}\n')

    def uuid_str():
        import uuid
        return str(uuid.UUID(bytes=rng.randbytes(16)))

    def kubectl(d):
        d.add("NAME                                READY   STATUS             RESTARTS   AGE   IP             NODE\n")
        for _ in range(rng.randint(3, 8)):
            app = rng.choice(["payments", "checkout", "auth", "search", "billing", "notifier", "gateway"])
            d.add(f"{app}-{rng.choice(['svc', 'api', 'worker'])}-{alnum(9)}-{alnum(5)}".ljust(36) + f"1/1     {rng.choice(['Running', 'CrashLoopBackOff', 'Pending'])}".ljust(27)
                  + f"{rng.randint(0, 9)}          {rng.randint(1, 30)}d   {private_ip()}".ljust(32) + f"node-{alnum(5)}\n")
        if rng.random() < 0.5:
            d.add("---\napiVersion: v1\nkind: Secret\nmetadata:\n  name: app-credentials\ndata:\n  DB_PASSWORD: ")
            import base64
            d.add(base64.b64encode(rng.randbytes(18)).decode(), "secret")
            d.add(f"\n  image: registry.example.com/app@sha256:{hexs(64)}\n")

    def docker(d):
        d.add("CONTAINER ID   IMAGE                          STATUS          PORTS                    NAMES\n")
        for _ in range(rng.randint(2, 6)):
            d.add(f"{hexs(12)}   {rng.choice(['postgres:16', 'redis:7.2', 'nginx:1.27', 'app:latest'])}".ljust(46) + f"Up {rng.randint(1, 59)} minutes   0.0.0.0:{rng.randint(3000, 9000)}->80/tcp   {rng.choice(['db', 'cache', 'web', 'worker'])}-{rng.randint(1, 3)}\n")
        if rng.random() < 0.5:
            d.add('"Env": [\n    "PATH=/usr/local/bin:/usr/bin",\n    "API_KEY=')
            d.add(secret(), "secret")
            d.add('",\n    "LOG_LEVEL=info"\n]\n')

    def git_log(d):
        for _ in range(rng.randint(2, 5)):
            _, name, _, email, _ = person()
            d.add(f"commit {hexs(40)}\nAuthor: ")
            d.add(name, "private_person")
            d.add(" <")
            d.add(email, "private_email")
            d.add(f">\nDate:   {rng.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])} Nov {rng.randint(1, 28)} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} 2026 +0000\n\n    {rng.choice(['Fix retry on timeout', 'Add billing export', 'Bump deps', 'Refactor auth middleware'])}\n\n")

    def traceback(d):
        user = person()[2]
        d.add("Traceback (most recent call last):\n")
        for _ in range(rng.randint(2, 4)):
            d.add('  File "/home/')
            d.add(user, "private_person")
            d.add(f'/src/{rng.choice(["api", "worker", "billing"])}/{rng.choice(["views", "tasks", "models", "client"])}.py", line {rng.randint(10, 900)}, in {rng.choice(["handle", "run", "save", "fetch"])}\n')
        d.add(f"{rng.choice(['KeyError', 'ValueError', 'ConnectionError'])}: request {uuid_str()} failed\n")

    def csv_export(d):
        d.add("id,name,email,phone,city,plan,amount\n")
        for i in range(rng.randint(3, 10)):
            f, name, _, email, _ = person()
            d.add(f"{i + 1},")
            d.add(name, "private_person")
            d.add(",")
            d.add(email, "private_email")
            d.add(",")
            d.add(f.phone_number(), "private_phone")
            d.add(",")
            d.add(f.city(), "private_address")
            d.add(f",{rng.choice(['free', 'pro', 'team'])},{rng.randint(0, 900)}.{rng.randint(0, 99):02d}\n")

    def config(d):
        _, _, user, email, _ = person()
        d.add(f"service:\n  name: {rng.choice(['orders', 'ledger', 'mailer'])}\n  host: {rng.choice(['db.internal', 'redis.svc.cluster.local', 'api.staging.example.com'])}\n  port: {rng.choice([5432, 6379, 8080, 443])}\n")
        d.add("database:\n  user: app\n  password: ")
        d.add(secret(), "secret")
        if rng.random() < 0.5:  # credentials inside a connection URL: the password is the secret
            d.add(f"\n  url: {rng.choice(['postgres', 'mysql', 'redis', 'amqp', 'mongodb'])}://{rng.choice([user, 'app', 'svc'])}:")
            d.add(loose.password(length=rng.randint(10, 18), special_chars=False), "secret")
            d.add(f"@{rng.choice(['db.internal', '10.0.3.7', 'cache.svc.cluster.local'])}:{rng.choice([5432, 3306, 6379, 5672, 27017])}/app")
        for name in rng.sample(["nonce", "salt", "passphrase", "pin", "client_key", "app_key", "secret_access_key", "signing_secret"], rng.randint(1, 2)):
            d.add(f"\n  {name}: ")
            d.add(secret(), "secret")
        d.add("\nsmtp:\n  username: ")
        d.add(email, "private_email")
        d.add("\n  token: ")
        d.add(secret(), "secret")
        d.add(f"\n  request_timeout: {rng.randint(5, 60)}s\n  build_sha: {hexs(40)}\n")

    def github(d):
        """Pull request and review text: people's handles are private, organizations and teams are not."""
        org = rng.choice(["acme", "initech", "globex", "hooli", "wayne-labs"])
        for _ in range(rng.randint(2, 6)):
            f, name, user, email, _ = person()
            handle = user.replace(".", "-")
            form = rng.randint(0, 5)
            if form == 0:
                d.add("@")
                d.add(handle, "private_person")
                d.add(rng.choice([" could you rebase this onto main?", " approved these changes", " requested changes", " left a review"]) + "\n")
            elif form == 1:
                d.add(rng.choice(["cc ", "/cc ", "Assigned to "]) + "@")
                d.add(handle, "private_person")
                d.add(f" for the {rng.choice(['flaky test', 'release notes', 'migration', 'docs fix'])}\n")
            elif form == 2:
                d.add(f"{rng.choice(['/src/billing/', '/docs/', '*.rs ', '/.github/'])}  @{org}/{rng.choice(['backend', 'platform', 'docs'])} @")
                d.add(handle, "private_person")
                d.add("\n")
            elif form == 3:
                d.add(rng.choice(["Co-authored-by: ", "Signed-off-by: ", "Reviewed-by: "]))
                d.add(name, "private_person")
                d.add(" <")
                d.add(email, "private_email")
                d.add(">\n")
            elif form == 4:
                d.add(f"Merge pull request #{rng.randint(10, 9999)} from ")
                d.add(handle, "private_person")
                d.add(f"/{rng.choice(['fix-timeout', 'feature/export', 'deps'])}\n")
            else:
                d.add(f"{rng.choice(['Fixes', 'Closes', 'Refs'])} {org}/{rng.choice(['api', 'web', 'cli'])}#{rng.randint(1, 9999)}, thanks @")
                d.add(handle, "private_person")
                d.add("\n")

    def chat(d):
        """Workplace chat and email: first names alone, often lowercase, are private; weekdays, teams, regions,
        quarters and companies are not."""
        def first():
            n = rng.choice(fakes[:9]).first_name()  # scripts with case
            return n.lower() if rng.random() < 0.5 else n
        filler = ["the Q3 numbers", "the EMEA rollout", "Thursday's sync", "the Design review", "the Acme renewal", "the Platform team's plan",
                  "Monday standup", "the Northwind contract", "the March launch", "the APAC forecast"]
        interjections = ["yeah", "yep", "btw", "lol", "ok", "okay", "sure", "hmm", "nvm", "fyi", "haha", "np", "ty", "omg", "kk", "yo", "ah", "oh",
                         "thanks", "cool", "great", "agreed", "same", "lgtm", "ack", "welp"]
        groups = ["team", "all", "everyone", "folks", "legal", "design", "finance", "y'all", "both"]
        for _ in range(rng.randint(3, 7)):
            if rng.random() < 0.4:  # messages with no name in them, opening the way a name sometimes does
                word = rng.choice(interjections)
                word = word if rng.random() < 0.7 else word.capitalize()
                d.add(rng.choice([f"{word} that works for me\n", f"{word}, the deck is updated\n", f"{word} quick q on {rng.choice(filler)}\n", f"{word}!\n",
                                  f"{word} - will do\n", f"hey {rng.choice(groups)}! {word}, {rng.choice(filler)} is moved to Friday\n",
                                  f"cc {rng.choice(groups)} on {rng.choice(filler)}\n", f"thanks {rng.choice(groups)}\n"]))
                continue
            form = rng.randint(0, 6)
            n1, n2 = first(), first()
            if form == 0:
                d.add(rng.choice(["hey ", "hi ", "morning ", "Hi "]))
                d.add(n1, "private_person")
                d.add(rng.choice(["! you around?", ", quick question on ", "! got a sec for "]) + (rng.choice(filler) if rng.random() < 0.6 else "") + "\n")
            elif form == 1:
                d.add(rng.choice(["looping in ", "can you ping ", "I'll pull ", "cc "]))
                d.add(n1, "private_person")
                d.add(rng.choice([" so they're in the loop", " directly", " in as well", ""]) + f" on {rng.choice(filler)}\n")
            elif form == 2:
                d.add(f"got {rng.choice(filler)} back from ")
                d.add(n1, "private_person")
                d.add(". " + rng.choice(["looks good", "a few gaps", "can we discuss on Friday?"]) + "\n")
            elif form == 3:
                d.add(n1, "private_person")
                d.add(rng.choice([", can you take a look at ", " — did you see ", ": thoughts on "]) + rng.choice(filler) + "?\n")
            elif form == 4:
                d.add(rng.choice(["Thanks,\n", "Best,\n", "—", "cheers\n"]))
                d.add(n1, "private_person")
                d.add("\n")
            elif form == 5:
                d.add(rng.choice(["good catch. ", "agreed. ", ""]))
                d.add(n1, "private_person")
                d.add(" and ")
                d.add(n2, "private_person")
                d.add(f" asked about {rng.choice(filler)} on {rng.choice(['Monday', 'Tuesday', 'Thursday'])}.\n")
            else:
                d.add(f"Agenda for {rng.choice(['Monday', 'Wednesday', 'Friday'])}: {rng.choice(filler)}, then {rng.choice(filler)} with the {rng.choice(['Sales', 'Finance', 'Legal', 'Marketing'])} team.\n")

    def payments(d):
        """Card numbers, CVCs and IBANs are private; masked cards, order and invoice ids, amounts and times are not."""
        def card():
            digits = loose.credit_card_number()
            sep = rng.choice([" ", "-", ""])
            return sep.join(digits[i:i + 4] for i in range(0, len(digits), 4)) if sep else digits
        for _ in range(rng.randint(2, 5)):
            form = rng.randint(0, 4)
            amount = f"{rng.randint(1, 2000)}.{rng.randint(0, 99):02d}"
            if form == 0:
                d.add('{"event": "' + rng.choice(["charge.failed", "charge.succeeded", "card.updated"]) + '", "card": {"number": "')
                d.add(card(), "account_number")
                d.add(f'", "exp": "{rng.randint(1, 12):02d}/{rng.randint(27, 33)}", "cvc": "')
                d.add(str(rng.randint(100, 9999)), "secret")
                d.add(f'"}}, "customer": "cus_{alnum(14)}", "amount": {amount}, "order": "ORD-{rng.randint(2024, 2026)}-{rng.randint(1000, 9999)}"}}\n')
            elif form == 1:
                d.add(f"{ts()} payment {rng.choice(['declined', 'captured', 'refunded'])} for card ")
                d.add(card(), "account_number")
                d.add(f" (invoice INV-{rng.randint(10000, 99999)}, {amount} {rng.choice(['USD', 'EUR', 'GBP'])})\n")
            elif form == 2:
                d.add(f"Charged ${amount} to {rng.choice(['Visa', 'Mastercard', 'Amex'])} ending in {rng.randint(1000, 9999)} on {rng.choice(['Nov', 'Dec', 'Jan'])} {rng.randint(1, 28)}\n")
            elif form == 3:
                d.add(f"refund {rng.randint(100000, 999999)} -> IBAN ")
                d.add(rng.choice(fakes[:9]).iban(), "account_number")
                d.add(f" amount {amount}\n")
            else:
                d.add(f"order_id,card,amount\n{rng.randint(1, 9999)},")
                d.add(card(), "account_number")
                d.add(f",{amount}\n{rng.randint(1, 9999)},**** **** **** {rng.randint(1000, 9999)},{amount}\n")

    kinds = [access_log, auth_log, json_log, kubectl, docker, git_log, traceback, csv_export, config, github, chat, payments]
    for _ in range(n):
        d = Doc()
        for fmt in rng.sample(kinds, rng.randint(1, 3)):
            fmt(d)
            d.add("\n")
        yield {"source": "agent", "text": d.text(), "gold": d.spans, "teacher": []}


def secret_rows(n: int):
    """Secrets as source code writes them, each kind next to values of the same shape that are not secrets: byte
    arrays (authData vs magic), short passwords (pass vs mode), SQL and command-line passwords, Basic authorization,
    provider tokens and private keys. Code lines from real crates sit between them."""
    import base64
    rng = random.Random(29)
    files = [f for d in crates() for f in sorted(glob.glob(f"{d}/src/*.rs"))[:1]]
    filler = []
    for f in rng.sample(files, min(200, len(files))):
        lines = [x for x in Path(f).read_text(errors="ignore").splitlines() if x.strip()]
        if len(lines) > 10:
            at = rng.randint(0, len(lines) - 5)
            filler.append(lines[at:at + rng.randint(2, 5)])
    alnum = lambda k, chars="abcdefghijklmnopqrstuvwxyz0123456789": "".join(rng.choices(chars, k=k))
    secret_names = ["authData", "key", "secretKey", "hmacKey", "aesKey", "privKey", "seed", "psk", "masterSecret", "sessionKey", "signingKey",
                    "salt", "nonce", "token", "apiSecret", "encKey"]
    plain_names = ["magic", "header", "prefix", "delimiter", "pngSignature", "gzipMagic", "versionBytes", "opcodes", "palette", "lut", "crcTable",
                   "bom", "marker", "padding", "testInput", "fixture"]
    pw_names = ["pass", "password", "passwd", "pwd", "psw", "pw", "db_pass", "mysql_pwd", "root_password", "admin_pass", "repsw", "userPassword"]
    word_names = ["mode", "status", "theme", "color", "lang", "region", "label", "title", "env", "tier", "kind", "level"]
    words = ["admin", "hunter2", "letmein", "changeme", "qwerty", "secret", "dragon", "monkey", "sunshine", "welcome1", "admin123", "pass1234"]

    def byte_list():
        k = rng.choice([8, 12, 16, 24, 32])
        vals = [rng.randint(0, 255) for _ in range(k)]
        return ", ".join(f"0x{v:02x}" for v in vals) if rng.random() < 0.5 else ", ".join(str(v) for v in vals)

    def short_value():
        return rng.choice([alnum(rng.randint(4, 9), "abcdefghijklmnopqrstuvwxyz"), rng.choice(words), rng.choice(words) + str(rng.randint(1, 99))])

    def line(d, secret: bool):
        kind = rng.randint(0, 6)
        if kind == 0:  # byte arrays
            name = rng.choice(secret_names if secret else plain_names)
            body = byte_list()
            pre, post = rng.choice([(f"{name} := []byte{{", "}"), (f"var {name} = [{body.count(',') + 1}]byte{{", "}"),
                                    (f"static const uint8_t {name}[] = {{", "};"), (f"byte[] {name} = new byte[]{{", "};"),
                                    (f"{name} = bytes([", "])"), (f"const {name.upper()}: [u8; {body.count(',') + 1}] = [", "];")])
            d.add(pre)
            d.add(body, "secret" if secret else None)
            d.add(post + "\n")
        elif kind == 1:  # short passwords and words of the same shape under other names
            name, v = rng.choice(pw_names if secret else word_names), short_value()
            pre, post = rng.choice([(f"var {name} = '", "';"), (f"{name}: '", "'"), (f'{name} = "', '"'), (f"${name} = '", "';"),
                                    (f"{name.upper()}=", ""), (f'String {name} = "', '";'), (f'"{name}": "', '",')])
            d.add(pre)
            d.add(v, "secret" if secret else None)
            d.add(post + "\n")
        elif kind == 2:  # SQL and command lines
            user, v = rng.choice(["gotest", "app", "reporting", "gh-ost", "svc_etl"]), short_value() if rng.random() < 0.5 else alnum(rng.randint(10, 20))
            if secret:
                pre, post = rng.choice([(f"CREATE USER {user} IDENTIFIED BY '", "';"), (f"ALTER USER '{user}'@'%' IDENTIFIED BY \"", '";'),
                                        (f"CREATE ROLE {user} WITH LOGIN PASSWORD '", "';"), ("mysql -uroot -p", " -e 'SHOW DATABASES'"),
                                        ("PGPASSWORD=", f" psql -h db -U {user}"), (f"sqlcmd -S db -U {user} -P ", " -Q \"SELECT 1\""),
                                        (f"curl -u {user}:", " https://api.example.com/v1/jobs")])
                d.add(pre)
                d.add(v, "secret")
                d.add(post + "\n")
            else:
                d.add(rng.choice([f"SELECT * FROM users WHERE status = '{v}';", f"CREATE TABLE {user}_log (id bigint, note text);",
                                  f"GRANT SELECT ON reports TO {user};", f"mysql -uroot -e 'USE {v}'", f"psql -h db -U {user} -c 'SELECT 1'"]) + "\n")
        elif kind == 3:  # Basic authorization and other headers
            if secret:
                cred = base64.b64encode(f"{alnum(rng.randint(4, 8), 'abcdefghijklmnopqrstuvwxyz')}:{alnum(rng.randint(6, 14))}".encode()).decode()
                pre, post = rng.choice([("Authorization: Basic ", ""), ("BASIC_AUTH_HEADER_VALUE = 'Basic ", "'"), ('headers["Authorization"] = "Basic ', '"'),
                                        ('curl -H "Authorization: Basic ', '" https://ci.example.com/api')])
                d.add(pre)
                d.add(cred, "secret")
                d.add(post + "\n")
            else:
                d.add(rng.choice(["Content-Type: application/json", "Accept: text/html", "Authorization: Bearer ***", "X-Request-Id: " + alnum(16),
                                  "Cache-Control: no-cache", 'headers["Accept"] = "application/json"']) + "\n")
        elif kind == 4:  # provider tokens, bare or assigned
            if secret:
                v = rng.choice([lambda: rng.choice(["xoxb-", "xoxp-", "xoxa-", "xoxe-", "xapp-", "xoxr-"]) + "-".join(alnum(rng.randint(8, 13)) for _ in range(3)),
                                lambda: rng.choice(["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]) + alnum(36, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
                                lambda: "github_pat_" + alnum(82, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"),
                                lambda: rng.choice(["AKIA", "ASIA"]) + alnum(16, "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"),
                                lambda: "AIza" + alnum(35, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"),
                                lambda: rng.choice(["sk_live_", "rk_live_", "sk_test_"]) + alnum(24, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
                                lambda: "SG." + alnum(22, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_") + "." + alnum(43, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"),
                                lambda: rng.choice(["sk-proj-", "sk-ant-api03-", "hf_", "npm_", "pypi-", "glpat-"]) + alnum(rng.randint(30, 48), "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
                                lambda: "eyJ" + base64.urlsafe_b64encode(rng.randbytes(24)).decode().rstrip("=") + ".eyJ" + base64.urlsafe_b64encode(rng.randbytes(30)).decode().rstrip("=") + "." + base64.urlsafe_b64encode(rng.randbytes(32)).decode().rstrip("=")])()
                pre = rng.choice(["", "token: ", "SLACK_BOT_TOKEN=", "export GITHUB_TOKEN=", "aws_access_key_id = ", "Authorization: Bearer ", "  - api_key: "])
                d.add(pre)
                d.add(v, "secret")
                d.add("\n")
            else:
                d.add(rng.choice([f"commit {alnum(40, '0123456789abcdef')}", f"image: app@sha256:{alnum(64, '0123456789abcdef')}",
                                  f"request_id={alnum(8)}-{alnum(4)}-{alnum(4)}-{alnum(4)}-{alnum(12)}", f"run_id: {rng.randint(10**9, 10**10)}",
                                  f"build: {alnum(7, '0123456789abcdef')}"]) + "\n")
        elif kind == 5:  # private keys
            if secret:
                kind_name = rng.choice(["RSA PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"])
                d.add(f"-----BEGIN {kind_name}-----\n")
                body = "\n".join(base64.b64encode(rng.randbytes(48)).decode() for _ in range(rng.randint(2, 5)))
                d.add(body, "secret")
                d.add(f"\n-----END {kind_name}-----\n")
            else:
                d.add("-----BEGIN CERTIFICATE-----\n" + "\n".join(base64.b64encode(rng.randbytes(48)).decode() for _ in range(2)) + "\n-----END CERTIFICATE-----\n")
        else:  # connection strings with the password field
            if secret:
                pre = rng.choice(["Server=db;Database=app;User Id=sa;Password=", "Data Source=db;Initial Catalog=app;User ID=app;Pwd=",
                                  "host=db port=5432 dbname=app user=app password="])
                d.add(pre)
                d.add(short_value() if rng.random() < 0.4 else alnum(rng.randint(10, 18)), "secret")
                d.add(";\n" if "Server" in pre or "Data Source" in pre else "\n")
            else:
                d.add("Server=db;Database=app;Integrated Security=true;Encrypt=true;\n")

    for _ in range(n):
        d = Doc()
        for _ in range(rng.randint(3, 8)):
            if rng.random() < 0.35:
                for x in rng.choice(filler):
                    d.add(x + "\n")
            line(d, rng.random() < 0.5)
        yield {"source": "secret", "text": d.text(), "gold": d.spans, "teacher": []}


def zh_docs_rows():
    """Chinese technical writing from the docs of MIT-licensed projects (ant-design, vant, ahooks, umi), in chunks;
    one in three gets a planted line with a Faker name, email or phone."""
    from faker import Faker
    rng = random.Random(41)
    fake = Faker("zh_CN")
    fake.seed_instance(41)
    for name, (url, commit) in SOURCES["zh_docs"].items():
        checkout(url, commit, DATA / "zh_docs" / name, ZH_DOCS_FILES)
    files = sorted(glob.glob(str(DATA / "zh_docs/**/*.md"), recursive=True))
    cjk = re.compile(r"[\u4e00-\u9fff]")
    chunks = []
    for f in files:
        lines, cur = Path(f).read_text(errors="ignore").splitlines(keepends=True), ""
        for line in lines:
            cur += line
            if len(cur) > 800:
                chunks.append(cur)
                cur = ""
        if len(cur) > 300:
            chunks.append(cur)
    chunks = [c for c in chunks if len(cjk.findall(c)) > 0.12 * len(c)]
    rng.shuffle(chunks)
    templates = ["> 如有疑问，请联系{name}（{email}）。", "本节由{name}整理，问题请打电话 {phone}。", "感谢{name}提供的示例。",
                 "维护者：{name} <{email}>", "反馈请发邮件给 {email}，或在群里找{name}。"]
    for n, text in enumerate(chunks):
        d = Doc()
        if n % 3 == 0:
            cut = text.find("\n", rng.randint(0, len(text) - 1)) + 1
            d.add(text[:cut])
            for part in re.split(r"(\{\w+\})", rng.choice(templates)):
                if part == "{name}":
                    d.add(fake.name(), "private_person")
                elif part == "{email}":
                    d.add(fake.email(), "private_email")
                elif part == "{phone}":
                    d.add(fake.phone_number(), "private_phone")
                else:
                    d.add(part)
            d.add("\n" + text[cut:])
        else:
            d.add(text)
        yield {"source": "zh_docs", "text": d.text(), "gold": d.spans}


CJK = re.compile(r"[一-鿿]")


def not_a_name(text: str, span: list) -> bool:
    """A person span in Chinese documentation that cannot be a name: a package path such as `umi/max`, a dotted name,
    or Chinese text longer than four characters or with punctuation in it."""
    v = text[span[0]:span[1]]
    return span[2] == "private_person" and (bool(CJK.search(v)) and (len(v) > 4 or any(c in "，。、；：！？（）《》“”‘’【】" for c in v)) or "/" in v or "." in v)


def write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(path.name, dict(Counter(r["source"] for r in rows)))


def natural():
    """data/natural.jsonl: the public rows with the teacher's spans, then the developer text, whose planted values are
    its only labels. Labeling takes most of the time and resumes where it stopped."""
    DATA.mkdir(exist_ok=True)
    rows = list(public_rows())
    partial = DATA / "natural.partial.jsonl"
    done = sum(1 for _ in open(partial)) if partial.exists() else 0
    with open(partial, "a") as out:
        for i in range(done, len(rows), 400):
            batch = rows[i:i + 400]
            for r, s in zip(batch, spans([r["text"] for r in batch])):
                out.write(json.dumps(dict(r, teacher=s["detected"]), ensure_ascii=False) + "\n")
            out.flush()
            print(f"teacher labels {i + len(batch)} of {len(rows)}", flush=True)
    write(DATA / "natural.jsonl", [json.loads(line) for line in open(partial)] + [dict(r, teacher=[]) for r in dev_rows()])
    partial.unlink()


def constructed():
    """data/constructed.jsonl. The Chinese documentation is labeled by the teacher followed by holdback's rules, minus
    person spans that cannot be names, plus the planted values the teacher missed."""
    DATA.mkdir(exist_ok=True)
    rows = list(contrast_rows(1500)) + list(agent_rows(2000)) + list(secret_rows(1500)) + list(lock_rows())
    docs = list(zh_docs_rows())
    for r, s in zip(docs, spans([r["text"] for r in docs])):
        found = [x for x in s["refined"] if not not_a_name(r["text"], x)]
        r["gold"] = found + [g for g in r["gold"] if not any(g[0] < e and s0 < g[1] for s0, e, _ in found)]
    write(DATA / "constructed.jsonl", rows + [dict(r, teacher=[]) for r in docs])


if __name__ == "__main__":
    {"natural": natural, "constructed": constructed}[sys.argv[1]]()
