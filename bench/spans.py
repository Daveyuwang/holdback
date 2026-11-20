"""Span sources: the holdback binary and the three tools it is compared with. Spans are [start, end, label] in
character offsets. Model-based results are kept in bench/results/; holdback's are computed again when the texts, the
binary or the model change.

Compared tools: Codex's built-in secret regexes, Presidio's default analyzer, and rehydra with NER and secret
detection on. Presidio needs `uv sync --group compare`; rehydra needs `npm install` in bench/rehydra."""
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
BINARY = os.environ.get("HOLDBACK_BIN", str(HERE.parent / "rust/holdback/target/release/holdback"))

CODEX = [  # openai/codex codex-rs/secrets/src/sanitizer.rs; the last pattern keeps its first three groups
    re.compile(r"(?i:\bBearer)[ \t]+()()()[A-Za-z0-9._~+/-]{16,}=*"),
    re.compile(r"()()()sk-[A-Za-z0-9]{20,}"),
    re.compile(r"()()()\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[:=]\s*)([\"']?)[^\s\"']{8,}"),
]


def cached(name: str, compute, key=None):
    """compute()'s result, kept in bench/results/<name>.json; with `key`, computed again when the key changes."""
    path = RESULTS / f"{name}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if key is None:
            return saved
        if isinstance(saved, dict) and saved.get("key") == key:
            return saved["value"]
    RESULTS.mkdir(exist_ok=True)
    value = compute()
    path.write_text(json.dumps(value if key is None else {"key": key, "value": value}, ensure_ascii=False))
    return value


def run(texts: list[str], sequence: bool = False, env: dict | None = None) -> list[dict]:
    """`holdback spans` for each text: {"detected": the detector alone, "refined": with holdback's rules}.
    With sequence, the texts go through one placeholder map in order and each text gets {"all", "twice"}: what
    is replaced when a known value is replaced everywhere, or when a one-word person waits for a second text."""
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"texts": texts, "sequence": sequence}, f, ensure_ascii=False)
        f.flush()
        return json.loads(subprocess.run([BINARY, "spans", f.name], env=env, capture_output=True, text=True, check=True).stdout)


def detector_env(detector: str, model_dir=None) -> dict:
    """This environment with HOLDBACK_DETECTOR set to `detector` and, for the distilled one, another model's folder."""
    env = {k: v for k, v in os.environ.items() if k not in ("HOLDBACK_DETECTOR", "HOLDBACK_STUDENT_DIR")}
    return env | {"HOLDBACK_DETECTOR": detector} | ({"HOLDBACK_STUDENT_DIR": str(model_dir)} if model_dir else {})


def spans(texts: list[str], name: str, sequence: bool = False, env: dict | None = None) -> list[dict]:
    """run() with the detector the environment picks, kept per detector: the published distilled model without a
    prefix, HOLDBACK_DETECTOR=privacy-filter under privacy_filter_, a model in HOLDBACK_STUDENT_DIR under student_."""
    env = dict(os.environ) if env is None else env
    stamp = lambda path: [os.stat(path).st_size, os.stat(path).st_mtime_ns]
    if env.get("HOLDBACK_DETECTOR") == "privacy-filter":
        prefix, model = "privacy_filter_", env.get("HOLDBACK_MODEL_DIR") and stamp(Path(env["HOLDBACK_MODEL_DIR"]) / "onnx/model_q4.onnx_data")
    elif env.get("HOLDBACK_STUDENT_DIR"):
        prefix, model = "student_", [env["HOLDBACK_STUDENT_DIR"], stamp(Path(env["HOLDBACK_STUDENT_DIR"]) / "model.safetensors")]
    else:
        prefix, model = "", None  # the published model, pinned by the binary
    key = {"texts": hashlib.sha256(json.dumps(texts).encode()).hexdigest(), "binary": stamp(BINARY), "model": model}
    return cached(prefix + name + "_sequence" * sequence, lambda: run(texts, sequence, env), key)


def codex(texts: list[str], name: str) -> list:
    return [[[m.end(3), m.end(), "SECRET"] for rx in CODEX for m in rx.finditer(t)] for t in texts]


def presidio(texts: list[str], name: str) -> list:
    def compute():
        from presidio_analyzer import AnalyzerEngine
        engine = AnalyzerEngine()
        return [[[r.start, r.end, r.entity_type] for r in engine.analyze(t, language="en")] for t in texts]
    return cached(f"presidio_{name}", compute)


def rehydra(texts: list[str], name: str) -> list:
    def compute():
        with tempfile.TemporaryDirectory() as d:
            Path(d, "in.json").write_text(json.dumps(texts, ensure_ascii=False))
            subprocess.run(["node", "spans.mjs", f"{d}/in.json", f"{d}/out.json"], cwd=HERE / "rehydra", check=True)
            return json.loads(Path(d, "out.json").read_text())
    return cached(f"rehydra_{name}", compute)


COMPARED = {"codex": codex, "presidio": presidio, "rehydra": rehydra}


def merged(spans: list) -> list:
    out = []
    for s, e, label in sorted(spans):
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, label])
    return out


def mask(text: str, spans: list, token: str = "<X>") -> str:
    out, last = [], 0
    for s, e, _ in merged(spans):
        out += [text[last:s], token]
        last = e
    return "".join(out + [text[last:]])
