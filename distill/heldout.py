"""Test sets beyond holdback's benchmarks, scored for the Privacy Filter and a distilled model, each alone and followed
by holdback's rules, with holdback's metric code (bench/public_eval.py):

  personas      three PrivacyBench personas that no training row and no benchmark uses, 300 rows each
  creddata      400 CredData credentials from files that neither the benchmark's sample nor two earlier samples used;
                needs bench/results/CredData (see bench/public_eval.py)
  msra          person names in the odd rows of the MSRA NER test split; places and organizations count as neither
                leak nor false positive. For evaluation only: the original release has no license that allows training
  requests      30 Chinese developer requests written for this check (zh_requests.json)
  requests_dev  20 more, written while fixing false positives, so a development set
  european      100 validation rows in each of ai4privacy's languages other than English, scored per language; its
                license allows academic and non-commercial use, so it is read for evaluation only, as in bench/

The sets and the Privacy Filter's spans are cached in distill/data/.
Usage: uv run --group distill python distill/heldout.py MODEL_DIR [SET ...]"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "bench"))
from public_eval import ROWS, entities, evaluate, fetch  # noqa: E402
from spans import BINARY, RESULTS  # noqa: E402

DATA = HERE / "data"
PERSONAS = ["elena_stripe", "grace_deere_company", "hannah_salesforce"]
MSRA = ("PassbyGrocer/msra-ner", "01b2a657f0baa679c86154a3499706a069b73179")
MSRA_TAGS = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
AI4PRIVACY = ("ai4privacy/pii-masking-400k", "414d0a3b5798a152588a0828f1c08a5787de10f4")


def personas():
    query = lambda p: ROWS.format("TonicAI/Privacy-Bench", "ground_truth_spans", p)
    return [row for p in PERSONAS for row in fetch(query(p), lambda r: (r["text"], entities(r["ground_truth_spans"], "label")), 300)]


def creddata(n=400, around=10):
    """Credentials from files none of three earlier samples took anything from: the benchmark's evenly spaced sample
    (public_eval.creddata), one from the repositories that sample did not touch, and every credential of the
    repositories neither touched. The repositories overlap with those samples; the files do not."""
    root = RESULTS / "CredData"
    true = {}  # file -> its true credentials as (first line, start, last line, end, category, repository), 0-based
    for meta in sorted(root.glob("meta/*.csv")):
        for r in csv.DictReader(meta.open(encoding="utf-8")):
            if r["GroundTruth"] == "T" and r["ValueStart"] and r["ValueEnd"] and (root / r["FilePath"]).exists():
                true.setdefault(r["FilePath"], []).append((int(r["LineStart"]) - 1, int(r["ValueStart"]), int(r["LineEnd"]) - 1,
                                                           int(r["ValueEnd"]), r["Category"], r["RepoName"]))
    every = [(path, c) for path, creds in true.items() for c in creds]
    sample = every[:: len(every) // 400][:400]
    used_files, used_repos = {path for path, _ in sample}, {c[5] for _, c in sample}
    fresh = [(path, c) for path, c in every if c[5] not in used_repos]
    second = fresh[:: max(1, len(fresh) // 400)][:400]
    used_files |= {path for path, _ in second}
    touched = {c[5] for _, c in second}
    used_files |= {path for path, c in fresh if c[5] not in touched}
    rest = [(path, c) for path, c in every if path not in used_files]
    rows = []
    for path, (first, _, last, _, _, _) in rest[:: max(1, len(rest) // n)][:n]:
        lines = (root / path).read_text(encoding="utf-8", errors="replace").split("\n")
        top, bottom = max(0, first - around), min(len(lines), last + around + 1)
        at = {top: 0}  # line number -> offset of its first character in the excerpt
        for i in range(top, bottom):
            at[i + 1] = at[i] + len(lines[i]) + 1
        rows.append(["\n".join(lines[top:bottom]), [[at[a] + s, at[b] + e, kind] for a, s, b, e, kind, _ in true[path] if top <= a and b < bottom]])
    return rows


def msra():
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset(MSRA[0], split="test", revision=MSRA[1], streaming=True)):
        if i % 2 == 0:
            continue
        text, spans, kind = "", [], None
        for token, t in zip(r["tokens"], r["ner_tags"]):
            tag = MSRA_TAGS[t]
            if tag == "O":
                kind = None
            elif tag.startswith("B-") or tag[2:] != kind:
                kind = tag[2:]
                spans.append([len(text), len(text) + len(token), "name" if kind == "PER" else "ORGANIZATION"])
            else:
                spans[-1][1] = len(text) + len(token)
            text += token
        rows.append([text, spans])
    return rows


def european():
    """The first 100 validation rows of each language other than English, with the language as a third field."""
    from datasets import load_dataset
    rows, count = [], {}
    for r in load_dataset(AI4PRIVACY[0], split="validation", revision=AI4PRIVACY[1], streaming=True):
        if r["language"] != "en" and count.get(r["language"], 0) < 100:
            count[r["language"]] = count.get(r["language"], 0) + 1
            rows.append([r["source_text"], entities(r["privacy_mask"], "label"), r["language"]])
        if len(count) == 5 and min(count.values()) == 100:
            return rows


SETS = {
    "personas": ("PrivacyBench, three personas never trained on", personas),
    "creddata": ("CredData, credentials from files no earlier sample used", creddata),
    "msra": ("Chinese person names, MSRA NER test, odd rows", msra),
    "requests": ("Chinese developer requests", lambda: json.loads((HERE / "zh_requests.json").read_text())["fresh"]),
    "requests_dev": ("Chinese developer requests, development set", lambda: json.loads((HERE / "zh_requests.json").read_text())["dev"]),
    "european": ("ai4privacy validation", european),
}


def holdback(texts: list[str], student: Path | None) -> list[dict]:
    """`holdback spans` with the Privacy Filter, or with the distilled model in `student`."""
    env = {k: v for k, v in os.environ.items() if k != "HOLDBACK_STUDENT_DIR"} | {"HOLDBACK_DETECTOR": "privacy-filter"}
    if student:
        env |= {"HOLDBACK_DETECTOR": "distilled", "HOLDBACK_STUDENT_DIR": str(student)}
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "texts.json"
        f.write_text(json.dumps({"texts": texts}))
        return json.loads(subprocess.run([BINARY, "spans", str(f)], env=env, capture_output=True, text=True, check=True).stdout)


def cached(name: str, compute, texts: list[str] | None = None):
    """A result kept in distill/data/; with `texts`, recomputed when the texts it was computed for change."""
    path = DATA / f"heldout_{name}.json"
    if path.exists() and (saved := json.loads(path.read_text()))["texts"] == texts:
        return saved["value"]
    DATA.mkdir(exist_ok=True)
    value = compute()
    path.write_text(json.dumps({"texts": texts, "value": value}, ensure_ascii=False))
    return value


def main():
    model = Path(sys.argv[1]).resolve()
    for key in sys.argv[2:] or list(SETS):
        title, build = SETS[key]
        if key == "creddata" and not (RESULTS / "CredData/data").exists():
            print(f"{title}: skipped, bench/results/CredData is not downloaded")
            continue
        rows = cached(key, build)
        texts = [r[0] for r in rows]
        teacher = cached(f"{key}_teacher", lambda: holdback(texts, None), texts)
        student = holdback(texts, model)
        groups = {}  # the European set is scored per language
        for i, r in enumerate(rows):
            groups.setdefault(r[2] if len(r) > 2 else "", []).append(i)
        for group, idx in groups.items():
            part = [rows[i][:2] for i in idx]
            print(f"{title}{', ' + group if group else ''}, {len(part)} rows")
            evaluate(part, [teacher[i]["detected"] for i in idx], "Privacy Filter")
            evaluate(part, [teacher[i]["refined"] for i in idx], " + holdback rules")
            evaluate(part, [student[i]["detected"] for i in idx], model.name)
            evaluate(part, [student[i]["refined"] for i in idx], " + holdback rules")


if __name__ == "__main__":
    main()
