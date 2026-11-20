"""Held-out check on public sets with span labels. The rules were written before any of them was run; the README
names the two rules changed after a first run:
PII Bench ZH (Chinese, Apache-2.0), the English rows of ai4privacy/pii-masking-400k (its own license),
NVIDIA's Nemotron-PII (English documents, CC BY 4.0) and one persona of Tonic's PrivacyBench (workplace email and
chat, CC BY 4.0). The data is downloaded, not redistributed.

Samsung's CredData (secrets in source files, values obfuscated) is added when bench/results/CredData exists:
  git clone https://github.com/Samsung/CredData bench/results/CredData && cd bench/results/CredData
  uv run --no-project --with pybase62 python download_data.py --data_dir data
The download clones 337 repositories (13 GB in tmp/, which can be deleted afterwards) and needs git-lfs.

Leak: a labeled value with no character replaced. False positive: a replaced character outside every label.
Labels for attributes that do not identify a person on their own (dates, occupation, country, organization)
count as neither. --compare adds the tools in spans.py. --sequence runs each set through one placeholder map in
order, as one long session would, to measure the pass that replaces a known value wherever it appears.
Usage: uv run python bench/public_eval.py [--n 400] [--compare]"""
import argparse
import ast
import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter

from spans import COMPARED, RESULTS, cached, spans

API = "https://datasets-server.huggingface.co/{}&offset={}&length=100"
ROWS = "rows?dataset={}&config={}&split={}"
ATTRIBUTES = {"date", "time", "date_time", "occupation", "company_name", "url", "country", "county", "state", "race_ethnicity",
              "religious_belief", "employment_status", "gender", "age", "education_level", "political_view", "blood_type",
              "sexuality", "language", "ORGANIZATION"}


def entities(spans, kind):
    return [[g["start"], g["end"], g[kind]] for g in (ast.literal_eval(spans) if isinstance(spans, str) else spans)]


SETS = {  # cache name -> (title, query, row -> (text, labeled spans))
    "pii_bench_zh": ("PII Bench ZH", ROWS.format("asgdaudahu/pii-bench-zh", "default", "train"),
                     lambda r: (r["text"], entities(r["entities"], "type"))),
    "ai4privacy_en": ("ai4privacy EN", "filter?dataset=ai4privacy/pii-masking-400k&config=default&split=validation&where=%22language%22%3D%27en%27",
                      lambda r: (r["source_text"], entities(r["privacy_mask"], "label"))),
    "nemotron_pii": ("Nemotron-PII", ROWS.format("nvidia/Nemotron-PII", "default", "test"),
                     lambda r: (r["text"], entities(r["spans"], "label"))),
    "privacy_bench": ("PrivacyBench", ROWS.format("TonicAI/Privacy-Bench", "ground_truth_spans", "aaron_pfizer"),
                      lambda r: (r["text"], entities(r["ground_truth_spans"], "label"))),
}


def fetch(query, parse, n):
    """n rows as (text, labeled spans), in pages of 100 spread evenly over the set."""
    def page(offset):
        for wait in (0, 30, 60, 90):  # a filter query fails until its index has loaded, which takes about a minute
            time.sleep(wait)
            try:
                with urllib.request.urlopen(API.format(query, offset), timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                error = e
        raise error
    step = page(0)["num_rows_total"] // (n // 100)
    return [parse(x["row"]) for i in range(n // 100) for x in page(i * step)["rows"]]


def credentials() -> dict:
    """CredData's true credentials: file -> [(first line, start in it, last line, end in it, category, repository)],
    lines 0-based."""
    root, true = RESULTS / "CredData", {}
    for meta in sorted(root.glob("meta/*.csv")):
        for r in csv.DictReader(meta.open(encoding="utf-8")):
            if r["GroundTruth"] == "T" and r["ValueStart"] and r["ValueEnd"] and (root / r["FilePath"]).exists():
                true.setdefault(r["FilePath"], []).append((int(r["LineStart"]) - 1, int(r["ValueStart"]), int(r["LineEnd"]) - 1,
                                                           int(r["ValueEnd"]), r["Category"], r["RepoName"]))
    return true


def spread(items: list, n: int) -> list:
    """n items spread evenly over the list."""
    return items[:: max(1, len(items) // n)][:n]


def excerpt(true: dict, path: str, cred: tuple, around: int = 10) -> list:
    """The credential's lines with `around` lines before and after, and every true credential inside them."""
    lines = (RESULTS / "CredData" / path).read_text(encoding="utf-8", errors="replace").split("\n")
    top, bottom = max(0, cred[0] - around), min(len(lines), cred[2] + around + 1)
    at = {top: 0}  # line number -> offset of its first character in the excerpt
    for i in range(top, bottom):
        at[i + 1] = at[i] + len(lines[i]) + 1
    return ["\n".join(lines[top:bottom]), [[at[a] + s, at[b] + e, kind] for a, s, b, e, kind, _ in true[path] if top <= a and b < bottom]]


def creddata(n):
    """n true credentials spread evenly over CredData, each with ten lines of code on either side."""
    true = credentials()
    return [excerpt(true, path, cred) for path, cred in spread([(p, c) for p, creds in true.items() for c in creds], n)]


def evaluate(rows, per_row, title):
    leaked = total = fp = outside = 0
    by_type = Counter()
    for (text, labels), sp in zip(rows, per_row):
        replaced = {i for s, e, _ in sp for i in range(s, e)}
        labeled = {i for s, e, _ in labels for i in range(s, e)}
        for s, e, kind in labels:
            if kind in ATTRIBUTES:
                continue
            total += 1
            if not replaced & set(range(s, e)):
                leaked += 1
                by_type[kind] += 1
        others = [i for i, ch in enumerate(text) if i not in labeled and not ch.isspace()]
        outside += len(others)
        fp += sum(i in replaced for i in others)
    print(f"  {title:18s} leaked {leaked}/{total} ({leaked / total:.1%})   false-positive characters {fp / outside:.1%}   leaks by type {dict(by_type.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--sequence", action="store_true", help="also run each set through one placeholder map in order")
    a = ap.parse_args()
    sets = {key: (title, lambda q=query, p=parse: fetch(q, p, a.n)) for key, (title, query, parse) in SETS.items()}
    if (RESULTS / "CredData/data").exists():
        sets["creddata"] = ("CredData", lambda: creddata(a.n))
    for key, (title, load) in sets.items():
        rows = cached(f"{key}_{a.n}", load)
        texts, name = [text for text, _ in rows], f"{key}_spans_{a.n}"
        print(f"{title}, {len(rows)} rows")
        for tool, source in COMPARED.items() if a.compare else ():
            evaluate(rows, source(texts, name), tool)
        per = spans(texts, name)
        evaluate(rows, [r["detected"] for r in per], "detector only")
        evaluate(rows, [r["refined"] for r in per], "detector + rules")
        if a.sequence:
            per = spans(texts, name, sequence=True)
            evaluate(rows, [r["all"] for r in per], "one map, spread all")
            evaluate(rows, [r["twice"] for r in per], "one map, twice")


if __name__ == "__main__":
    main()
