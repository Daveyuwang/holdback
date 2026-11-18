"""Detection on generated agent-style tasks, no model calls: private values that leak and answer-bearing values
that get replaced, for the detector alone and with holdback's rules. --compare adds the tools in spans.py.
Usage: uv run python bench/detect_eval.py [--seed 7] [--show] [--compare]"""
import argparse
from collections import Counter

from families import generate
from spans import COMPARED, mask, spans


def score(tasks, per_task, title, show=False):
    leaked = broken = 0
    by_family = Counter()
    for t, sp in zip(tasks, per_task):
        red = mask(t.ctx, sp)
        for s in t.secrets:
            if s in red:
                leaked += 1
                by_family[t.family] += 1
                if show:
                    print(f"  leak   [{t.family}] {s!r}")
        for k in t.keep:
            if k not in red:
                broken += 1
                by_family[t.family] += 1
                if show:
                    print(f"  broken [{t.family}] {k!r}")
    n_s, n_k = sum(len(t.secrets) for t in tasks), sum(len(t.keep) for t in tasks)
    print(f"{title:18s} leaked {leaked}/{n_s} ({leaked / n_s:.1%})   broken {broken}/{n_k} ({broken / n_k:.1%})   by family {dict(by_family)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    tasks = generate(seed=a.seed)
    texts, name = [t.ctx for t in tasks], f"spans_seed{a.seed}"
    for tool, source in COMPARED.items() if a.compare else ():
        score(tasks, source(texts, name), tool)
    rows = spans(texts, name)
    score(tasks, [r["detected"] for r in rows], "detector only")
    score(tasks, [r["refined"] for r in rows], "detector + rules", a.show)


if __name__ == "__main__":
    main()
