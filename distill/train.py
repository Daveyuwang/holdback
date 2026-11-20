"""Train the student on data/natural.jsonl and data/constructed.jsonl, from the first layers of
intfloat/multilingual-e5-small with its vocabulary cut to the pieces the training text uses. A row's labels are the
Privacy Filter's spans in its `teacher` field, if it has one, plus every labeled value those spans do not touch.

Usage: uv run --group distill python distill/train.py OUT_DIR [--seed 0]"""
import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from student import BODY, DEVICE, batches, load, prune, save, token_labels

HERE = Path(__file__).parent


def targets(row):
    teacher = row.get("teacher", [])
    return teacher + [g for g in row["gold"] if not any(g[0] < e and s < g[1] for s, e, _ in teacher)]


def windows(tok, rows):
    ids, labels = [], []
    for row in rows:
        enc = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
        lab = token_labels(enc["offset_mapping"], targets(row))
        for i in range(0, len(lab), BODY):
            ids.append(enc["input_ids"][i:i + BODY])
            labels.append(lab[i:i + BODY])
    return ids, labels


def run(model, tok, ids, labels, size, step=None):
    """One pass. With `step` it trains; without, it returns token counts for precision and recall of non-O."""
    hit = pred_n = gold_n = 0
    for x, mask, idx in batches(tok, ids, size):
        width = x.shape[1]
        y = torch.tensor([[-100] + labels[j] + [-100] * (width - 1 - len(labels[j])) for j in idx])
        out = model(input_ids=x.to(DEVICE), attention_mask=mask.to(DEVICE), labels=y.to(DEVICE))
        if step:
            step(out.loss)
        else:
            p = out.logits.argmax(-1).cpu()
            keep = y != -100
            hit += ((p == y) & (y > 0) & keep).sum().item()
            pred_n += ((p > 0) & keep).sum().item()
            gold_n += ((y > 0) & keep).sum().item()
    return hit, pred_n, gold_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--init", default="intfloat/multilingual-e5-small")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/natural.jsonl", help="natural text with teacher labels, relative to distill/")
    ap.add_argument("--constructed", default="data/constructed.jsonl", help="rows with exact labels, relative to distill/")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(line) for name in (a.data, a.constructed) for line in open(HERE / name)]
    random.shuffle(rows)
    held = len(rows) // 50
    with tempfile.TemporaryDirectory() as d:  # keep only the pieces the training text uses: a small embedding, gradient and optimizer state
        tok, model = load(a.init, a.layers)
        model.save_pretrained(Path(d) / "full")
        tok.save_pretrained(Path(d) / "full")
        prune(Path(d) / "full", Path(d) / "pruned", (r["text"] for r in rows))
        tok, model = load(str(Path(d) / "pruned"))
    model.to(DEVICE)
    if DEVICE == "mps":  # a cap, so a run cannot push the rest of the Mac into swap; about 8 GB on a 24 GB Mac
        torch.mps.set_per_process_memory_fraction(0.45)
    val = windows(tok, rows[:held])
    ids, labels = windows(tok, rows[held:])
    print(f"{len(rows) - held} texts, {len(ids)} windows, {sum(map(len, ids)) / 1e6:.1f}M tokens; validation {len(val[0])} windows on {DEVICE}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(ids) // a.batch + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    seen, t0 = 0, time.time()

    def step(loss):
        nonlocal seen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        seen += 1
        if DEVICE == "mps" and seen % 10 == 0:  # freed buffers otherwise pile up until the cap is reached
            torch.mps.empty_cache()
        if seen % 200 == 0:
            print(f"  step {seen}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    for epoch in range(a.epochs):
        model.train()
        order = list(range(len(ids)))
        random.shuffle(order)
        # batches() sorts by length, so shuffle in blocks: similar lengths together, block order random
        for i in range(0, len(order), a.batch * 50):
            block = order[i:i + a.batch * 50]
            run(model, tok, [ids[j] for j in block], [labels[j] for j in block], a.batch, step)
            if DEVICE == "mps":
                torch.mps.empty_cache()
        model.eval()
        with torch.no_grad():
            hit, pred_n, gold_n = run(model, tok, *val, 64)
        print(f"epoch {epoch + 1}: validation token precision {hit / max(pred_n, 1):.3f} recall {hit / max(gold_n, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    save(model, tok, out)
    (out / "training_args.json").write_text(json.dumps(vars(a), indent=1) + "\n")


main()
