"""The student: a small multilingual encoder with a token-classification head. The label scheme, loading, batching
and vocabulary pruning that train.py uses; holdback's student.rs reads the saved directory."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

CATEGORIES = ["account_number", "private_address", "private_date", "private_email", "private_person", "private_phone", "private_url", "secret"]
LABELS = ["O"] + [f"{p}-{c}" for c in CATEGORIES for p in "BI"]
BODY = 254  # tokens per window, without the two special tokens; student.rs uses the same
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(checkpoint: str, layers: int | None = None):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(LABELS))
    if layers and layers < len(model.bert.encoder.layer):
        model.bert.encoder.layer = model.bert.encoder.layer[:layers]
        model.config.num_hidden_layers = layers
    return tok, model


def token_labels(offsets, spans) -> list[int]:
    """BIO label ids for tokens with character `offsets`, from [start, end, category] spans."""
    out = [0] * len(offsets)
    for s, e, cat in sorted(spans):
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts < e and s < te and out[i] == 0:
                out[i] = 1 + 2 * CATEGORIES.index(cat) + (0 if first else 1)
                first = False
    return out


def batches(tok, windows, size):
    """Padded (input_ids, attention_mask, order) batches over token-id windows, longest first."""
    order = sorted(range(len(windows)), key=lambda i: -len(windows[i]))
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        rows = [[tok.cls_token_id] + windows[j] + [tok.sep_token_id] for j in idx]
        width = -(-max(map(len, rows)) // 32) * 32  # few distinct shapes: the GPU backend caches memory per shape
        ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        yield ids, mask, idx


def prune(src: Path, dst: Path, texts) -> None:
    """The model in `src` with its vocabulary cut from 250k pieces to the ones it can use: every piece `texts`
    tokenize into, every single-character piece (so any text still tokenizes) and the special tokens.

    A Unigram tokenizer picks the best-scoring split among the pieces it has. Every piece of a text's split is kept,
    with its score, and pruning only removes options, so every one of `texts` splits exactly as before; other text
    splits into smaller kept pieces."""
    from tokenizers import Tokenizer
    spec = json.loads((src / "tokenizer.json").read_text())
    vocab = spec["model"]["vocab"]
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    keep = set(range(4)) | {i for i, (piece, _) in enumerate(vocab) if len(piece.lstrip("▁")) <= 1}
    for text in texts:
        keep.update(tok.encode(text, add_special_tokens=False).ids)
    keep.update(a["id"] for a in spec["added_tokens"])
    order = sorted(keep)
    new_id = {old: new for new, old in enumerate(order)}
    spec["model"]["vocab"] = [vocab[i] for i in order]
    spec["model"]["unk_id"] = new_id[spec["model"]["unk_id"]]
    for a in spec["added_tokens"]:
        a["id"] = new_id[a["id"]]
    for special in spec["post_processor"].get("special_tokens", {}).values():
        special["ids"] = [new_id[i] for i in special["ids"]]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False))
    (dst / "tokenizer_config.json").write_text((src / "tokenizer_config.json").read_text())
    model = AutoModelForTokenClassification.from_pretrained(src)
    emb = model.get_input_embeddings().weight.data
    small = torch.nn.Embedding(len(order), emb.shape[1])
    small.weight.data = emb[torch.tensor(order)].clone()
    model.set_input_embeddings(small)
    model.config.vocab_size = len(order)
    model.save_pretrained(dst)
    print(f"vocabulary: kept {len(order)} of {len(vocab)} pieces", flush=True)


def save(model, tok, out: Path) -> None:
    """The directory holdback loads: config.json with the label names, model.safetensors and the tokenizer."""
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {name: i for i, name in enumerate(LABELS)}
    model.save_pretrained(out)
    tok.save_pretrained(out)
    config = json.loads((out / "tokenizer_config.json").read_text())
    for k in ("is_local", "local_files_only"):  # where this copy was loaded from, not part of the tokenizer
        config.pop(k, None)
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
