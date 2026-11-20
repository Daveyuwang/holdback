"""Write rust/holdback/src/testdata/student_logits.json: PyTorch's logits for three texts under one model, which
student::tests::matches_pytorch compares with holdback's runtime. The token ids depend on the model's vocabulary, so
write it for the model the test runs with, the published one.

Usage: uv run --group distill python distill/fixture.py MODEL_DIR"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

TEXTS = [
    "Contact Maria Example at maria.example@example.com or +1 415 555 0100 about invoice 4471.",
    "export API_TOKEN=ghp_" + 'EXAMPLEexample0EXAMPLEexample1234 checksum = "234113d19d0d7d613b40e86fb654acf958910802bcceab913a4f9e7cda03b1a4"',
    "张伟的电话是 138 0013 8000，住在北京市海淀区中关村大街 27 号。",
]

tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForTokenClassification.from_pretrained(sys.argv[1]).eval()
windows = [tok(t, add_special_tokens=False)["input_ids"] for t in TEXTS]
logits = []
with torch.no_grad():
    for w in windows:  # each window attends only to itself, as in the runtime
        out = model(input_ids=torch.tensor([[tok.cls_token_id] + w + [tok.sep_token_id]])).logits[0, 1:-1]
        logits += [round(x, 3) for x in out.flatten().tolist()]
path = Path(__file__).parent.parent / "rust/holdback/src/testdata/student_logits.json"
path.write_text(json.dumps({"windows": windows, "logits": logits}))
print(f"{path}: {len(windows)} windows, {len(logits)} logits")
