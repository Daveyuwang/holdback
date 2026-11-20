# Distilled detector

holdback's default detector since 0.2 is a distilled model: a 6-layer multilingual encoder that labels the same
eight categories as OpenAI's Privacy Filter, which taught it. This folder builds its training data from public
sources at fixed versions, trains it, and scores it on the test sets beyond holdback's benchmarks. The published
model, [limbo23/holdback-student](https://huggingface.co/limbo23/holdback-student), is this folder's output at
commit a270fc6 with the steps below; its model card has the results and the earlier versions.

## Requirements

- holdback's release binary: `cargo build --release` in `rust/holdback`. The teacher labels come from `holdback
  spans` with the Privacy Filter, which the binary downloads on first use (about 0.9 GB).
- [uv](https://docs.astral.sh/uv/) and `uv sync --group distill --group compare`. The `distill` group runs the
  scripts; the developer text also reads the source of the packages in `python_packages.txt`, some of which only
  the `compare` group installs.
- A GPU: Apple silicon or CUDA. On an Apple M5, building the data takes about 40 minutes, most of it teacher labels,
  and training about 30 minutes.
- Network access to Hugging Face, crates.io and GitHub.

## Steps

```bash
uv sync --group distill --group compare
uv run --group distill python distill/data.py constructed
uv run --group distill python distill/data.py natural
uv run --group distill python distill/train.py distill/runs/student
```

`distill/runs/student` is the directory holdback loads. `data.py natural` resumes where it stopped if interrupted.
`train.py` labels each row with its `teacher` spans, where it has them, plus every `gold` value those spans do not
touch.

## Training data

| Rows | Source | Labels | License |
| --- | --- | --- | --- |
| 16,000 | nvidia/Nemotron-PII, training rows 0 to 15,999 | the Privacy Filter's spans, plus dataset labels it missed | CC BY 4.0, NVIDIA |
| 6,000 | asgdaudahu/pii-bench-zh, from row 1,000 on, without the rows the benchmark reads | the same | Apache-2.0 |
| 2,000 | Source of the crates in `rust/holdback/Cargo.lock` and of the Python packages in `python_packages.txt`, with planted values | the planted values and the real people `people()` finds | the packages' own licenses |
| 408 | Those crates' `Cargo.lock` files and file checksums, and git object listings of ripgrep, serde and fd | the planted values | the projects' own licenses |
| 1,500 | Generated code and config where values of the same shape are secrets under secret names and plain under checksum or ID names | exact | |
| 2,000 | Generated agent output: logs, cluster and CI output, chat, payments, configs | exact, by a fixed policy | |
| 1,500 | Generated secrets as source code writes them, next to values of the same shape that are not secrets | exact | |
| 1,495 | Chinese documentation of ant-design, vant, ahooks and umi, with planted values | the Privacy Filter's spans after holdback's rules, without person spans that cannot be names, plus the planted values | MIT |

The starting weights are 6 of the 12 layers of intfloat/multilingual-e5-small (MIT); the teacher is
openai/privacy-filter (Apache-2.0). ai4privacy/pii-masking-400k is not used: its license allows academic and
non-commercial use only.

In the developer text, `people()` in `bench/dev_eval.py` labels real people's names and emails: the `authors` of a
`Cargo.toml`, a `Name <email>` line, an author comment, a copyright holder. It leaves out organizations and project
teams. A name it does not recognize, such as a lone first name or a handle without an email, trains as not private.

Training reads no row of holdback's public benchmark: Nemotron-PII's test split, PrivacyBench and ai4privacy are not
training sources, and `data.py` skips the three pages of 100 PII Bench ZH rows that lie in the slice taken here. The
developer text leaves out the crates that `bench/dev_eval.py` and `bench/real_eval.py` read, but not their
maintainers' other crates.

The datasets and repositories are pinned in `data.py`, the crates by `rust/holdback/Cargo.lock`, the Python packages
and Faker by `uv.lock`, so one commit of this repository builds the same rows on the same platform. `data.py` fixes
Python's hash seed, because Faker builds some of its value lists from sets. Training is seeded, but GPU arithmetic
is not exactly repeatable; two seeds of this recipe differed by up to 3.2 points of leaked values on the tests below.

## Evaluation

holdback's benchmark scripts run the published model. With `HOLDBACK_STUDENT_DIR` set to another model's
directory they run that one instead and cache its spans in `bench/results` under names that start with `student_`,
computed again when its model file changes.

```bash
export HOLDBACK_STUDENT_DIR=$PWD/distill/runs/student
uv run python bench/public_eval.py
uv run python bench/detect_eval.py
uv run python bench/detect_eval.py --seed 11
uv run python bench/dev_eval.py
uv run python bench/dev_eval.py --held-out
uv run python bench/dev_eval.py --authors
uv run --group distill python distill/heldout.py distill/runs/student
```

`heldout.py` scores the Privacy Filter and the given model on the sets beyond the benchmarks: three PrivacyBench
personas no benchmark uses, CredData credentials from files no earlier sample used (once CredData is downloaded, see
`bench/public_eval.py`), person names in the MSRA NER test split, the Chinese developer requests in
`zh_requests.json`, and ai4privacy's French, German, Italian, Spanish and Dutch rows. MSRA and ai4privacy are used
for evaluation only, because their licenses do not allow training a model meant to ship.

`fixture.py MODEL_DIR` writes the PyTorch logits that `student::tests::matches_pytorch` compares holdback's runtime
with. The token ids depend on the model's vocabulary, so the committed fixture is for the published model.
