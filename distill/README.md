# Distilled detector

holdback's default detector since 0.2 is a distilled model: a 6-layer multilingual encoder that labels the same
eight categories as OpenAI's Privacy Filter, which taught it. This folder builds its training data from public
sources at fixed versions, trains it, and scores it on the test sets beyond holdback's benchmarks. The published
model is [limbo23/holdback-student](https://huggingface.co/limbo23/holdback-student); its model card has the
results.

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
| 2,000 | Source of the crates in `rust/holdback/Cargo.lock` and of the Python packages in `python_packages.txt`, with planted values | the planted values and the real people's names and emails | the packages' own licenses |
| 408 | Those crates' `Cargo.lock` files and file checksums, and git object listings of ripgrep, serde and fd | the same | the projects' own licenses |
| 1,500 | Generated code and config where values of the same shape are secrets under secret names and plain under checksum or ID names | exact | |
| 2,000 | Generated agent output: logs, cluster and CI output, chat, payments, configs | exact, by a fixed policy | |
| 1,500 | Generated secrets as source code writes them, next to values of the same shape that are not secrets | exact | |
| 1,495 | Chinese documentation of ant-design, vant, ahooks and umi, with planted values | the Privacy Filter's spans after holdback's rules, without person spans that cannot be names, plus the planted values | MIT |

The starting weights are 6 of the 12 layers of intfloat/multilingual-e5-small (MIT); the teacher is
openai/privacy-filter (Apache-2.0). ai4privacy/pii-masking-400k is not used: its license allows academic and
non-commercial use only.

Real people's names and emails in text taken from repositories, such as the `authors` of a `Cargo.toml`, a
`Name <email>` line or a copyright holder, are labeled by `people()` in `bench/dev_eval.py`; names of organizations
and project teams are not.

No training row repeats a text that holdback's public benchmark reads: `data.py` leaves out every row of the four
public sets. Those are Nemotron-PII's test split, PrivacyBench, ai4privacy's validation split and four pages of PII
Bench ZH, three of which lie in the slice taken here. The developer text leaves out the crates that
`bench/dev_eval.py` and `bench/real_eval.py` read.

The datasets and repositories are pinned in `data.py`, the crates by `rust/holdback/Cargo.lock`, the Python packages
and Faker by `uv.lock`, so one commit of this repository builds the same rows on the same platform. `data.py` fixes
Python's hash seed, because Faker builds some of its value lists from sets. Training is
seeded, but GPU arithmetic is not exactly repeatable; two seeds of this recipe differed by about 1.5 points of leaked
values on PrivacyBench.

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

`dev_eval.py --authors` counts the real names and emails replaced in the manifests of the nine crates
`bench/real_eval.py` uses.

`heldout.py` scores the Privacy Filter and the given model on the sets beyond the benchmarks: three PrivacyBench
personas no benchmark uses, CredData credentials from files no earlier sample used (once CredData is downloaded, see
`bench/public_eval.py`), person names in the MSRA NER test split, the Chinese developer requests in
`zh_requests.json`, and ai4privacy's French, German, Italian, Spanish and Dutch rows. MSRA and ai4privacy are used
for evaluation only, because their licenses do not allow training a model meant to ship.

`fixture.py MODEL_DIR` writes the PyTorch logits that `student::tests::matches_pytorch` compares holdback's runtime
with. The token ids depend on the model's vocabulary, so the committed fixture is for the published model.

## The published model

The model on Hugging Face is this folder's output at commit 362aa9e with the default settings: `data.py
constructed`, `data.py natural`, then `train.py`. Its model card has the results.

A first upload was trained with the same recipe before the code moved into this repository, with developer text from
other crates and packages and some generated passwords and card numbers drawn without a seed. On the tests above, the
rebuild differed from it by about as much as two seeds of one recipe, except on Chinese names in news: it missed 4.2%
of them instead of 13.1%, and replaced 2.5% of other characters instead of 1.4%.

Four faults in the published model's training data were fixed after it was built:

- Its PII Bench ZH rows, 1,000 to 6,999, included 300 of the 400 rows the benchmark reads, so its PII Bench ZH
  numbers are not held out. `data.py` now leaves out every text the public benchmark reads.
- Its developer text labeled everything but the planted values as not private, the names and emails of real
  maintainers in 127 `Cargo.toml` excerpts included, so the model leaves such names and emails in place. `data.py`
  now labels them with `people()`.
- `category()` mapped Nemotron-PII's `mac_address` label to an address, because it looked for "address" before "mac".
- Faker drew Italian city names from a list it builds from a set, so the names followed the hash seed of the run: a
  rebuild names a different city in 136 of the 2,000 agent rows. `data.py` now fixes the seed.
