# Distilled detector

holdback can run a distilled detector in place of OpenAI's Privacy Filter: a 6-layer multilingual encoder that
labels the same eight categories, switched on with `HOLDBACK_STUDENT_DIR`. This folder builds its training data from
public sources at fixed versions, trains it, and scores it on the test sets beyond holdback's benchmarks. The
published model is [limbo23/holdback-student](https://huggingface.co/limbo23/holdback-student); its model card has
the results.

## Requirements

- holdback's release binary: `cargo build --release` in `rust/holdback`. The teacher labels come from `holdback
  spans` with the Privacy Filter, which the binary downloads on first use (about 0.9 GB).
- [uv](https://docs.astral.sh/uv/). The Python packages are the `distill` group in `pyproject.toml`; `uv run --group
  distill` installs them.
- A GPU: Apple silicon or CUDA. On an Apple M5, building the data takes about 40 minutes, most of it teacher labels,
  and training about 30 minutes.
- Network access to Hugging Face, crates.io and GitHub.

## Steps

```bash
uv run --group distill python distill/data.py constructed
uv run --group distill python distill/data.py natural
uv run --group distill python distill/train.py distill/runs/student
```

`distill/runs/student` is the directory holdback loads. `data.py natural` resumes where it stopped if interrupted.
`train.py --targets` picks the labels on natural text: `mix`, the published recipe, takes the Privacy Filter's spans
plus the dataset labels it missed on the public sets and only the planted values on developer text; `teacher`,
`gold` and `union` are the alternatives it was compared with.

## Training data

| Rows | Source | Labels | License |
| --- | --- | --- | --- |
| 16,000 | nvidia/Nemotron-PII, training rows 0 to 15,999 | the Privacy Filter's spans, plus dataset labels it missed | CC BY 4.0, NVIDIA |
| 6,000 | asgdaudahu/pii-bench-zh, rows 1,000 to 6,999 | the same | Apache-2.0 |
| 2,000 | Source of the crates in `rust/holdback/Cargo.lock` and of the Python packages in this environment, with planted values | the planted values | the packages' own licenses |
| 408 | Those crates' `Cargo.lock` files and file checksums, and git object listings of ripgrep, serde and fd | the planted values | the projects' own licenses |
| 1,500 | Generated code and config where values of the same shape are secrets under secret names and plain under checksum or ID names | exact | |
| 2,000 | Generated agent output: logs, cluster and CI output, chat, payments, configs | exact, by a fixed policy | |
| 1,500 | Generated secrets as source code writes them, next to values of the same shape that are not secrets | exact | |
| 1,495 | Chinese documentation of ant-design, vant, ahooks and umi, with planted values | the Privacy Filter's spans after holdback's rules, without person spans that cannot be names, plus the planted values | MIT |

The starting weights are 6 of the 12 layers of intfloat/multilingual-e5-small (MIT); the teacher is
openai/privacy-filter (Apache-2.0). ai4privacy/pii-masking-400k is not used: its license allows academic and
non-commercial use only.

None of these rows is read by an evaluation: the benchmarks use Nemotron-PII's test split, rows 0 to 399 of PII Bench
ZH, PrivacyBench and ai4privacy's validation split, and the developer text leaves out the crates that
`bench/dev_eval.py` and `bench/real_eval.py` read.

The datasets and repositories are pinned in `data.py`, the crates by `rust/holdback/Cargo.lock`, the Python packages
and Faker by `uv.lock`, so one commit of this repository builds the same rows on the same platform. Training is
seeded, but GPU arithmetic is not exactly repeatable; two seeds of this recipe differed by about 1.5 points of leaked
values on PrivacyBench.

## Evaluation

With `HOLDBACK_STUDENT_DIR` set, holdback's benchmark scripts run the distilled model. They cache its spans in
`bench/results` under names that start with `student_`; delete those files after training a new model.

```bash
export HOLDBACK_STUDENT_DIR=$PWD/distill/runs/student
uv run python bench/public_eval.py
uv run python bench/detect_eval.py
uv run python bench/detect_eval.py --seed 11
uv run python bench/dev_eval.py
uv run python bench/dev_eval.py --held-out
uv run --group distill python distill/heldout.py distill/runs/student
```

`heldout.py` scores the Privacy Filter and the given model on the sets beyond the benchmarks: three PrivacyBench
personas no benchmark uses, CredData credentials from files no earlier sample used (once CredData is downloaded, see
`bench/public_eval.py`), person names in the MSRA NER test split, and the Chinese developer requests in
`zh_requests.json`. MSRA is used for evaluation only, because its original release has no license that allows
training.

`fixture.py MODEL_DIR` writes the PyTorch logits that `student::tests::matches_pytorch` compares holdback's runtime
with. The token ids depend on the model's vocabulary, so the committed fixture is for the published model.

## The published model and a rebuild

The published model was trained with this recipe before the code moved into this repository. Its public rows were
the same. Its developer text, lockfile rows and the code lines around the constructed secrets came from the crates
and Python packages on the machine that built it, and an unseeded Faker drew some of its generated passwords and card
numbers. A rebuild from this folder, trained once with the same seed, scores as follows. The first number is leaked
labeled values, the second false-positive characters, both with holdback's rules.

| Test | Published model | Rebuilt here |
| --- | ---: | ---: |
| PrivacyBench, persona never trained on | 0.8% / 0.6% | 0.6% / 0.5% |
| PrivacyBench, three more personas | 3.2% / 0.4% | 2.9% / 0.4% |
| ai4privacy EN validation | 6.0% / 10.8% | 5.4% / 11.2% |
| CredData, files no earlier sample used | 20.2% / 4.7% | 18.5% / 5.1% |
| Generated agent tasks (seeds 7 and 11), values leaked of 288 and tasks broken of 186 | 0 and 0; 6 and 6 | 0 and 0; 6 and 6 |
| Developer text, false replacements per 100 KB (tuning, held-out) | 4, 2 | 5, 2 |
| Chinese names, MSRA NER test odd rows, 1,301 names | 13.1% / 1.4% | 4.2% / 2.5% |
| 30 Chinese developer requests, 10 values | 0 of 10 / 0.6% | 0 of 10 / 0.5% |

The two models differ by about as much as two seeds of one recipe, except on Chinese names: the rebuild misses fewer
of them and takes more common words for names.
