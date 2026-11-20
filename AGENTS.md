# holdback agent guide

holdback is a local proxy for coding agents. It replaces personal data and secrets in each request with numbered
placeholders and restores the real values in the reply. The README describes the behavior users rely on; this
file covers how to work on the code.

## Layout

| Path | Contents |
| --- | --- |
| `rust/holdback/src/main.rs` | CLI: `serve`, `run claude\|codex`, `status`, `review`, `keep`, `replace`, `hook`, `setup`, `doctor`, `spans <json>` |
| `rust/holdback/src/models.rs` | The two detectors' models: pinned revisions and SHA-256 sums, the choice by `HOLDBACK_DETECTOR`, the download and the cache's revision marker |
| `rust/holdback/src/native.rs` | The Privacy Filter runtime: memory-mapped q4 weights, Accelerate on macOS, `gemm` elsewhere |
| `rust/holdback/src/detect.rs` | The Privacy Filter's tokenizing, 128-token windows that end at a line break, separator or space where they can, Viterbi decoding into spans; `Detector` is either detector |
| `rust/holdback/src/student.rs` | The distilled model, the default detector: a 6-layer BERT encoder run from its memory-mapped safetensors file, its own tokenizer, 254-token windows, BIO labels into spans |
| `rust/holdback/src/precision.rs` | Rules that keep hosts, ports, paths, dates and IDs and replace only the secret part |
| `rust/holdback/src/vault.rs` | Placeholder assignment, the known-values pass, the user's keep and replace lists, restore, the saved map for hook mode |
| `rust/holdback/src/hooks.rs` | Hook mode: Claude Code hook events in, hook answers out |
| `rust/holdback/src/setup.rs` | `holdback doctor` and `holdback setup claude` |
| `install.sh`, `.github/workflows/release.yml` | The install script; the tag-triggered build, clean-machine check and draft release |
| `rust/holdback/src/protocols.rs` | Anthropic Messages, OpenAI Chat Completions and Responses payload walking |
| `rust/holdback/src/server.rs` | axum proxy, streaming rewrite, status page at `/_holdback` |
| `plugins/claude-code` | Claude Code plugin: starts the proxy, runs hook mode, `/holdback:status` |
| `bench` | Python benchmark and real-agent checks; they call the `holdback` binary |
| `distill` | The distilled student's training data from pinned public sources, its training, and test sets beyond the benchmarks; see `distill/README.md` |

The redaction rules exist only in Rust. The bench scripts get detector and rule output from `holdback spans`.

## Build and test

```bash
cd rust/holdback
cargo clippy --release --all-targets -- -D warnings
cargo build --release
cargo test --release
```

The tests run both detectors, so download both once: `holdback serve` fetches the distilled model (about 115 MB)
to `~/.cache/holdback/holdback-student`, and `HOLDBACK_DETECTOR=privacy-filter holdback serve` the Privacy Filter
(about 0.9 GB) to `~/.cache/holdback/privacy-filter-q4`. The parity tests read these caches even when
`HOLDBACK_STUDENT_DIR` or `HOLDBACK_MODEL_DIR` is set, because their fixtures belong to the pinned files.
`models.rs` pins both models' revisions and SHA-256 sums; a new published model means new pins there and a new
fixture from `distill/fixture.py`. CI runs the same steps on
Linux on every push. The release workflow builds all four targets and checks macOS
and Linux on clean runners; it runs on a tag and on request, because macOS minutes cost more. Check macOS
locally before a normal push.

## Invariants

- **Fail closed.** If a request cannot be redacted, holdback does not send it.
- **Original values stay in memory.** They never appear on the status page, in `holdback status`, in logs or
  in error messages. Two exceptions: `holdback review` prints them in the user's terminal, and hook mode saves
  the placeholder map to a file only the user can read, because its placeholders stay in the conversation. Beside
  it, `sightings.log` records the shape of each new value and of the text around it, never the value. The
  endpoints that handle real values answer only requests addressed to `127.0.0.1` or `localhost`.
- **Hook mode fails closed before a tool runs.** If `holdback hook` cannot answer, the tool call is denied.
  Claude Code drops a replacement output that does not fit the tool's own format and sends the original, so
  withholding after the fact is not reliable.
- **Stable placeholders.** A value keeps its placeholder for the life of the process, so resent history is
  byte-identical and prompt caching keeps working.
- **Thinking blocks pass through untouched.** Providers sign them.
- **Runtime parity.** `native::tests::matches_onnx_runtime` compares logits with ONNX Runtime's reference output
  in `src/testdata/logits.json`, and `student::tests::matches_pytorch` compares the distilled model's with
  PyTorch's in `src/testdata/student_logits.json`. A runtime change that fails them changes model output; do not
  loosen the tolerance to make them pass.
- **Tests use synthetic values.** No real names, emails or keys in code, tests or fixtures.

## Checking a change

Run the checks that cover what changed.

| Change | Checks |
| --- | --- |
| `native.rs`, `detect.rs` | `cargo test`; a runtime change must not move benchmark numbers, a change to the windows is checked like a rule change |
| `student.rs` | `cargo test --release`, then the benchmarks |
| `distill/` | Rebuild the data and train; then the benchmarks with `HOLDBACK_STUDENT_DIR` set to the new model (`bench/dev_eval.py --authors` among them) and `distill/heldout.py`, against the model card's numbers |
| `precision.rs` | `bench/detect_eval.py` (also `--seed 11`), `bench/public_eval.py` and `bench/dev_eval.py` (also `--held-out`), for the default detector and with `HOLDBACK_DETECTOR=privacy-filter` |
| `vault.rs`, the known-values pass | `bench/public_eval.py --sequence`, which runs each set through one map in order |
| `protocols.rs`, `server.rs` | `bench/check_proxy.py`, then `bench/check_agents.py` with Claude Code and Codex |
| `hooks.rs`, the plugin's hooks | `bench/check_hooks.py` and `bench/real_eval.py`, on more than one model |
| A rule that changes what hook mode saves | `bench/real_eval.py`: tasks passed with hook mode on and off, and the wrong share of saved values |
| A rule tuned for one case | Add the case to the test in `precision.rs`, which runs the real detector. A new task family in `bench/families.py` needs the task tables rerun on every model |

```bash
uv run python bench/detect_eval.py [--seed 11]
uv run python bench/public_eval.py
uv run python bench/task_eval.py --runs 1          # calls model APIs
```

`check_hooks.py` needs a logged-in `claude` command and starts what it needs.
`check_proxy.py` and `check_agents.py` need a running proxy with `HOLDBACK_DUMP` set; each docstring has the
command. The bench scripts run the release binary, so run `cargo build --release` after a change; `cargo test` does not
rebuild it. Results go to `bench/results/`, which is ignored by git. holdback's spans are cached there per
detector and computed again when the texts, the release binary or the model file change. The rules were tuned on
the generated tasks, so the public sets are the held-out check: report both.

## Configuration

| Variable | Effect |
| --- | --- |
| `HOLDBACK_PORT` | Proxy port, default 8787 |
| `HOLDBACK_UPSTREAMS` | Extra upstreams, `name=https://host/base,other=...` |
| `HOLDBACK_KEY_ENV` | Variable that holds the upstream key for `run codex`, default `OPENAI_API_KEY` |
| `HOLDBACK_DETECTOR` | `distilled` (default) or `privacy-filter` |
| `HOLDBACK_STUDENT_DIR` | The distilled model's directory instead of the cache: `config.json`, `model.safetensors` (f32) and `tokenizer.json`. holdback only reads it and stops with an error when a file is missing. The published model is [limbo23/holdback-student](https://huggingface.co/limbo23/holdback-student); `distill/` trains it |
| `HOLDBACK_MODEL_DIR` | The Privacy Filter's directory instead of the cache, read the same way |
| `HF_ENDPOINT` | A Hugging Face mirror to download the models from |
| `HOLDBACK_DUMP` | Append each redacted upstream request to this file, for the checks |
| `HOLDBACK_HOOKS` | `1` switches hook mode on for sessions that are not routed through the proxy |
| `HOLDBACK_STATE_DIR` | Where hook mode saves its placeholder map, default `~/.local/state/holdback` |
| `HOLDBACK_CONFIG_DIR` | Where `holdback keep` and `holdback replace` store their lists, default `~/.config/holdback` |

## Code and docs

- Delete superseded code instead of keeping it for reference.
- Keep `README.md` and `README.zh-CN.md` in sync. Every number in them comes from a bench script that a reader
  can rerun; update the tables when a change moves them.
