# Status and next steps

## 0.1.1

holdback keeps personal data and secrets out of what coding agents send to cloud models, in two modes: a
local proxy that redacts whole requests, and hooks inside Claude Code that redact what tools return. The
README has the measurements and the limits.

What the release rests on:

- Task success stays at 98 to 100% on the task benchmark, and at 44 to 45 of 45 real agent tasks with hook
  mode on, the same as with it off.
- No planted value appears in any recorded request in the hook-mode checks, on Haiku 5.5, Sonnet 5.5 and
  Opus 5.5.
- 0 to 4 false replacements per 100 KB of developer output.
- Binaries for macOS and Linux, built and checked on clean machines by the release workflow.

## Known gaps

- 10 to 14% of labeled personal values are missed on public sets; Chinese personal names most of all.
- Hook mode does not cover the system prompt, what the user types, or the output of a failed tool call.
- The real-session benchmark is narrow: short tasks on Rust crates.
- Checking new content is slow for large outputs: about 0.7 s per 10 KB.
- Codex is checked through an OpenAI-compatible provider, not with GPT models.

## Next

1. The distilled detector is on [Hugging Face](https://huggingface.co/limbo23/holdback-student) and runs behind
   `HOLDBACK_STUDENT_DIR`. In the first measurements it missed 13% of the Chinese names in a news test set where
   the Privacy Filter missed 67%, though it takes more common words for names. It found secrets in code about as
   often and checked text 7 to 14 times faster on an Apple M5. Still to do: publish its training and evaluation
   code, and decide whether it becomes the default.
2. Widen the real-session benchmark: other languages, larger tool output, more kinds of tools.
3. Homebrew and npm packages, when asked for.

Reports of wrong replacements are welcome. `holdback review --masked` output is safe to attach to an issue.
