# Status and next steps

## 0.1.0

holdback keeps personal data and secrets out of what coding agents send to cloud models, in two modes: a
local proxy that redacts whole requests, and hooks inside Claude Code that redact what tools return. The
README has the measurements and the limits.

What the first release rests on:

- Task success stays at 98 to 100% on the task benchmark, and at 44 to 45 of 45 real agent tasks with hook
  mode on, the same as with it off.
- No planted value appears in any recorded request in the hook-mode checks, on Haiku 5.5, Sonnet 5.5 and
  Opus 5.5.
- 1 to 2 false replacements per 100 KB of developer output.
- Binaries for macOS and Linux, built and checked on clean machines by the release workflow.

## Known gaps

- 10 to 14% of labeled personal values are missed on public sets; Chinese personal names most of all.
- Hook mode does not cover the system prompt, what the user types, or the output of a failed tool call.
- The real-session benchmark is narrow: short tasks on Rust crates.
- Checking new content is slow for large outputs: about 0.7 s per 10 KB.
- Codex is checked through an OpenAI-compatible provider, not with GPT models.

## Next

1. Widen the real-session benchmark: other languages, larger tool output, more kinds of tools.
2. Recall on Chinese personal names.
3. Faster checking of new content.
4. Homebrew and npm packages, when asked for.

Reports of wrong replacements are welcome. `holdback review --masked` output is safe to attach to an issue.
