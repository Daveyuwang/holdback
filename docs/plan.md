# Status and next steps

## 0.2

holdback keeps personal data and secrets out of what coding agents send to cloud models, in two modes: a
local proxy that redacts whole requests, and hooks inside Claude Code that redact what tools return. Since 0.2 its
detector is a model of 28M parameters distilled from OpenAI's Privacy Filter, which `HOLDBACK_DETECTOR` can bring
back. The README has the measurements and the limits.

What the release rests on:

- 1.0% and 5.8% of labeled personal values missed on the two public sets the distilled model never trained on,
  against 9.7% and 12.7% with the Privacy Filter, and no value leaked on the generated agent tasks.
- All 45 real agent tasks pass with hook mode on and off on Haiku 5.5, Sonnet 5.5 and Opus 5.5. No planted secret
  reaches the provider, and of the maintainers' details only a lone first name and a handle do.
- 2 to 3 false replacements per 100 KB of developer output, and 12 of 14 maintainers' names and all 12 emails
  replaced in the `authors` lists of the real-session crates.
- A request is 6 to 7 times faster than with the Privacy Filter, and the model is a 115 MB download.
- Binaries for macOS and Linux, built and checked on clean machines by the release workflow.

## Known gaps

- In French, German, Italian, Spanish and Dutch the distilled model misses 9 to 18% of labeled values, where the
  Privacy Filter misses 5 to 16%; other scripts are untested.
- In Chinese it sometimes takes a common word for a name.
- Names in developer text that `people()` does not recognize, such as a lone first name or a handle without an
  email, train as not private, so the model keeps them.
- Both detectors sometimes take a number that stands alone, such as a count or a process id, for an account number.
- Task success under redaction was last measured with the Privacy Filter; it needs a rerun with the distilled
  model, which calls model APIs.
- Hook mode does not cover the system prompt, what the user types, or the output of a failed tool call.
- The real-session benchmark is narrow: short tasks on Rust crates.
- Codex is checked through an OpenAI-compatible provider, not with GPT models.

## Next

1. A rule that keeps a number standing alone, such as a count, unless the text around it names an account.
2. Rerun the task-success benchmark with the distilled model.
3. Widen the real-session benchmark: other languages, larger tool output, more kinds of tools.
4. At the next retrain, label the developer text with the teacher's person and email spans as well as `people()`'s.
5. Homebrew and npm packages, when asked for.

Reports of wrong replacements are welcome. `holdback review --masked` output is safe to attach to an issue.
