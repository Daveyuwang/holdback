# Status and next steps

## 0.2

holdback keeps personal data and secrets out of what coding agents send to cloud models, in two modes: a
local proxy that redacts whole requests, and hooks inside Claude Code that redact what tools return. Since 0.2 its
detector is a model of 28M parameters distilled from OpenAI's Privacy Filter, which `HOLDBACK_DETECTOR` can bring
back. The README has the measurements and the limits.

What the release rests on:

- 1.0% and 5.8% of labeled personal values missed on the two public sets the distilled model never trained on,
  against 9.7% and 12.8% with the Privacy Filter, and no value leaked on the generated agent tasks.
- All 45 real agent tasks pass with hook mode on and off on Haiku 5.5, Sonnet 5.5 and Opus 5.5. No planted secret
  reaches the provider, and of the maintainers' details only a lone first name and a handle do.
- 2 false replacements per 100 KB of developer output, and 12 of 14 maintainers' names and all 12 emails
  replaced in the `authors` lists of the real-session crates.
- A request is 6 to 7 times faster than with the Privacy Filter, and the model is a 115 MB download.
- Binaries for macOS and Linux, built and checked on clean machines by the release workflow.

## Known gaps

- In French, German, Italian, Spanish and Dutch the distilled model misses 9 to 18% of labeled values, where the
  Privacy Filter misses 5 to 16%; other scripts are untested.
- In Chinese it sometimes takes a common word for a name.
- Names in developer text that `people()` does not recognize, such as a lone first name or a handle without an
  email, train as not private, so the model keeps them.
- Hook mode does not cover the system prompt, what the user types, or the output of a failed tool call.
- The real-session benchmark is narrow: short tasks on Rust crates.
- Codex is checked through an OpenAI-compatible provider, not with GPT models.

## Wrap-up

The rules for wrong replacements seen in real sessions, the demo, the technical report and the task-success rerun
(100% on DeepSeek, Claude Haiku and Qwen) are done. This round of work ends when the steps below are. No new
features; each step ends with a check that a reader can rerun or see.

| Step | Done when |
| --- | --- |
| The owner column of `ls -l`: the distilled model replaces a login name there in some listings and keeps it in others | Replaced in every line unless it is a system account; cases in `precision.rs`; benchmarks no worse |
| 0.2.1 with the fixes since 0.2.0 | Released; the install script installs it on clean machines; real sessions pass on Haiku, Sonnet and Opus |
| An issue form for a wrong replacement and for a missed value | A report needs the version, the mode and masked output, nothing private |
| Real use: hook mode in the maintainer's daily work, on 0.2.1, for at least three working days and 200 new values, read from the shapes in `sightings.log` | At most 10% of the saved values wrong; the counts in the README. Above 10%, fix the most frequent errors in a patch release |
| An announcement in two or more communities | Posted; each report answered within two days; what came in recorded here two weeks later |

Done: the `ls -l` rule, 0.2.1 (released 2026-11-20, with a rule for pieces of long numbers that the real sessions
turned up) and the issue forms. Open: real use and the announcement.

After that holdback takes bug fixes from reports. Waiting until someone asks: a wider real-session benchmark
(other languages, larger tool output, more kinds of tools); training data for French, German, Italian, Spanish and
Dutch; the teacher's person and email spans on developer text at the next retrain; Homebrew and npm packages.

Reports of wrong replacements are welcome. `holdback review --masked` output is safe to attach to an issue.
