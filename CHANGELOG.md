# Changelog

## 0.1.2

- The experimental distilled detector is on Hugging Face:
  [limbo23/holdback-student](https://huggingface.co/limbo23/holdback-student). Download it and point
  `HOLDBACK_STUDENT_DIR` at its folder; the model card has the commands and the measurements.
- The distilled detector's memory on long input is back down. 0.1.1's new window cuts had raised its peak for an
  8,192-token input from 110 to 165 MB; its working buffers are now sized once, and its output is unchanged.

## 0.1.1

- A name or email in long input without line breaks, such as a one-line JSON response, could be cut by the edge
  of a detection window and replaced only in part. Windows now end at a line break, else at a separator such as
  a comma or bracket, else at a space. In 200 records on one line, 23 of 1,928 names and emails were replaced
  only in part before; none are now.
- Fewer false replacements in developer output: hashes in lockfiles and wheel RECORD files (`sha256:<hex>`,
  `sha512-<base64>`), packages with a version after @ (`openssl@3`), paths under system folders such as `/usr`
  and `/opt`, epoch times under keys that name a time, and the random tails of names such as Kubernetes pod
  names.
- A user mention that looks like a constant, as in Slack's `<@U02LIZ_MARCHETTI>`, is replaced as a person.
- `holdback doctor` and the status JSON say which detector runs.
- An experimental distilled detector with its own runtime, used when `HOLDBACK_STUDENT_DIR` points at its
  files. The model is not published yet, so the default detector is still the Privacy Filter.

## 0.1.0

First release.

- Proxy mode for Claude Code, Codex and clients of the Anthropic Messages, OpenAI Chat Completions and OpenAI
  Responses APIs: private values become numbered placeholders in every request and return in the reply.
- Hook mode for Claude Code sessions that cannot be routed through a proxy, such as the Claude desktop app:
  tool output is redacted, tool input and displayed replies get the real values back.
- A native runtime for the OpenAI Privacy Filter model: one binary, about 50 MB of memory when idle.
- Rules that keep what an agent needs (hosts, ports, paths, dates, versions, IDs) and replace only the secret
  part of a URL, connection string or config line.
- `holdback review`, `holdback keep` and `holdback replace` to inspect and correct what is replaced.
- `holdback setup claude` and `holdback doctor`.
- A benchmark of detection and task success, with Codex's patterns, Presidio and rehydra on the same data.
