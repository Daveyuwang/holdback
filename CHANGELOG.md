# Changelog

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
