# Changelog

## 0.2.2

- `holdback replace` now replaces a listed phrase in full even when the detector recognizes only part of it.
  The reply restores the original phrase.
- Restarting holdback preserves the placeholders for values on the replace list, so older turns in a resumed
  hook session still restore correctly.

## 0.2.1

- Numbers that stand alone, such as counts and process ids, are kept, and so are shell syntax, command-line
  options and the header of a CSV file. A placeholder no longer takes a line break with it, so a CSV row is no
  longer merged into the header above it.
- A placeholder that a model writes without its brackets, such as `PRIVATE_PERSON_1`, is restored too.
- The owner of a file in `ls -l` output is replaced, as the user name in a home folder is. The distilled model
  found it in some listings only. System accounts such as `root` stay.
- When the detector takes a piece of a longer number for an account number, the whole number is replaced. A
  piece used to break a constant in source code in two, and was then replaced wherever it appeared alone.

## 0.2.0

- holdback's default detector is now a distilled model of 28M parameters, trained on public data to label what
  OpenAI's Privacy Filter labels. It is a 115 MB download instead of 0.9 GB, handles a request 6 to 7 times faster,
  and misses fewer values in English and Chinese: 1.0% and 5.8% on the two English public sets it never trained on,
  against 9.7% and 12.7%, and 8.7% of the names in Chinese news text, against 67.3%. It misses more in French,
  German, Italian, Spanish and Dutch; `HOLDBACK_DETECTOR=privacy-filter` brings back the Privacy Filter.
- The first start downloads the model from Hugging Face at a pinned revision and checks each file's SHA-256.
  `HF_ENDPOINT` points the download at a mirror, and `HOLDBACK_STUDENT_DIR` names a folder with your own copy. A
  Privacy Filter download from 0.1.x is checked and kept.
- If you set `HOLDBACK_STUDENT_DIR` for 0.1.2, unset it: that folder holds an older version of the model, which
  leaves the names and emails in a `Cargo.toml` `authors` list in place. `holdback doctor` says when the folder
  holds another version than the pinned one.
- `holdback doctor` reads the `env` blocks of Claude Code's settings files, as a session started in the folder does.
- The model's training data, training and evaluation code are in `distill/`.

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
