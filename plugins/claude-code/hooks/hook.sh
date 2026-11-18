#!/bin/sh
# Hook mode, for a session the proxy cannot route (the Claude desktop app): replace private values in what a
# tool returns, put them back into what a tool receives and into the text shown to the user.
# Switched on with HOLDBACK_HOOKS=1 in the settings env. A session routed through the proxy needs none of it.
[ "$HOLDBACK_HOOKS" = 1 ] || exit 0
case "$ANTHROPIC_BASE_URL" in "http://127.0.0.1:${HOLDBACK_PORT:-8787}"/*) exit 0 ;; esac
# the install script's folder and cargo's are often missing from the PATH a hook gets
HB=$(command -v holdback) || for d in "$HOME/.local/bin" "${CARGO_HOME:-$HOME/.cargo}/bin"; do [ -x "$d/holdback" ] && HB="$d/holdback" && break; done
out=$([ -x "$HB" ] && "$HB" hook 2>/dev/null)
case "$out" in "{"*) printf '%s\n' "$out"; exit 0 ;; esac
# No answer: holdback is not installed, or is older than this plugin. No tool runs, because its output could
# not be checked.
[ "$1" = PreToolUse ] && printf '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "holdback hook mode is switched on but the holdback command is missing or out of date, so tool output cannot be checked. Install or update holdback, or switch hook mode off."}}\n'
exit 0
