#!/bin/sh
# Start the privacy proxy if Claude Code is routed through it, and warn when it is not.
PORT="${HOLDBACK_PORT:-8787}"
BASE="http://127.0.0.1:$PORT"
say() { printf '{"systemMessage": "%s"}\n' "$1"; exit 0; }
# the install script's folder and cargo's are often missing from the PATH a hook gets
HB=$(command -v holdback) || for d in "$HOME/.local/bin" "${CARGO_HOME:-$HOME/.cargo}/bin"; do [ -x "$d/holdback" ] && HB="$d/holdback" && break; done

case "$ANTHROPIC_BASE_URL" in
  "$BASE"/*) ;;
  *) if [ "$HOLDBACK_HOOKS" = 1 ]; then  # hook mode: `holdback hook` starts the local process and answers
       out=$([ -x "$HB" ] && "$HB" hook 2>/dev/null)
       case "$out" in "{"*) printf '%s\n' "$out"; exit 0 ;; esac
       say "holdback: hook mode is switched on but the holdback command is missing or out of date, so tool calls will be denied. See the holdback README to install it"
     fi
     [ "$CLAUDE_CODE_ENTRYPOINT" = claude-desktop ] && say "holdback: off. The Claude desktop app sets its own API address and ignores ANTHROPIC_BASE_URL from settings, so this session sends private values as is. Set HOLDBACK_HOOKS to 1 in settings.json env for hook mode, or run holdback run claude in a terminal."
     say "holdback: Claude Code is NOT routed through the privacy proxy, so private values are sent as is. Set ANTHROPIC_BASE_URL to $BASE/anthropic in settings.json env." ;;
esac

if curl -fsS "$BASE/_holdback/status" >/dev/null 2>&1; then
  say "holdback: on. Private values stay on this machine. Status: $BASE/_holdback"
fi
[ -x "$HB" ] || say "holdback: the proxy is not running and the holdback command is not installed, so Claude Code cannot reach the model. See the holdback README to install it"

nohup "$HB" serve --port "$PORT" >"${TMPDIR:-/tmp}/holdback.log" 2>&1 &
i=0
while [ $i -lt 110 ]; do
  curl -fsS "$BASE/_holdback/status" >/dev/null 2>&1 && say "holdback: started. Private values stay on this machine. Status: $BASE/_holdback"
  sleep 1; i=$((i + 1))
done
say "holdback: the proxy did not start within 110 s; see ${TMPDIR:-/tmp}/holdback.log"
