#!/bin/sh
# Installs the holdback binary for this machine.
#   curl -fsSL https://raw.githubusercontent.com/Daveyuwang/holdback/main/install.sh | sh
#   sh install.sh --uninstall
# HOLDBACK_VERSION picks a release tag (default: the latest release).
# HOLDBACK_INSTALL_DIR picks the folder (default: ~/.local/bin).
# HOLDBACK_ARCHIVE installs from an archive already on disk instead of downloading one.
set -eu
REPO="Daveyuwang/holdback"
DIR="${HOLDBACK_INSTALL_DIR:-$HOME/.local/bin}"

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$DIR/holdback"
  echo "removed $DIR/holdback"
  echo "left in place: the model in ~/.cache/holdback, saved values in ~/.local/state/holdback, your lists in ~/.config/holdback"
  exit 0
fi

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target=aarch64-apple-darwin ;;
  Darwin-x86_64) target=x86_64-apple-darwin ;;
  Linux-x86_64) target=x86_64-unknown-linux-musl ;;
  Linux-aarch64 | Linux-arm64) target=aarch64-unknown-linux-musl ;;
  *) echo "holdback has no prebuilt binary for $(uname -s) $(uname -m); build it from source with cargo" >&2; exit 1 ;;
esac
name="holdback-$target.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ -n "${HOLDBACK_ARCHIVE:-}" ]; then
  cp "$HOLDBACK_ARCHIVE" "$tmp/$name"
  cp "$HOLDBACK_ARCHIVE.sha256" "$tmp/$name.sha256"
else
  base="https://github.com/$REPO/releases/latest/download"
  [ -z "${HOLDBACK_VERSION:-}" ] || base="https://github.com/$REPO/releases/download/$HOLDBACK_VERSION"
  curl -fsSL "$base/$name" -o "$tmp/$name"
  curl -fsSL "$base/$name.sha256" -o "$tmp/$name.sha256"
fi

want=$(cut -d' ' -f1 "$tmp/$name.sha256")
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$tmp/$name" | cut -d' ' -f1); else got=$(shasum -a 256 "$tmp/$name" | cut -d' ' -f1); fi
[ "$want" = "$got" ] || { echo "checksum mismatch for $name; nothing was installed" >&2; exit 1; }

tar -xzf "$tmp/$name" -C "$tmp"
mkdir -p "$DIR"
install -m 755 "$tmp/holdback" "$DIR/holdback"
echo "installed $("$DIR/holdback" --version) to $DIR/holdback"
case ":$PATH:" in
  *":$DIR:"*) ;;
  *) echo "add it to PATH: put   export PATH=\"$DIR:\$PATH\"   in your shell profile" ;;
esac
echo "next: holdback setup claude"
