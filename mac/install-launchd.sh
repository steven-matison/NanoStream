#!/usr/bin/env bash
# Keep the clip server running on the Mac: a launchd agent that starts at login, restarts the
# server if it dies, waits for the drive, and holds off idle sleep (caffeinate -i) while it serves.
# Closing the lid still sleeps the Mac, even on AC power: keep it open.
#
#   bash mac/install-launchd.sh "/Volumes/<drive>/<folder>" [PORT]   install, or replace, and start
#   bash mac/install-launchd.sh --status                             is it loaded, and does :PORT answer
#   bash mac/install-launchd.sh --uninstall                          stop and remove it
#
# The log is ~/Library/Logs/nanostream-clips.log. No sudo: it is a per-user LaunchAgent.
# launchd runs a copy of serve-clips.sh from ~/Library/Application Support/NanoStream, not the
# clone: macOS denies a launchd bash any script under ~/Documents, ~/Desktop or ~/Downloads
# (exit 126, "Operation not permitted"), and a clone is often there.
set -euo pipefail
LABEL=com.nanostream.clips
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGF="$HOME/Library/Logs/nanostream-clips.log"
HERE="$(cd "$(dirname "$0")" && pwd)"
APPDIR="$HOME/Library/Application Support/NanoStream"
DOMAIN="gui/$(id -u)"

[ "$(uname -s)" = Darwin ] || { echo "this is the macOS installer; on Linux run mac/serve-clips.sh under systemd or tmux" >&2; exit 2; }

case "${1:-}" in
  --uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST" "$APPDIR/serve-clips.sh"
    rmdir "$APPDIR" 2>/dev/null || true
    echo "removed $LABEL"
    exit 0
    ;;
  --status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E '^\s*(state|pid|last exit code) ' || echo "$LABEL is not loaded"
    port=8787   # PlistBuddy prints "File Doesn't Exist" on stdout, so read the plist only when it is there
    [ -f "$PLIST" ] && port=$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:5' "$PLIST" 2>/dev/null || echo 8787)
    curl -s -o /dev/null -w "http://127.0.0.1:$port/ -> %{http_code}\n" "http://127.0.0.1:$port/" || true
    exit 0
    ;;
  ""|-h|--help)
    sed -n '2,9p' "$0"; exit 64
    ;;
esac

DIR=$1
PORT=${2:-8787}
xml() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOGF")" "$APPDIR"
install -m 755 "$HERE/serve-clips.sh" "$APPDIR/serve-clips.sh"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-i</string>
    <string>/bin/bash</string>
    <string>$(xml "$APPDIR/serve-clips.sh")</string>
    <string>$(xml "$DIR")</string>
    <string>$PORT</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$(xml "$LOGF")</string>
  <key>StandardErrorPath</key><string>$(xml "$LOGF")</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "installed $LABEL: $DIR on :$PORT, log $LOGF"
echo "first start: give /usr/bin/caffeinate Full Disk Access (System Settings, Privacy & Security), or it serves 0 clips; macOS shows no prompt for it."
sleep 3
curl -s -o /dev/null -w "http://127.0.0.1:$PORT/ -> %{http_code}\n" "http://127.0.0.1:$PORT/" || echo "not answering yet; tail $LOGF"
