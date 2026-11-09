#!/usr/bin/env bash
# The video source: serve a folder of dashcam clips over plain HTTP from a MacBook (or any box with python3).
#   bash mac/serve-clips.sh <CLIP_DIR> [PORT]
# PORT defaults to 8787. Binds every interface so the Jetson can reach it; Ctrl-C stops it.
# When CLIP_DIR is on a drive that is not mounted yet (a launchd start at login), it waits for it.
# python3's http.server serves no byte ranges, so the Jetson downloads each clip whole before
# it opens it.
set -euo pipefail
DIR=${1:?usage: serve-clips.sh <CLIP_DIR> [PORT]}
PORT=${2:-8787}
waited=0
until [ -d "$DIR" ]; do
  [ "$waited" -eq 0 ] && echo "waiting for $DIR to appear (is the drive plugged in?)"
  sleep 10; waited=$((waited + 10))
done
cd "$DIR"
shopt -s nullglob nocaseglob
clips=(./*.mp4 ./*.mov ./*.mkv ./*.m4v ./*.ts ./*.avi)
[ "${#clips[@]}" -gt 0 ] || echo "warning: no clips directly in $DIR (subfolders are listed too)" >&2
ip=$( (ipconfig getifaddr en0 || ipconfig getifaddr en1 || hostname -I | awk '{print $1}') 2>/dev/null || echo "<this-host>")
echo "serving ${#clips[@]} clip(s) from $DIR at http://$ip:$PORT/"
exec python3 -m http.server "$PORT" --bind 0.0.0.0
