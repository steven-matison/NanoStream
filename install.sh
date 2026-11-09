#!/usr/bin/env bash
# Install Nano Stream on a Jetson, as your own user. Never sudo; safe to run again.
#
#   ./install.sh                 venv packages, HEFs, MediaMTX, the two user units (enabled and started)
#   ./install.sh --no-start      everything, but leave the units stopped
#   ./install.sh --check         say what is there and what is missing; change nothing
#
# Environment (all optional):
#   NANOSTREAM_VENV    the venv to use or create     the one the last install recorded, else ~/.nanostream/venv
#                      point it at a venv that already holds HailoRT's Python wheel to reuse it
#   HAILO_MODELS       where the HEFs go              ~/hailo/models
#   MEDIAMTX_DIR       where MediaMTX goes            ~/mediamtx
#   MEDIAMTX_VERSION   the MediaMTX release           v1.20.1
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
STATE="$HOME/.nanostream"
# an earlier install recorded the venv it used in $STATE/env; a re-run or --check follows it unless told otherwise
if [ -z "${NANOSTREAM_VENV:-}" ] && [ -f "$STATE/env" ]; then
  # shellcheck disable=SC1091  # the file install.sh itself writes: one `: "${NANOSTREAM_PY:=<python>}"` line
  recorded=$(unset NANOSTREAM_PY; . "$STATE/env"; printf '%s' "${NANOSTREAM_PY%/bin/python}")
  [ -x "$recorded/bin/python" ] && NANOSTREAM_VENV=$recorded
fi
VENV="${NANOSTREAM_VENV:-$STATE/venv}"
MODELS="${HAILO_MODELS:-$HOME/hailo/models}"
MTX="${MEDIAMTX_DIR:-$HOME/mediamtx}"
MTX_VER="${MEDIAMTX_VERSION:-v1.20.1}"
ZOO="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8"
HEFS=(yolov8s tiny_yolov4_license_plates lprnet)
UNITS="$HOME/.config/systemd/user"

MODE=install
case "${1:-}" in
  --check) MODE=check ;;
  --no-start) MODE=nostart ;;
  "") ;;
  *) sed -n '2,15p' "$0"; exit 64 ;;
esac

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
miss() { printf '  \033[33mmiss\033[0m  %s\n' "$*"; }
step() { printf '\n%s\n' "$*"; }

arch=$(uname -m)
[ "$arch" = aarch64 ] || echo "note: this is $arch; Nano Stream is built for a Jetson (aarch64) with a Hailo-8"

# ---- the venv ---------------------------------------------------------------------------
step "venv: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  if ! python3 -c 'import ensurepip, venv' 2>/dev/null; then
    miss "python3 cannot make a venv here: install the python3-venv package first (or point NANOSTREAM_VENV at one)"
    [ "$MODE" = check ] || exit 1
  fi
  if [ "$MODE" = check ]; then miss "no venv"; else
    mkdir -p "$(dirname "$VENV")"
    # system site packages: a HailoRT built and installed for the system python stays visible
    python3 -m venv --system-site-packages "$VENV"
    ok "created"
  fi
fi
if [ -x "$VENV/bin/python" ]; then
  if [ "$MODE" != check ]; then
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$HERE/requirements.txt"
  fi
  if "$VENV/bin/python" -c 'import cv2, numpy, PIL, imageio_ffmpeg, rapidocr_onnxruntime' 2>/dev/null; then
    ok "app packages"
  else
    miss "app packages (run without --check)"
  fi
  if "$VENV/bin/python" -c 'import hailo_platform' 2>/dev/null; then
    ok "hailo_platform $("$VENV/bin/python" -c 'import hailo_platform; print(hailo_platform.__version__)')"
  else
    miss "hailo_platform: install HailoRT's Python wheel into this venv (README, 'The Hailo side')"
  fi
  ff=$("$VENV/bin/python" -c 'import imageio_ffmpeg as m; print(m.get_ffmpeg_exe())' 2>/dev/null || true)
  # read the list whole: grep -q would close the pipe early, and pipefail then reads ffmpeg's SIGPIPE as a miss
  encoders=$( [ -n "$ff" ] && "$ff" -hide_banner -encoders 2>/dev/null || true)
  if [[ $encoders == *libx264* ]]; then ok "static ffmpeg with libx264"; else miss "static ffmpeg with libx264"; fi
fi

# ---- the HEFs ---------------------------------------------------------------------------
step "HEFs: $MODELS (Hailo Model Zoo v2.20.0, Hailo-8)"
for m in "${HEFS[@]}"; do
  if [ -s "$MODELS/$m.hef" ]; then ok "$m.hef"; continue; fi
  if [ "$MODE" = check ]; then miss "$m.hef"; continue; fi
  mkdir -p "$MODELS"
  if curl -fsSL -o "$MODELS/$m.hef.part" "$ZOO/$m.hef"; then
    mv "$MODELS/$m.hef.part" "$MODELS/$m.hef"; ok "$m.hef downloaded"
  else
    rm -f "$MODELS/$m.hef.part"; echo "  download failed: $ZOO/$m.hef" >&2; exit 1
  fi
done
if [ -e /dev/hailo0 ]; then ok "/dev/hailo0"; else miss "/dev/hailo0: the hailo_pci driver is not loaded (README, 'The Hailo side')"; fi

# ---- MediaMTX ---------------------------------------------------------------------------
step "MediaMTX: $MTX"
if [ -x "$MTX/mediamtx" ]; then
  ok "$("$MTX/mediamtx" --version 2>/dev/null || echo present)"
elif [ "$MODE" = check ]; then
  miss "mediamtx"
else
  case "$arch" in aarch64) a=arm64 ;; x86_64) a=amd64 ;; *) a=$arch ;; esac
  mkdir -p "$MTX"
  curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/$MTX_VER/mediamtx_${MTX_VER}_linux_$a.tar.gz" \
    | tar -xz -C "$MTX" mediamtx mediamtx.yml
  ok "mediamtx $MTX_VER"
fi

# ---- the user units ---------------------------------------------------------------------
step "user units: $UNITS"
if [ "$MODE" = check ]; then
  for u in hailo-infer mediamtx; do
    if [ -f "$UNITS/$u.service" ]; then ok "$u: $(systemctl --user is-active "$u" 2>/dev/null || true)"; else miss "$u.service"; fi
  done
else
  mkdir -p "$UNITS" "$STATE"
  for u in hailo-infer mediamtx; do
    sed -e "s#@PY@#$VENV/bin/python#g" -e "s#@APP@#$HERE/nanostream#g" -e "s#@STATE@#$STATE#g" \
        -e "s#@MODELS@#$MODELS#g" -e "s#@MTX@#$MTX#g" "$HERE/systemd/$u.service" > "$UNITS/$u.service"
  done
  systemctl --user daemon-reload
  if [ "$MODE" = install ]; then
    systemctl --user enable --now hailo-infer mediamtx
    systemctl --user restart hailo-infer      # a re-install picks up a changed unit or daemon
    for _ in $(seq 1 30); do curl -sf -m 2 127.0.0.1:5911/health >/dev/null && break; sleep 1; done
    if curl -sf -m 2 127.0.0.1:5911/health >/dev/null; then ok "hailo-infer answering on :5911"
    else miss "hailo-infer not answering yet: journalctl --user -u hailo-infer"; fi
    ok "mediamtx: $(systemctl --user is-active mediamtx)"
  else
    ok "written, not started (systemctl --user enable --now hailo-infer mediamtx)"
  fi
  if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]; then
    echo "  note: the units stop when you log out. To keep them up: loginctl enable-linger $USER"
  fi
fi

# ---- the command ------------------------------------------------------------------------
step "command"
if [ "$MODE" != check ]; then
  mkdir -p "$STATE"
  # nano-stream reads this; a NANOSTREAM_PY set in the environment still wins
  # shellcheck disable=SC2016  # written literally, expanded when nano-stream sources it
  printf ': "${NANOSTREAM_PY:=%s}"\n' "$VENV/bin/python" > "$STATE/env"
  ok "$STATE/env: nano-stream runs $VENV/bin/python"
fi
if [ "$MODE" != check ]; then
  mkdir -p "$HOME/.local/bin"
  ln -sf "$HERE/bin/nano-stream" "$HOME/.local/bin/nano-stream"
fi
if [ -L "$HOME/.local/bin/nano-stream" ]; then ok "nano-stream in ~/.local/bin"; else miss "nano-stream in ~/.local/bin"; fi
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "  note: ~/.local/bin is not on PATH in this shell (a new login adds it): export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac
echo
echo "next: nano-stream keeper && nano-stream start http://<mac-ip>:8787/   (README, 'Run it')"
