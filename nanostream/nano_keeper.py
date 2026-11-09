#!/usr/bin/env python3
"""The always-on half of Nano Stream: it owns the x264 encoder,
the mic and the RTMP publish, and never stops sending.

    python3 nano_keeper.py --url rtmp://127.0.0.1:1935/nano --audio plughw:2,0

render_stream.py connects to its UNIX socket (--url unix:/tmp/nano-keeper.sock)
and sends raw frames. While frames arrive they go straight to the encoder. When
they stop (the renderer restarting, skipping a clip, crashing) the last frame
is held for --hold seconds, then the NANO STREAM card takes over: the ASCII art
in Nano green, a scanline, how long the feed has been away, the die and power.
MediaMTX, the relay and X never see a gap, so X never shows its own offline
screen; the renderer can be restarted at will.

Protocol: the client sends b"NANO" + <uint32 width> + <uint32 height>, then
width*height*3 bytes of BGR per frame. One client at a time; a new one
replaces the old.
"""

import argparse
import os
import signal
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_stream import Encoder, SysStats, ffmpeg_exe, NANO_GREEN  # noqa: E402

GLYPHS = {
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "M": ["███╗   ███╗", "████╗ ████║", "██╔████╔██║", "██║╚██╔╝██║", "██║ ╚═╝ ██║", "╚═╝     ╚═╝"],
    " ": ["  "] * 6,
}
MONO = os.environ.get("NANOSTREAM_MONO_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def ascii_lines(word):
    return ["".join(GLYPHS[c][r] for c in word) for r in range(6)]


def build_card_base(w, h):
    """The static part of the card, rendered once: grid, the ASCII art with a
    green glow, the subtitle."""
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = (20, 10, 6)
    for x in range(0, w, 48):
        cv2.line(bg, (x, 0), (x, h), (34, 20, 12), 1)
    for y in range(0, h, 48):
        cv2.line(bg, (0, y), (w, y), (34, 20, 12), 1)
    lines = ascii_lines("NANO") + [""] + ascii_lines("STREAM")
    size = 24
    font = ImageFont.truetype(MONO, size)
    cw = font.getbbox("█")[2]
    lh = int(size * 1.08)
    y0 = 70
    art = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(art)
    for i, line in enumerate(lines):
        lw = len(line) * cw
        d.text(((w - lw) // 2, y0 + i * lh), line, fill=255, font=font)
    mask = np.array(art, np.float32) / 255.0
    glow = cv2.GaussianBlur(mask, (0, 0), 9) * 0.9
    out = bg.astype(np.float32)
    g = np.array(NANO_GREEN, np.float32)
    out = out * (1 - glow[..., None] * 0.6) + g * glow[..., None] * 0.6
    out = out * (1 - mask[..., None]) + g * mask[..., None]
    base = np.clip(out, 0, 255).astype(np.uint8)
    art_bottom = y0 + len(lines) * lh
    sub = "Hailo-8 dashcam review  .  live from the Jetson Orin Nano"
    (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    cv2.putText(base, sub, ((w - tw) // 2, art_bottom + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (210, 210, 210), 1, cv2.LINE_AA)
    return base, art_bottom


def draw_card(base, art_bottom, t, away_s, status, sysv, hailo_c):
    img = base.copy()
    h, w = img.shape[:2]
    # a slow scanline over the art
    y = int((t * 90) % (art_bottom + 40))
    band = img[max(y - 3, 0):y + 3]
    cv2.addWeighted(band, 0.6, np.full_like(band, NANO_GREEN), 0.4, 0, band)
    # status with a blinking cursor
    cursor = "_" if int(t * 2) % 2 == 0 else " "
    msg = f"> {status} {away_s:4.0f} s {cursor}"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, msg, ((w - tw) // 2, art_bottom + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                NANO_GREEN, 2, cv2.LINE_AA)
    # a sweeping bar under it
    bx0, bx1, by = w // 2 - 260, w // 2 + 260, art_bottom + 124
    cv2.rectangle(img, (bx0, by), (bx1, by + 6), (60, 60, 60), -1)
    s = int((t * 260) % (bx1 - bx0 - 90))
    cv2.rectangle(img, (bx0 + s, by), (bx0 + s + 90, by + 6), NANO_GREEN, -1)

    def f(v, spec, unit):
        return "--" if v is None else f"{v:{spec}}{unit}"

    foot = (f"hailo {f(hailo_c, '.1f', ' C')}   cpu {f(sysv.get('cpu_c'), '.1f', ' C')}   "
            f"{f(sysv.get('power_w'), '.1f', ' W')}   ram {f(sysv.get('ram_used_gb'), '.1f', ' GB')}   "
            + time.strftime("%H:%M:%S"))
    (tw, _), _ = cv2.getTextSize(foot, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, foot, ((w - tw) // 2, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/nano")
    ap.add_argument("--sock", default="/tmp/nano-keeper.sock")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--bitrate", default="3500k")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--tune", default="zerolatency", help="x264 tune; none drops it (B-frames and lookahead back)")
    ap.add_argument("--audio", help="ALSA capture, e.g. plughw:2,0 (default: silent track)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds the last frame is held before the card")
    ap.add_argument("--stall", dest="stall_s", type=float, default=8.0)
    ap.add_argument("--daemon", default="http://127.0.0.1:5911")
    ap.add_argument("--log", default="/tmp/nano-keeper.log")
    a = ap.parse_args()

    def log(msg):
        with open(a.log, "a") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}\n")

    W, H = a.width, a.height
    base, art_bottom = build_card_base(W, H)
    sysstats = SysStats()
    shared = {"frame": None, "t": 0.0, "client": False, "frames_in": 0, "hailo_c": None}
    lock = threading.Lock()

    def poll_hailo():
        import json
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen(a.daemon + "/health", timeout=5) as r:
                    shared["hailo_c"] = json.loads(r.read().decode()).get("chip_temp_c")
            except Exception:
                pass
            time.sleep(5)

    def recv_exact(conn, buf):
        view, got = memoryview(buf), 0
        while got < len(buf):
            n = conn.recv_into(view[got:])
            if n == 0:
                raise ConnectionError("client closed")
            got += n

    def serve():
        try:
            os.unlink(a.sock)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(a.sock)
        srv.listen(2)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=client, args=(conn,), daemon=True).start()

    current = {"id": 0}

    def client(conn):
        current["id"] += 1
        me = current["id"]
        try:
            hdr = bytearray(12)
            recv_exact(conn, hdr)
            if bytes(hdr[:4]) != b"NANO":
                raise ConnectionError("bad header")
            fw, fh = struct.unpack("<II", bytes(hdr[4:]))
            log(f"client {me} connected, {fw}x{fh}")
            shared["client"] = True
            buf = bytearray(fw * fh * 3)
            while current["id"] == me:
                recv_exact(conn, buf)
                fr = np.frombuffer(bytes(buf), np.uint8).reshape(fh, fw, 3)
                if (fw, fh) != (W, H):
                    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                with lock:
                    shared["frame"], shared["t"] = fr, time.time()
                    shared["frames_in"] += 1
        except Exception as e:
            log(f"client {me} gone: {e}")
        finally:
            if current["id"] == me:
                shared["client"] = False
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    threading.Thread(target=poll_hailo, daemon=True).start()

    ff = ffmpeg_exe()
    audio = {"src": a.audio}

    def new_enc():
        return Encoder(ff, W, H, a, log, audio["src"])

    enc = new_enc()

    def on_term(signum, _frame):
        # keeper-stop sends SIGTERM: take the encoder down with us and free the socket
        enc.kill()
        try:
            os.unlink(a.sock)
        except OSError:
            pass
        log(f"signal {signum}: keeper exiting")
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)
    log(f"keeper up: {W}x{H} @ {a.fps} fps -> {a.url}, socket {a.sock}, audio {a.audio}")
    away_since = time.time()
    mode = "card"
    cards = 0
    while True:
        now = time.time()
        with lock:
            fr, t = shared["frame"], shared["t"]
        if fr is not None and now - t < a.hold:
            out = fr
            if mode != "live":
                log(f"live again after {now - away_since:.1f} s, {cards} card frames")
                mode, cards = "live", 0
        else:
            if mode == "live":
                away_since = t
                mode = "card"
                log("feed stopped; holding, then the card")
            status = "renderer restarting" if not shared["client"] else "waiting for frames"
            out = draw_card(base, art_bottom, now, now - away_since, status, sysstats.v, shared["hailo_c"])
            cards += 1
        if not enc.write(out):
            dead = enc
            dead.kill()
            if dead.audio and time.time() - dead.started < 5:
                log(f"encoder with audio {dead.audio} died within 5 s; silent track from here")
                audio["src"] = None
            log("encoder gone; starting a new one")
            enc = new_enc()


if __name__ == "__main__":
    main()
