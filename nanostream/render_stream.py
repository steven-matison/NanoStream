#!/usr/bin/env python3
"""Play dashcam video at play speed with the Hailo detections drawn on every
frame, and push it as one RTMP stream (Nano Stream).

    python3 render_stream.py --source /media/$USER/<label> --url rtmps://a.live.x.com:443/x/<key>
    python3 render_stream.py --source http://<mac>:8787/ --url rtmp://127.0.0.1:1935/nano

One process: cv2 reads the files in order (a file, a directory, or the Mac's
http listing through dashcam_walker's cache), every frame goes to the daemon's
/detect (~22 ms on the chip), every --plates-every Nth frame to /plates, the
boxes and plate text are drawn and held between reads, and the BGR frames are
piped into ffmpeg (libx264 + silent AAC, X rejects video-only) that publishes to
--url. The encoder is the static ffmpeg in the venv (imageio-ffmpeg), because
the Orin Nano has no hardware encoder and NVIDIA's ffmpeg build has no x264.

The stream never goes silent. The next clips (--lookahead) are fetched in the
background while the current one plays; if the next is not there yet when the clip ends, the last
frame is held on screen at the stream's frame rate until it is. A download
that fails is retried, a clip that is gone from the listing (404) is skipped,
and the listing is read again on every pass so the server can change its set.
Frames go to ffmpeg through a bounded queue; when ffmpeg stops taking them
(the RTMP side is dead) the encoder is killed and started again. No
-rw_timeout on the output: with it, this ffmpeg's RTMP writer stalls after
~40 s while x264 keeps spinning, and MediaMTX drops the publisher.

Measured on an Orin Nano: 720p30 x264 veryfast runs 3.6x play speed alone
and holds 30 fps beside the detect loop at 23 fps. 1080p30 is 1.8x.

--audio plughw:<card>,0 puts a live mic on the stream in place of the silent
track (find the card with `arecord -l`; the plug device, not hw:, since this
ffmpeg cannot open some USB mics' hw: format). If that encoder dies inside
5 s the mic is gone and the next encoder goes back to the silent track.

The band across the top (TopBand) reads NANO STREAM, Jetson Orin Nano and
Hailo-8, then the Hailo-8's counters: the boxes on screen now, big, and the
objects seen this run, small (HailoTally follows boxes pass to pass; a track
counts after 3 passes), with the clock and the clip on the right.

The MiNiFi bar (minifi_bar.py) runs along the bottom, under the G and
RPM meters: every clip the renderer has on disk is hard-linked into the C++
agent's ListFile folder (--minifi-clips), and the bar draws the agent's own
per-frame JSONs (--minifi-meta) for the clip and second on screen. --no-minifi
turns it off.

Writes: --log, the http cache under --cache (a played clip's copy is removed, old
ones pruned on start), the links in --minifi-clips, and the agent's JPEGs under
--minifi-out that were written during this run (pruned after --out-keep-min).
"""

import argparse
import concurrent.futures
import glob
import json
import os
import queue
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashcam_walker import list_videos, mask_hud, HUD_MASKS  # noqa: E402
try:
    # the MiNiFi C++ bar, minifi_bar.py beside this file
    from minifi_bar import MiniFiFeed, MiniFiBar  # noqa: E402
except ImportError:
    MiniFiFeed = MiniFiBar = None

VEHICLES = {"car", "truck", "bus", "motorcycle"}
ORANGE, GREEN, WHITE = (0, 140, 255), (60, 220, 0), (255, 255, 255)
NANO_GREEN = (90, 220, 118)
BAND_H = 40
FETCH_BLOCK = 4 * 1024 * 1024     # one synced write of a download; a read waits behind one at most


# the top band's counter slots, fixed so the band never reflows: (label, Hailo labels, colour).
# The same eight as the MiNiFi bar along the bottom, plus the plates.
TOP_SLOTS = [("PERSON", {"person"}, WHITE), ("CAR", {"car"}, ORANGE), ("TRUCK", {"truck"}, ORANGE),
             ("BUS", {"bus"}, ORANGE), ("BIKE", {"bicycle", "motorcycle"}, ORANGE),
             ("LIGHT", {"traffic_light"}, (40, 190, 250)), ("SIGN", {"stop_sign"}, (60, 60, 235)),
             ("OTHER", None, (175, 175, 175)), ("PLATE", (), NANO_GREEN)]
PLATE_SLOT = len(TOP_SLOTS) - 1


def _top_slot(label):
    for i, (_, names, _) in enumerate(TOP_SLOTS[:PLATE_SLOT]):
        if names is not None and label in names:
            return i
    return PLATE_SLOT - 1


def _iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class HailoTally:
    """What the Hailo-8 sees, for the top band: the boxes on screen now, and the
    objects seen since the stream started. The detector has no ids, so a light
    tracker follows each box from one pass to the next (same slot, IoU >= 0.2 or
    the centre within half the box's diagonal); a track counts once it has been
    seen on 3 passes (~0.25 s at 12 passes/s), and is dropped after 0.8 s unseen.
    The total is an estimate of objects, not of boxes. Plates: on screen is the
    last plate pass's boxes, the total is the distinct plate texts read."""

    def __init__(self, confirm=3, ttl=0.8):
        self.confirm, self.ttl = confirm, ttl
        self.lock = threading.Lock()
        self.tracks = []                       # [slot, box, hits, last_seen, counted]
        self.now = [0] * len(TOP_SLOTS)
        self.total = [0] * len(TOP_SLOTS)
        self.texts = set()
        self.bumped = {}                       # slot -> time its total last rose

    def update(self, dets, plates=None, now=None):
        now = time.time() if now is None else now
        with self.lock:
            on = [0] * len(TOP_SLOTS)
            free = [t for t in self.tracks if now - t[3] <= self.ttl]
            for d in sorted(dets, key=lambda d: -d.get("score", 0)):
                k, box = _top_slot(d["label"]), d["box"]
                on[k] += 1
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                best, bs = None, 0.0
                for t in free:
                    if t[0] != k:
                        continue
                    tb = t[1]
                    sc = _iou(box, tb)
                    if sc < 0.2:
                        diag = ((tb[2] - tb[0]) ** 2 + (tb[3] - tb[1]) ** 2) ** 0.5
                        dist = ((cx - (tb[0] + tb[2]) / 2) ** 2 + (cy - (tb[1] + tb[3]) / 2) ** 2) ** 0.5
                        sc = 0.1 if dist < 0.5 * diag else 0.0
                    if sc > bs:
                        best, bs = t, sc
                if best is None:
                    self.tracks.append([k, box, 1, now, False])
                    continue
                free.remove(best)
                best[1], best[3] = box, now
                best[2] += 1
                if not best[4] and best[2] >= self.confirm:
                    best[4] = True
                    self.total[k] += 1
                    self.bumped[k] = now
            self.tracks = [t for t in self.tracks if now - t[3] <= self.ttl]
            if plates is None:
                on[PLATE_SLOT] = self.now[PLATE_SLOT]
            else:
                on[PLATE_SLOT] = len(plates)
                for p in plates:
                    if p.get("text") and p["text"] not in self.texts:
                        self.texts.add(p["text"])
                        self.total[PLATE_SLOT] += 1
                        self.bumped[PLATE_SLOT] = now
            self.now = on

    def view(self):
        with self.lock:
            return list(self.now), list(self.total), dict(self.bumped)


class TopBand:
    """The band across the top of every frame, in the MiNiFi bar's style:
    NANO STREAM and the hardware on the left, the Hailo-8's counters in the
    middle (big: on screen now; small: seen this run), the clock and the clip
    on the right. Drawn at 1280 wide into a cache, redrawn only when what it
    shows changes, and scaled to the frame."""

    H = BAND_H
    W = 1280
    L, M, R = 0, 300, 1080                     # the three sections

    def __init__(self, tally):
        self.tally = tally
        self.key = self.cache = None
        base = np.empty((self.H, self.W, 3), np.uint8)
        top, bot = (16, 30, 16), (9, 16, 9)    # BGR, a dark green-black, darker at the bottom
        for y in range(self.H):
            f = y / (self.H - 1)
            base[y] = [int(top[c] * (1 - f) + bot[c] * f) for c in range(3)]
        self.base = base

    def draw(self, img, clip, msg=None):
        now = time.time()
        on, tot, bumped = self.tally.view()
        fl = tuple(k for k, t in bumped.items() if now - t < 0.6)
        key = (clip, msg, tuple(on), tuple(tot), fl, int(now * 10) if fl else int(now))
        if key != self.key or self.cache is None:
            self.cache = self._render(clip, msg, on, tot, bumped, now)
            self.key = key
        w = img.shape[1]
        band = self.cache if w == self.W else cv2.resize(
            self.cache, (w, int(round(self.H * w / self.W))), interpolation=cv2.INTER_AREA)
        img[0:band.shape[0]] = band
        cv2.line(img, (0, band.shape[0]), (w, band.shape[0]), NANO_GREEN, 1)

    def _render(self, clip, msg, on, tot, bumped, now):
        F, FD, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
        G, DIM, SEP = (175, 175, 175), (90, 98, 90), (40, 64, 40)
        GDIM = (60, 140, 76)
        img = self.base.copy()
        W, H, M, R = self.W, self.H, self.M, self.R
        for x in (M, R):
            cv2.line(img, (x, 7), (x, H - 7), SEP, 1)

        # ---- left: the title and the hardware ----
        cv2.putText(img, "NANO STREAM", (14, 28), FD, 0.78, NANO_GREEN, 1, AA)
        cv2.putText(img, "Jetson Orin Nano", (186, 17), F, 0.38, WHITE, 1, AA)
        cv2.putText(img, "Hailo-8", (186, 33), F, 0.38, NANO_GREEN, 1, AA)
        cv2.putText(img, "26 TOPS", (240, 33), F, 0.32, G, 1, AA)

        # ---- right: the clock and the clip ----
        clock = time.strftime("%H:%M:%S", time.localtime(now))
        (cw, _), _ = cv2.getTextSize(clock, FD, 0.58, 1)
        cv2.putText(img, clock, (W - 14 - cw, 20), FD, 0.58, WHITE, 1, AA)
        c = clip or ""
        while c and cv2.getTextSize(c, F, 0.34, 1)[0][0] > W - R - 28:
            c = c[:-1]
        (nw, _), _ = cv2.getTextSize(c, F, 0.34, 1)
        cv2.putText(img, c, (W - 14 - nw, 34), F, 0.34, G, 1, AA)

        # ---- middle: the counters, or the loading card's message ----
        if msg:
            (mw, _), _ = cv2.getTextSize(msg, F, 0.55, 1)
            cv2.putText(img, msg, (M + (R - M - mw) // 2, 27), F, 0.55, NANO_GREEN, 1, AA)
            return img
        cv2.putText(img, "NOW", (M + 10, 23), F, 0.32, WHITE, 1, AA)
        cv2.putText(img, "TOTAL", (M + 10, 34), F, 0.3, GDIM, 1, AA)
        x0 = M + 56
        cw_ = (R - 6 - x0) // len(TOP_SLOTS)
        for k, (lab, _, col) in enumerate(TOP_SLOTS):
            cx = x0 + k * cw_
            f = now - bumped.get(k, 0)
            if f < 0.6:                        # a new object flashes its cell
                a_ = 1 - f / 0.6
                bg = [int(v) for v in img[20, cx + 2]]
                cv2.rectangle(img, (cx + 1, 4), (cx + cw_ - 3, H - 4),
                              tuple(int(bg[i] + (col[i] - bg[i]) * 0.35 * a_) for i in range(3)), -1)
            n, t = on[k], tot[k]
            live = n > 0
            if live:
                cv2.line(img, (cx + 4, 3), (cx + cw_ - 6, 3), col, 2)
            txt = str(n)
            (tw, _), _ = cv2.getTextSize(txt, FD, 0.66, 1)
            cv2.putText(img, txt, (cx + 30 - tw, 29), FD, 0.66, col if live else DIM, 1, AA)
            cv2.putText(img, lab, (cx + 35, 17), F, 0.32, G if live else DIM, 1, AA)
            tt = f"{t:,}"
            cv2.putText(img, tt, (cx + 35, 33), F, 0.4, NANO_GREEN if t else DIM, 1, AA)
        return img


def ffmpeg_exe():
    for cand in (os.environ.get("FFMPEG"), shutil.which("ffmpeg-static")):
        if cand and os.path.exists(cand):
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def post(url, jpg, timeout=10):
    req = urllib.request.Request(url, data=jpg, headers={"Content-Type": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class SysStats:
    """Jetson health, sampled once a second on its own thread from sysfs and
    /proc: thermal zones by type, the INA3221 rails, CPU load and clock, RAM,
    the fan. The Orin's GPU is not sampled: nothing in the stream runs on it.
    Nothing here touches the Hailo; its die temperature comes back in every
    daemon reply."""

    def __init__(self, keep=120):
        self.keep = keep
        self.v = {}
        self.hist = {"hailo": [], "power": []}
        self._cpu = None
        self._hw = {}
        for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            try:
                self._hw[open(h + "/name").read().strip()] = h
            except OSError:
                pass
        self._zones = {}
        for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
            try:
                self._zones[open(z + "/type").read().strip()] = z + "/temp"
            except OSError:
                pass
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _read(path, div=1.0):
        try:
            return int(open(path).read().split()[0]) / div
        except (OSError, ValueError, IndexError):
            return None

    def _run(self):
        while True:
            v = {}
            for key, zone in (("cpu_c", "cpu-thermal"), ("tj_c", "tj-thermal")):
                v[key] = self._read(self._zones.get(zone, "/nonexistent"), 1000.0)
            try:
                f = open("/proc/stat").readline().split()[1:]
                f = [int(x) for x in f]
                idle, total = f[3] + f[4], sum(f)
                if self._cpu:
                    di, dt = idle - self._cpu[0], total - self._cpu[1]
                    v["cpu_pct"] = 100.0 * (1 - di / dt) if dt else None
                self._cpu = (idle, total)
            except (OSError, ValueError):
                pass
            v["cpu_ghz"] = self._read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1e6)
            try:
                mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo")}
                v["ram_used_gb"] = (mi["MemTotal"] - mi["MemAvailable"]) / 1048576
                v["ram_total_gb"] = mi["MemTotal"] / 1048576
            except (OSError, KeyError, ValueError):
                pass
            ina = self._hw.get("ina3221")
            if ina:
                mv, ma = self._read(ina + "/in1_input"), self._read(ina + "/curr1_input")
                v["power_w"] = mv * ma / 1e6 if mv and ma else None
                mv, ma = self._read(ina + "/in2_input"), self._read(ina + "/curr2_input")
                v["cpugpu_w"] = mv * ma / 1e6 if mv and ma else None
            fan = self._hw.get("pwmfan")
            if fan:
                pwm = self._read(fan + "/pwm1")
                v["fan_pct"] = pwm * 100 / 255 if pwm is not None else None
            tach = self._hw.get("pwm_tach")
            if tach:
                v["fan_rpm"] = self._read(tach + "/rpm")
            self.v = v
            if v.get("power_w") is not None:
                self._push("power", v["power_w"])
            time.sleep(1.0)

    def _push(self, key, val):
        h = self.hist[key]
        h.append(val)
        del h[:-self.keep]


def _heat(c, warm, hot):
    if c is None:
        return (160, 160, 160)
    return (60, 60, 235) if c >= hot else (40, 190, 250) if c >= warm else NANO_GREEN


def _spark(img, vals, x, y, w, h, col, lo=None, hi=None):
    if len(vals) < 2:
        return
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = max(hi - lo, 1e-6)
    step = w / (len(vals) - 1)
    pts = np.array([[int(x + i * step), int(y + h - (min(max(v, lo), hi) - lo) / span * h)]
                    for i, v in enumerate(vals)], np.int32)
    cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)


def metrics_panel(img, sysv, hist, rates):
    """The health panel, top left, over the PDR's speed readout (which it
    hides on purpose): Hailo die, CPU, power, RAM, fan, stream and inference
    rates, six rows spaced to fill the height the readout needs covered. Text
    colours turn amber and red with the heat."""
    x0, y0, w, h = 0, BAND_H + 1, 330, 214
    img[y0:y0 + h, x0:x0 + w] = (22, 12, 8)    # solid: the PDR speed underneath stays hidden
    cv2.line(img, (x0 + w, y0), (x0 + w, y0 + h), NANO_GREEN, 1)
    cv2.line(img, (x0, y0 + h), (x0 + w, y0 + h), NANO_GREEN, 1)
    F, S, G = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 175, 175)

    def fmt(val, spec, unit=""):
        return "--" if val is None else f"{val:{spec}}{unit}"

    def row(i, label, value, col=WHITE, extra="", extra_col=G):
        y = y0 + 28 + i * 28
        cv2.putText(img, label, (x0 + 12, y), F, 0.45, G, 1, cv2.LINE_AA)
        cv2.putText(img, value, (x0 + 92, y), F, S, col, 1, cv2.LINE_AA)
        if extra:
            cv2.putText(img, extra, (x0 + 176, y), F, 0.45, extra_col, 1, cv2.LINE_AA)
        return y

    hc = rates.get("hailo_c")
    y = row(0, "HAILO-8", fmt(hc, ".1f", " C"), _heat(hc, 80, 88))
    _spark(img, hist["hailo"], x0 + 176, y - 12, 140, 14, _heat(hc, 80, 88), 50, 95)
    c = sysv.get("cpu_c")
    row(1, "CPU", fmt(c, ".1f", " C"), _heat(c, 75, 90),
        f"{fmt(sysv.get('cpu_pct'), '.0f', '%')}  {fmt(sysv.get('cpu_ghz'), '.2f', ' GHz')}")
    pw = sysv.get("power_w")
    y = row(2, "POWER", fmt(pw, ".1f", " W"), WHITE)
    _spark(img, hist["power"], x0 + 176, y - 12, 140, 14, NANO_GREEN, 5, 25)
    row(3, "RAM", f"{fmt(sysv.get('ram_used_gb'), '.1f')} / {fmt(sysv.get('ram_total_gb'), '.1f')} GB", WHITE,
        f"fan {fmt(sysv.get('fan_pct'), '.0f', '%')} {fmt(sysv.get('fan_rpm'), '.0f', ' rpm')}")
    row(4, "STREAM", fmt(rates.get("fps"), ".1f", " fps"), WHITE, f"x264 {rates.get('mode', '')}")
    row(5, "INFER", fmt(rates.get("ips"), ".0f", " /s"), WHITE, f"{fmt(rates.get('ms'), '.0f', ' ms')} last")
    cv2.putText(img, f"25W mode   Tj {fmt(sysv.get('tj_c'), '.1f', ' C')}   up {rates.get('up', '')}",
                (x0 + 12, y0 + h - 10), F, 0.42, G, 1, cv2.LINE_AA)


def chip(img, text, x, y, col):
    """A filled label tab above a box, dark text on the box colour."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    y = max(y, BAND_H + th + 8)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y), col, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def draw(frame, dets, plates):
    for d in dets:
        x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
        col = ORANGE if d["label"] in VEHICLES else WHITE
        cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
        chip(frame, f'{d["label"]} {d["score"]:.2f}', x0, y0 - 2, col)
    for p in plates:
        x0, y0, x1, y1 = [int(round(v)) for v in p["box"]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), NANO_GREEN, 2)
        if p.get("text"):
            chip(frame, p["text"], x0, y0 - 2, NANO_GREEN)
    return frame


def loading_card(w, h, base, title, name, prog, lines, t):
    """The picture while there is nothing to play: a dark Nano card, or the
    last frame dimmed under it, with the fetch progress and a spinning arc."""
    if base is not None:
        img = (base.astype(np.uint16) * 3 // 10).astype(np.uint8)
    else:
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (32, 16, 10)
        for x in range(0, w, 64):
            cv2.line(img, (x, 0), (x, h), (48, 28, 18), 1)
        for y in range(0, h, 64):
            cv2.line(img, (0, y), (w, y), (48, 28, 18), 1)
    green = (90, 220, 118)
    cv2.putText(img, "NANO STREAM", (48, 96), cv2.FONT_HERSHEY_DUPLEX, 2.2, green, 3, cv2.LINE_AA)
    cv2.putText(img, "Hailo-8 dashcam review  .  live from the Jetson", (52, 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, title, (48, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, name, (48, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
    # the bar
    x0, y0, x1, y1 = 48, h // 2 + 24, w - 48, h // 2 + 52
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    total, done = prog.get("total") or 0, prog.get("bytes", 0)
    if total:
        f = min(done / total, 1.0)
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + 2 + int((x1 - x0 - 4) * f), y1 - 2), green, -1)
        if prog.get("unit") == "frames":
            txt = f"{f * 100:4.0f} %   {done} / {total} frames scored by MiNiFi"
        else:
            rate = done * 8 / max(time.time() - prog.get("t0", t), 0.1) / 1e6
            txt = f"{f * 100:4.0f} %   {done / 1e6:.0f} / {total / 1e6:.0f} MB   {rate:.0f} Mbit/s"
    else:
        sweep = int(t * 300) % (x1 - x0 - 120)
        cv2.rectangle(img, (x0 + 2 + sweep, y0 + 2), (x0 + 2 + sweep + 116, y1 - 2), green, -1)
        txt = "waiting for the server"
    cv2.putText(img, txt, (x0, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
    # the arc
    cx, cy = w - 110, 100
    cv2.ellipse(img, (cx, cy), (36, 36), 0, int(t * 240) % 360, int(t * 240) % 360 + 270, green, 4, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (48, h - 96 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 1, cv2.LINE_AA)
    return img


class Encoder:
    """One ffmpeg process fed through a bounded queue by its own thread, so a
    stalled RTMP output shows up as a full queue here instead of a blocked
    write; the caller then kills it and starts a new one."""

    def __init__(self, ff, w, h, a, log, audio=None):
        self.log = log
        self.fps = a.fps
        self.audio = audio
        self.started = time.time()
        if audio:
            # a live ALSA capture (plughw:<card>,0: the plug layer converts the
            # mic's own format); mono in, stereo AAC out, small gaps resampled
            audio_in = ["-thread_queue_size", "2048", "-f", "alsa", "-ac", "1", "-ar", "48000", "-i", audio]
            audio_out = ["-c:a", "aac", "-b:a", "96k", "-ac", "2", "-af", "aresample=async=1000"]
        else:
            audio_in = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
            audio_out = ["-c:a", "aac", "-b:a", "96k"]
        cmd = ([ff, "-hide_banner", "-loglevel", "warning", "-stats",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-"]
               + audio_in
               + ["-c:v", "libx264", "-preset", a.preset]
               + ([] if a.tune == "none" else ["-tune", a.tune])
               + ["-pix_fmt", "yuv420p",
                  "-b:v", a.bitrate, "-maxrate", a.bitrate, "-bufsize", "7000k", "-g", str(int(a.fps * 2))]
               + audio_out + ["-shortest", "-f", "flv", a.url])
        log("encoder: " + " ".join(cmd[:-1]) + " <url>")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=open(a.log, "ab"))
        self.q = queue.Queue(maxsize=int(a.fps * 2))
        self.dead = False
        self.n = 0
        self.t0 = time.perf_counter()
        self.stall_s = a.stall_s
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while not self.dead:
            buf = self.q.get()
            if buf is None:
                break
            try:
                self.proc.stdin.write(buf)
            except (BrokenPipeError, OSError):
                self.dead = True
                break

    def write(self, frame):
        """Returns False when the encoder is gone or has not taken a frame for
        --stall seconds; the caller restarts it."""
        if self.dead or self.proc.poll() is not None:
            return False
        try:
            self.q.put(frame.tobytes(), timeout=self.stall_s)
        except queue.Full:
            self.log(f"encoder took no frame for {self.stall_s} s; killing it")
            self.kill()
            return False
        self.n += 1
        # play-speed pace: sleep until this frame's slot
        due = self.t0 + self.n / self.fps
        lag = time.perf_counter() - due
        if lag < 0:
            time.sleep(-lag)
        return True

    def kill(self):
        self.dead = True
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def close(self):
        self.dead = True
        try:
            self.q.put(None, timeout=2)
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.kill()


class KeeperSink:
    """The renderer's output when a keeper owns the encoder (--url
    unix:<socket>): frames go over the socket, paced here at --fps. A lost
    keeper is retried once a second and frames are dropped meanwhile; write()
    never asks for an encoder restart."""

    def __init__(self, path, w, h, fps, log):
        self.path, self.w, self.h, self.fps, self.log = path, w, h, fps, log
        self.sock = None
        self.next_try = 0.0
        self.n = 0
        self.t0 = time.perf_counter()
        self.audio = None
        self.started = time.time()
        self._connect()

    def _connect(self):
        self.next_try = time.time() + 1.0
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect(self.path)
            s.sendall(b"NANO" + struct.pack("<II", self.w, self.h))
            self.sock = s
            self.log(f"connected to the keeper at {self.path}")
        except OSError as e:
            self.sock = None
            self.log(f"keeper not reachable at {self.path}: {e}")

    def write(self, frame):
        if self.sock is None and time.time() >= self.next_try:
            self._connect()
        if self.sock is not None:
            try:
                self.sock.sendall(frame.tobytes())
            except OSError as e:
                self.log(f"keeper connection lost: {e}")
                self.sock.close()
                self.sock = None
        self.n += 1
        lag = time.perf_counter() - (self.t0 + self.n / self.fps)
        if lag < 0:
            time.sleep(-lag)
        return True

    def kill(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    close = kill


def prune_cache(cache, keep_h, log):
    """Leftovers from earlier runs: every half-written .part, and clips older
    than keep_h hours. A clip from the last few hours stays, so a restart
    reuses it instead of downloading it again."""
    if not keep_h or not os.path.isdir(cache):
        return
    cut, gone, freed = time.time() - keep_h * 3600, 0, 0
    for e in os.scandir(cache):
        try:
            st = e.stat()
            if e.is_file() and (e.name.endswith(".part") or st.st_mtime < cut):
                os.remove(e.path)
                gone, freed = gone + 1, freed + st.st_size
        except OSError:
            pass
    if gone:
        log(f"cache: removed {gone} stale file(s), {freed / 1e9:.1f} GB, from {cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--url", required=True,
                    help="rtmp(s):// target, or unix:/tmp/nano-keeper.sock to hand frames to nano_keeper.py")
    ap.add_argument("--daemon", default="http://127.0.0.1:5911")
    ap.add_argument("--width", type=int, default=1280, help="stream width; 1280 = 720p")
    ap.add_argument("--fps", type=float, default=30.0, help="stream frame rate")
    ap.add_argument("--bitrate", default="3500k")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--tune", default="zerolatency", help="x264 tune; none drops it (B-frames and lookahead back)")
    ap.add_argument("--plates-every", type=int, default=None,
                    help="run the plate pass every Nth frame (default 10, or 30 while the MiNiFi bar is on: "
                         "the plate OCR and the agent share the CPU)")
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--loop", action="store_true", help="start over at the end of the list")
    ap.add_argument("--cache", default=os.environ.get("NANOSTREAM_CACHE", "/tmp/nano-stream-cache"))
    ap.add_argument("--cache-keep-h", type=float, default=24,
                    help="on start, drop cached clips older than this many hours, and every .part (0 keeps all)")
    ap.add_argument("--log", default="/tmp/nano-stream.log")
    ap.add_argument("--records", help="also append one JSON record per plate pass here")
    ap.add_argument("--mask-hud", default="pdr", choices=sorted(HUD_MASKS),
                    help="black out a burned-in telemetry overlay before inference; the picture keeps it")
    ap.add_argument("--stall", dest="stall_s", type=float, default=8.0,
                    help="seconds the encoder may refuse frames before it is restarted")
    ap.add_argument("--fetch-retries", type=int, default=4)
    ap.add_argument("--lookahead", type=int, default=2,
                    help="clips downloaded (and handed to the agent) ahead of the one playing, one download at a time; "
                         "2 lets the agent use a long clip's play time to get ahead of a short one")
    ap.add_argument("--audio", help="ALSA capture for live commentary, e.g. plughw:2,0 (default: silent track)")
    ap.add_argument("--intro", type=float, default=3.0, help="seconds of the Nano card before the first clip")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N videos after sorting (first pass only)")
    ap.add_argument("--start-at", help="start at the first video whose basename sorts >= this (e.g. 250123)")
    ap.add_argument("--hours", help="play only clips recorded in this hour range, from the HHMMSS in the "
                                    "name (YYMMDD_HHMMSS_...): 9-15 is daylight")
    ap.add_argument("--minifi-clips", default="/tmp/minifi-detect/clips",
                    help="the MiNiFi C++ agent's ListFile folder; each clip is hard-linked in here")
    ap.add_argument("--minifi-meta", default="/tmp/minifi-detect/meta", help="where the agent writes its per-frame JSONs")
    ap.add_argument("--minifi-out", default="/tmp/minifi-detect/out",
                    help="the agent's annotated JPEGs; the ones written during this run are pruned")
    ap.add_argument("--out-keep-min", type=float, default=20, help="minutes an annotated JPEG is kept; 0 keeps all")
    ap.add_argument("--no-minifi", action="store_true", help="no MiNiFi bar and no clips handed to the agent")
    ap.add_argument("--minifi-cores", default="4,5",
                    help="the cores the agent is held to while a clip plays (it gets every core while the picture "
                         "waits on its scores); none leaves the agent where it is")
    ap.add_argument("--minifi-nice", type=int, default=15, help="the lowest priority the agent's threads run at")
    ap.add_argument("--minifi-fps", type=float, default=0.25, help="the agent flow's sampling rate, frames per second")
    ap.add_argument("--minifi-sync", type=float, default=0.9,
                    help="before a clip plays, hold on the card until the agent has scored this share of its "
                         "frames, so the bar is in step from the clip's first second (0 = never wait)")
    ap.add_argument("--minifi-wait", type=float, default=900,
                    help="the longest that hold lasts, in seconds; then the clip plays anyway")
    a = ap.parse_args()
    if a.plates_every is None:
        a.plates_every = 10 if a.no_minifi else 30

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        try:
            with open(a.log, "a") as f:
                f.write(line + "\n")
        except OSError:
            print(line, flush=True)

    ff = ffmpeg_exe()
    mfeed = mbar = None
    if not a.no_minifi:
        if MiniFiFeed is None:
            log("minifi_bar.py not found beside render_stream.py; no MiNiFi bar")
        else:
            mfeed = MiniFiFeed(a.minifi_meta, a.minifi_clips, log, keep_links=max(a.lookahead, 1) + 1,
                               out_dir=a.minifi_out, out_keep_min=a.out_keep_min)
            mbar = MiniFiBar(mfeed)
            log(f"minifi bar: clips -> {a.minifi_clips}, JSONs <- {a.minifi_meta}")
    prune_cache(a.cache, a.cache_keep_h, log)
    all_cores = sorted(os.sched_getaffinity(0))
    play_cores = None
    if mfeed is not None and a.minifi_cores != "none":
        play_cores = [int(c) for c in a.minifi_cores.split(",") if c.strip() != ""]
        if not set(play_cores) <= set(all_cores) or len(play_cores) >= len(all_cores):
            log(f"minifi: --minifi-cores {a.minifi_cores} does not leave the stream a core here; not placing the agent")
            play_cores = None
        else:
            mfeed.place(play_cores, a.minifi_nice)
            log(f"minifi: agent held to cores {a.minifi_cores} at nice {a.minifi_nice} while clips play")
    playing = {"local": None}

    def on_term(signum, _frame):
        # stop and next end the renderer with SIGTERM: hand back the link of the
        # clip on screen so it never outlives its cache copy, then go
        if mfeed is not None and playing["local"]:
            mfeed.release(playing["local"])
        log(f"signal {signum}: exiting")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    try:
        with urllib.request.urlopen(a.daemon + "/health", timeout=5) as r:
            h = json.loads(r.read().decode())
    except Exception as e:
        log(f"daemon not healthy at {a.daemon}: {e}")
        sys.exit(3)
    log(f"daemon {h.get('models')} die {h.get('chip_temp_c')} C; ffmpeg {ff}")

    # Inference runs beside the frame loop, never inside it: the encoder must get
    # a frame every 1/fps s, and a plate pass with the full OCR can take 1.5 s.
    # The worker always takes the newest frame and drops the rest; the loop draws
    # the latest result it has. Boxes trail the picture by a frame or two.
    state = {"dets": [], "plates": [], "infer_ms": 0.0, "passes": 0, "errors": 0,
             "latest": None, "stop": False, "temp": h.get("chip_temp_c", "?")}
    lock = threading.Lock()
    tally = HailoTally()
    band = TopBand(tally)

    def worker():
        k = 0
        while not state["stop"]:
            with lock:
                item = state["latest"]
                state["latest"] = None
            if item is None:
                time.sleep(0.005)
                continue
            jpg, path, n_frame = item
            t0 = time.perf_counter()
            try:
                if a.plates_every and k % a.plates_every == 0:
                    res = post(f"{a.daemon}/plates?min_score={a.min_score}", jpg, 60)
                    new_plates = [p for v in res.get("vehicles", []) for p in v.get("plates", [])]
                    if a.records:
                        with open(a.records, "a") as f:
                            f.write(json.dumps({"device": os.environ.get("DASHCAM_DEVICE", "nano"),
                                                "ts": time.time(), "source": path,
                                                "frame": n_frame, "counts": res.get("counts"),
                                                "vehicles": res.get("vehicles"),
                                                "chip_temp_c": res.get("chip_temp_c")}) + "\n")
                else:
                    res = post(f"{a.daemon}/detect?min_score={a.min_score}", jpg, 10)
                    new_plates = None
                with lock:
                    state["dets"] = res.get("detections", [])
                    state["temp"] = res.get("chip_temp_c", state["temp"])
                    if new_plates is not None:
                        state["plates"] = new_plates
                    state["infer_ms"] = (time.perf_counter() - t0) * 1000
                    state["passes"] += 1
                tally.update(res.get("detections", []), new_plates)
            except Exception as e:
                state["errors"] += 1
                if state["errors"] % 100 == 1:
                    log(f"daemon error: {e}")
            k += 1

    threading.Thread(target=worker, daemon=True).start()

    # --- the fetch side: --lookahead clips ahead, one download at a time, in the background ---
    fetches = {}

    def fetch_with_progress(url, prog):
        """dashcam_walker.fetch_to_cache with a byte counter for the card; the
        file lands as .part and is renamed when complete, so a partial copy is
        never mistaken for a cached one."""
        os.makedirs(a.cache, exist_ok=True)
        local = os.path.join(a.cache, urllib.parse.unquote(url.rsplit("/", 1)[-1]))
        prog.update({"bytes": 0, "total": None, "t0": time.time()})
        wait_for_space(head_size(url), url, local)
        prog["t0"] = time.time()
        with urllib.request.urlopen(url, timeout=60) as r:
            prog["total"] = int(r.headers.get("Content-Length") or 0) or None
            fetch_body(r, local, prog)
        os.replace(local + ".part", local)
        secs = max(time.time() - prog["t0"], 0.001)
        log(f"fetched {prog['bytes'] / 1e6:.0f} MB in {secs:.0f} s ({prog['bytes'] * 8 / secs / 1e6:.0f} Mbit/s): {url}")
        return local

    def head_size(url):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10) as r:
                return int(r.headers.get("Content-Length") or 0)
        except Exception:
            return 0

    def wait_for_space(size, url, local):
        """A clip fetched ahead waits for room: the cache keeps a GB free, and a
        played clip's copy is deleted as soon as it ends. It waits only while a
        clip this run is playing or has fetched ahead can still free that room."""
        warned = False
        while shutil.disk_usage(a.cache).free < size + 1e9:
            owned = {playing["local"]} | {h["local"] for h in fetches.values()}
            others = [p for p in owned if p and p != local and os.path.exists(p)]
            if not others:
                log(f"cache: under {(size + 1e9) / 1e9:.1f} GB free and nothing to wait on; fetching {url} anyway")
                return
            if not warned:
                log(f"cache: waiting for {(size + 1e9) / 1e9:.1f} GB free before {url}")
                warned = True
            time.sleep(5)

    def fetch_body(r, local, prog):
        """Write the clip in 4 MB blocks, each one on the eMMC before the next is
        read. Left to the page cache, a 1.4 GB download goes to disk in one burst
        when it ends. Synced every 32 MB and written flat out, it held the eMMC
        busy for the whole download, one 3 s write after another, and the clip
        on screen waited behind them: the keeper showed the card for 3.6 and
        4.3 s. So a block is small enough that a read waits a fraction of a
        second for it, and while a clip is playing the download rests after each
        block for twice the time the block took to write, which leaves the eMMC
        idle two thirds of the time. A clip fetched ahead has minutes to arrive.
        With nothing playing the picture is waiting on this clip: no rest."""
        with open(local + ".part", "wb") as f:
            while True:
                chunk = r.read(FETCH_BLOCK)
                if not chunk:
                    break
                t0 = time.perf_counter()
                f.write(chunk)
                f.flush()
                os.fdatasync(f.fileno())
                prog["bytes"] += len(chunk)
                if playing["local"]:
                    time.sleep(min(2 * (time.perf_counter() - t0), 2.0))

    def cached(url):
        """A complete copy already in the cache (same size as the server's), left
        by an earlier run; a restart then starts in seconds, not minutes."""
        local = os.path.join(a.cache, urllib.parse.unquote(url.rsplit("/", 1)[-1]))
        if not os.path.exists(local):
            return None
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                size = int(r.headers.get("Content-Length") or -1)
        except Exception:
            return None
        if size > 0 and os.path.getsize(local) == size:
            log(f"cached copy used ({size / 1e6:.0f} MB): {url}")
            return local
        return None

    def fetch_with_retry(url, prog):
        hit = cached(url)
        if hit:
            return hit
        for attempt in range(1, a.fetch_retries + 1):
            try:
                return fetch_with_progress(url, prog)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    log(f"gone from the server (404), skipping: {url}")
                    return None
                log(f"fetch {attempt}/{a.fetch_retries} failed HTTP {e.code}: {url}")
            except Exception as e:
                log(f"fetch {attempt}/{a.fetch_retries} failed ({e.__class__.__name__}: {e}): {url}")
            time.sleep(min(5 * attempt, 20))
        return None

    def start_fetch(url):
        if not url.startswith("http") or url in fetches:
            return
        h = {"done": threading.Event(), "local": None, "prog": {}}
        fetches[url] = h

        def run():
            h["local"] = fetch_with_retry(url, h["prog"])
            if mfeed is not None and h["local"]:
                mfeed.feed(h["local"])      # the agent starts on it while the clip before plays
            h["done"].set()
        fetch_pool.submit(run)

    # one download at a time, in the order asked: two on one link halve each other
    fetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fetch")

    # --- the encoder side ---
    enc = {"e": None, "w": a.width, "h": None, "last": None, "audio": a.audio}
    stats = {"n": 0, "clips": 0}
    sysstats = SysStats()
    rates = {"fps": None, "ips": None, "ms": None, "hailo_c": None, "up": "", "mode": "",
             "_t": time.time(), "_n": 0, "_p": 0, "t0": time.time()}

    def tick_rates():
        now = time.time()
        dt = now - rates["_t"]
        if dt >= 1.0:
            rates["fps"] = (stats["n"] - rates["_n"]) / dt
            rates["ips"] = (state["passes"] - rates["_p"]) / dt
            rates["_t"], rates["_n"], rates["_p"] = now, stats["n"], state["passes"]
            t = state["temp"]
            if isinstance(t, (int, float)):
                rates["hailo_c"] = t
                sysstats._push("hailo", t)
            up = int(now - rates["t0"])
            rates["up"] = f"{up // 3600}:{up % 3600 // 60:02d}:{up % 60:02d}"
            rates["mode"] = f"{enc['w']}x{enc['h'] or ''} {a.bitrate}"
        rates["ms"] = state["infer_ms"]

    def new_encoder():
        if a.url.startswith("unix:"):
            return KeeperSink(a.url[5:], enc["w"], enc["h"], a.fps, log)
        return Encoder(ff, enc["w"], enc["h"], a, log, enc["audio"])

    def emit(frame, hud):
        """Draw the HUD on a copy and push it; restart the encoder when it dies."""
        if enc["h"] != frame.shape[0] or enc["e"] is None:
            if enc["e"] is not None:
                enc["e"].close()
            enc["h"] = frame.shape[0]
            enc["e"] = new_encoder()
        out = frame.copy()
        band.draw(out, hud[0], hud[1])
        if not enc["e"].write(out):
            dead = enc["e"]
            dead.kill()
            if dead.audio and time.time() - dead.started < 5:
                # the mic is gone (unplugged, or taken by the desktop): carry on silent
                log(f"encoder with audio {dead.audio} died within 5 s; falling back to the silent track")
                enc["audio"] = None
            log("encoder gone; killing it and starting a new one")
            enc["e"] = new_encoder()
            enc["e"].write(out)
        stats["n"] += 1

    def hold(reason, h, name=""):
        """Keep the stream alive on the loading card until h['done'] is set: the
        last frame dimmed under it when there is one, the bare card when not."""
        held = 0
        t0 = time.time()
        w = enc["w"]
        hh = enc["last"].shape[0] if enc["last"] is not None else w * 9 // 16
        while not h["done"].wait(timeout=0):
            t = time.time()
            lines = [f"{stats['n']} frames streamed   {state['passes']} inference passes   "
                     f"{stats['clips']} clips played   die {state['temp']} C",
                     time.strftime("%H:%M:%S") + f"   {t - t0:.0f} s"]
            p = h.get("prog", {})
            if p.get("unit") == "frames" and mfeed is not None and mfeed.latest_clip:
                # a restarted agent first finishes the frames it had queued, from other clips
                lines.insert(0, f"MiNiFi is working on {mfeed.latest_clip}, "
                                f"{mfeed.scored(mfeed.latest_clip)} frames scored")
            card = loading_card(w, hh, enc["last"], reason, name, h.get("prog", {}), lines, t)
            emit(card, (name, reason))
            held += 1
        if held:
            log(f"showed the loading card {held} frames ({t0 and time.time() - t0:.0f} s) while {reason} {name}")

    def resize(frame):
        if frame.shape[1] != enc["w"]:
            scale = enc["w"] / frame.shape[1]
            frame = cv2.resize(frame, (enc["w"], int(frame.shape[0] * scale) // 2 * 2),
                               interpolation=cv2.INTER_AREA)
        return frame

    def in_hours(path):
        if not a.hours:
            return True
        lo, hi = (int(x) for x in a.hours.split("-"))
        m = re.match(r"\d{6}_(\d{2})\d{4}_", os.path.basename(urllib.parse.unquote(path)))
        return bool(m) and lo <= int(m.group(1)) < hi

    def wait_for_agent(stem, duration, name):
        """Hold on the card until the agent has scored a.minifi_sync of the clip,
        so the bar matches the picture from its first second. The renderer
        keeps clips ahead, so after the first clip this wait is normally zero.
        No wait when the agent is down; never longer than a.minifi_wait."""
        need = max(1, int(duration * a.minifi_fps * a.minifi_sync))
        if mfeed.scored(stem) >= need:
            return
        if mfeed.agent["pid"] is None:
            log(f"minifi: agent down, playing {name} without waiting for its scores")
            return
        h = {"done": threading.Event(), "prog": {"unit": "frames", "total": need, "bytes": mfeed.scored(stem)}}
        if play_cores:
            mfeed.place(all_cores, a.minifi_nice)      # nothing to play yet: the agent may use every core
        t0 = time.time()

        def watch():
            while time.time() - t0 < a.minifi_wait:
                h["prog"]["bytes"] = mfeed.scored(stem)
                if h["prog"]["bytes"] >= need or mfeed.agent["pid"] is None:
                    break
                time.sleep(0.5)
            h["done"].set()
        threading.Thread(target=watch, daemon=True).start()
        hold("MiNiFi is scoring the next clip", h, name)
        if play_cores:
            mfeed.place(play_cores, a.minifi_nice)
        got = mfeed.scored(stem)
        log(f"minifi: {got} of {need} frames scored for {name} after {time.time() - t0:.0f} s"
            + ("" if got >= need else "; playing anyway"))

    # --- the play loop ---
    done_names = set()
    first_pass = True
    if a.intro > 0:
        ev = threading.Event()
        threading.Timer(a.intro, ev.set).start()
        hold("Starting the stream", {"done": ev}, "Hailo-8 vision on the Jetson Orin Nano")
    while True:
        videos = list_videos(a.source)
        if a.hours:
            listed = len(videos)
            videos = [v for v in videos if in_hours(v)]
            if first_pass:
                log(f"hours {a.hours}: {len(videos)} of {listed} clips")
        if videos and first_pass:
            found = len(videos)
            if a.start_at:
                videos = [v for v in videos if os.path.basename(urllib.parse.unquote(v)) >= a.start_at]
                log(f"start-at {a.start_at!r}: {len(videos)} of {found} remain")
            if a.skip:
                videos = videos[a.skip:]
                log(f"skip {a.skip}: {len(videos)} of {found} remain")
            first_pass = False
            if not videos and not a.loop:
                log("nothing left to play after start-at/skip")
                break
        if not videos:
            log(f"no video under {a.source}; retrying in 10 s")
            ev = threading.Event()
            threading.Timer(10, ev.set).start()
            hold("Waiting for the video server", {"done": ev}, a.source)
            continue
        pending = [v for v in videos if os.path.basename(v) not in done_names]
        if not pending:
            if not a.loop:
                break
            done_names.clear()
            continue
        log(f"{len(pending)} video(s) to play of {len(videos)} listed")
        for i, path in enumerate(pending):
            name = os.path.basename(path)
            done_names.add(name)
            start_fetch(path)
            if path.startswith("http"):
                h = fetches.pop(path)
                if not h["done"].is_set():
                    hold("Loading the next clip", h, name)
                local = h["local"]
                if local is None:
                    continue
            else:
                local = path
            # the clips ahead download only now, so they never share the link with
            # the clip we were waiting on, and they queue behind each other
            ahead = pending[i + 1:i + 1 + max(a.lookahead, 1)]
            for nxt in ahead:
                start_fetch(nxt)
            if mfeed is not None:
                # on the fetch thread: a clip on a mount is copied, not linked, and a copy
                # must not hold the picture (http clips are handed over as they land)
                fetch_pool.submit(mfeed.feed, local)    # a no-op for a clip already handed over
                for nxt in ahead:
                    if not nxt.startswith("http"):
                        fetch_pool.submit(mfeed.feed, nxt)
            cap = cv2.VideoCapture(local)
            if not cap.isOpened():
                log(f"cannot open {path}")
                continue
            playing["local"] = local
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / src_fps
            stem = os.path.splitext(os.path.basename(local))[0]     # the agent's video.clip
            if mfeed is not None and a.minifi_sync > 0:
                wait_for_agent(stem, duration, name)
            stats["clips"] += 1
            log(f"playing {name} @ {src_fps:.2f} fps")
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = resize(frame)
                # hand the newest frame to the worker only when it is idle; a
                # JPEG at quality 80 costs ~3 ms here
                if state["latest"] is None:
                    ok, jpg = cv2.imencode(".jpg", mask_hud(frame, HUD_MASKS[a.mask_hud]),
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    with lock:
                        state["latest"] = (jpg.tobytes(), path, n)
                with lock:
                    dets, plates, infer_ms = state["dets"], state["plates"], state["infer_ms"]
                hud = (name, None)          # the band shows the Hailo counters
                frame = draw(frame, dets, plates)
                tick_rates()
                metrics_panel(frame, sysstats.v, sysstats.hist, rates)
                if mbar is not None:
                    mbar.draw(frame, stem, n / src_fps, duration)
                enc["last"] = frame
                emit(frame, hud)
                n += 1
                if n % 900 == 0:
                    log(f"{name} {n} frames, {len(dets)} objects, {state['passes']} inference passes, "
                        f"last {infer_ms:.0f} ms, encoder frame {enc['e'].n}")
            cap.release()
            playing["local"] = None
            if mfeed is not None:
                mfeed.release(local)
            if local != path:
                try:
                    os.remove(local)
                except OSError:
                    pass
        if not a.loop:
            break
    state["stop"] = True
    if enc["e"] is not None:
        enc["e"].close()
    log(f"done: {stats['n']} frames, {state['passes']} inference passes, {state['errors']} daemon errors")


if __name__ == "__main__":
    main()
