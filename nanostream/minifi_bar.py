#!/usr/bin/env python3
"""The MiNiFi bar along the bottom of the Nano Stream picture.

The MiNiFi C++ agent scores the same clips the stream
plays: the renderer hands each downloaded clip to the agent's ListFile folder
(--minifi-clips, a hard link, so the bytes are not copied), the agent runs
ExtractVideoFrames -> ClassifyImage (Places365) -> DetectObject (yolov8s on
Tract, the CPU) and writes one JSON per sampled frame into --minifi-meta. This
module reads those JSONs and draws, under the G and RPM meters:

  left    MiNiFi Object Detection (CPU): images scored, s per image, the
          agent's CPU and RSS from /proc, a pulse on every new JSON
  middle  object counters: the sampled frame at or just behind the playhead,
          big; the clip's running total of sightings up to the playhead, small
  right   the scene: Places365 top-1 with its confidence bar, top-2 and top-3
  top     a timeline of the clip: a tick for every frame MiNiFi has scored and
          the playhead, so the picture shows MiNiFi running ahead of it

Matching is by clip (video.clip, the file stem) and video.t_s against the
playhead, so the bar shows the agent's own result for the second on screen.
The renderer keeps two clips ahead (--lookahead) and hands each one to the
agent as its download finishes, then holds a clip on the card until the agent
has scored most of it (--minifi-sync); with two ahead that hold is usually
already over, and the bar's state chip says where the agent is.

Disk (the Nano's eMMC is the only disk): a hard link shares the cache file's
blocks, and the agent reads a clip only while ExtractVideoFrames walks it
(about 31 s for a 9.6 min clip, while the clip before it is still on screen).
A link is removed when its clip finishes playing (release), right before the
renderer deletes its cache copy, so a link never holds blocks of its own. The agent's
annotated JPEGs (--minifi-out, about 20 MB a clip) written during this run are
pruned past --out-keep-min; files from before the run are never touched.

Nothing here writes outside --minifi-clips and --minifi-out, and there it only
removes what it made itself or what the agent wrote during this run.
"""

import bisect
import json
import os
import shutil
import threading
import time

import cv2
import numpy as np

# BGR
BG = (24, 13, 9)
BG2 = (34, 20, 13)
CYAN = (234, 190, 30)        # MiNiFi's accent, apart from the Hailo green above
CYAN_DIM = (110, 80, 20)
NANO_GREEN = (90, 220, 118)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
GREY = (175, 175, 175)
DIM = (95, 90, 85)
AMBER = (40, 190, 250)
RED = (60, 60, 235)

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

BAR_H = 60          # at 720p; the meters end at about y 645
DESIGN_W = 1280

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

# the counter slots, fixed so the bar never reflows: (label, COCO ids, colour)
SLOTS = [("PERSON", {0}, WHITE), ("CAR", {2}, ORANGE), ("TRUCK", {7}, ORANGE), ("BUS", {5}, ORANGE),
         ("BIKE", {1, 3}, ORANGE), ("LIGHT", {9}, AMBER), ("SIGN", {11}, RED), ("OTHER", None, GREY)]


def slot_of(cid):
    for i, (_, ids, _) in enumerate(SLOTS):
        if ids is not None and cid in ids:
            return i
    return len(SLOTS) - 1


def _jl(v):
    """scene and detections arrive as JSON strings inside the JSON."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return []
    return v or []


class Clip:
    def __init__(self):
        self.ts = []        # sorted t_s
        self.recs = []      # same order
        self.cum = None     # prefix sums of the slot counts, rebuilt lazily

    def add(self, t, rec):
        # a JSON rewritten under the same name (the agent re-scoring a clip it saw
        # before) replaces its old record
        for j, old in enumerate(self.recs):
            if old["name"] == rec["name"]:
                del self.recs[j], self.ts[j]
                break
        i = bisect.bisect(self.ts, t)
        self.ts.insert(i, t)
        self.recs.insert(i, rec)
        self.cum = None

    def at(self, t):
        """The record at or just behind t, its index, and the running totals."""
        i = bisect.bisect(self.ts, t + 0.05) - 1
        if i < 0:
            if not self.ts:
                return None, -1, [0] * len(SLOTS)
            # nothing scored behind the playhead yet: the nearest frame ahead of it,
            # so the bar shows the agent's result for this stretch, not a wait
            return self.recs[0], 0, [0] * len(SLOTS)
        if self.cum is None:
            run, self.cum = [0] * len(SLOTS), []
            for r in self.recs:
                run = [a + b for a, b in zip(run, r["counts"])]
                self.cum.append(run)
        return self.recs[i], i, self.cum[i]


class MiniFiFeed:
    """The agent's side, read on its own thread: new JSONs in --minifi-meta
    (indexed by clip), the agent process in /proc, and the links handed to its
    ListFile folder."""

    def __init__(self, meta_dir, clips_dir, log, keep_links=3, out_dir=None, out_keep_min=20):
        self.meta_dir, self.clips_dir, self.log = meta_dir, clips_dir, log
        self.keep_links = keep_links
        self.out_dir, self.out_keep_s = out_dir, out_keep_min * 60
        self.lock = threading.Lock()
        self.clips = {}
        self.seen = {}             # name -> mtime_ns, so a rewritten JSON is read again
        self.dir_mtime = None      # meta/'s own mtime at the last full scan
        self.last_walk = 0.0       # and when that walk ran
        self.t_start = time.time()
        self.session = 0           # JSONs written since the renderer started
        self.arrivals = []         # their mtimes, for s per image
        self.last_arrival = 0.0
        self.latest_clip = None    # the clip the agent wrote last
        self.latest_rec = None     # and its newest record, shown while this clip has none
        self.agent = {"pid": None, "cpu_pct": None, "rss_mb": None, "hist": []}
        self._cpu = None
        self.links = []            # links made here, oldest first
        self._lk = threading.RLock()   # feed runs on the fetch thread, release on the main one
        self.fed = set()
        self.ncpu = os.cpu_count() or 1
        self.placement = None      # (cores, nice) the agent's threads are held to, or None to leave it alone
        self.manifest = os.path.join(clips_dir, ".nanostream-links") if clips_dir else None
        self._prune_orphans()
        # read here, not on the thread: its first pass walks all of meta/ first, and a
        # clip that is ready before that walk ends must still wait on a running agent
        self.agent["pid"] = self._agent_pid()
        threading.Thread(target=self._run, daemon=True).start()

    # --- the links this app made, across runs ---
    def _manifest_names(self):
        try:
            return [n for n in open(self.manifest).read().split("\n") if n]
        except OSError:
            return []

    def _write_manifest(self):
        if not self.manifest:
            return
        try:
            os.makedirs(self.clips_dir, exist_ok=True)
            with open(self.manifest + ".tmp", "w") as f:
                f.write("".join(os.path.basename(p) + "\n" for p in self.links))
            os.replace(self.manifest + ".tmp", self.manifest)
        except OSError as e:
            self.log(f"minifi: cannot write {self.manifest}: {e}")

    def _prune_orphans(self):
        """A link from an earlier run whose cache copy is gone holds the clip's
        blocks on its own (link count 1): remove it. Only names this app wrote
        to the manifest are touched, never a clip someone put there by hand."""
        if not self.manifest:
            return
        gone = 0
        for name in self._manifest_names():
            dst = os.path.join(self.clips_dir, name)
            try:
                if os.stat(dst).st_nlink == 1:
                    os.remove(dst)
                    gone += 1
                else:
                    self.links.append(dst)      # still shared with the cache: ours to release later
            except OSError:
                pass
        self._write_manifest()
        if gone:
            self.log(f"minifi: removed {gone} orphaned link(s) from {self.clips_dir}")

    # --- the feed: a downloaded clip into the agent's ListFile folder ---
    def feed(self, local):
        """Link a clip into the ListFile folder. A hard link when the clip is on
        the same filesystem (the http cache under /tmp), else a copy: ListFile
        lists by modification time, and a symlink to an old file on a mount
        would carry the recording's old mtime and never be listed."""
        if not self.clips_dir or not local or not os.path.isfile(local):
            return
        with self._lk:
            self._feed(local)

    def _feed(self, local):
        name = os.path.basename(local)
        if name in self.fed:
            return
        self.fed.add(name)
        dst = os.path.join(self.clips_dir, name)
        if os.path.exists(dst):
            try:
                same = os.path.samefile(local, dst)
            except OSError:
                same = False
            if same and dst not in self.links:
                self.links.append(dst)          # left by a restart: adopt it, so it is released after play
                self._write_manifest()
            self.log(f"minifi: {name} is already in {self.clips_dir}")
            return
        try:
            os.makedirs(self.clips_dir, exist_ok=True)
            try:
                os.link(local, dst)
                how = "linked"
            except OSError:
                shutil.copyfile(local, dst + ".part")
                os.replace(dst + ".part", dst)
                how = "copied"
            self.links.append(dst)
            self._write_manifest()
            self.log(f"minifi: {how} {name} into {self.clips_dir}")
        except OSError as e:
            self.log(f"minifi: cannot hand {name} to the agent: {e}")
            return
        while len(self.links) > self.keep_links:
            old = self.links.pop(0)
            try:
                os.remove(old)
                self.log(f"minifi: removed {os.path.basename(old)} from {self.clips_dir}")
            except OSError:
                pass
        self._write_manifest()

    def release(self, local):
        """The clip has played: drop its link (the agent read it a clip ago)."""
        dst = os.path.join(self.clips_dir or "", os.path.basename(local or ""))
        with self._lk:
            if dst not in self.links:
                return
            self.links.remove(dst)
            try:
                os.remove(dst)
                self.log(f"minifi: released {os.path.basename(dst)} from {self.clips_dir}")
            except OSError:
                pass
            self._write_manifest()

    # --- the reader ---
    def _agent_pid(self):
        pid = self.agent["pid"]
        if pid and os.path.exists(f"/proc/{pid}"):
            return pid
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")[0].decode(errors="replace")
            except OSError:
                continue
            if cmd.endswith("/bin/minifi"):      # the C++ agent; MiNiFi Java runs as java
                self._cpu = None
                return int(d)
        return None

    def _agent_stats(self):
        pid = self._agent_pid()
        a = self.agent
        a["pid"] = pid
        if not pid:
            a["cpu_pct"] = a["rss_mb"] = None
            return
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            ticks = int(f[11]) + int(f[12])
            now = time.time()
            if self._cpu:
                dt = now - self._cpu[1]
                a["cpu_pct"] = 100.0 * (ticks - self._cpu[0]) / os.sysconf("SC_CLK_TCK") / dt / self.ncpu
                a["hist"].append(a["cpu_pct"])
                del a["hist"][:-90]
            self._cpu = (ticks, now)
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    a["rss_mb"] = int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            a["pid"] = None

    def _scan(self):
        # PutFile lands each JSON with a rename, which moves meta/'s own mtime, so an
        # unchanged folder needs no walk (1,600 stats every 0.5 s held the renderer's GIL)
        try:
            dm = os.stat(self.meta_dir).st_mtime_ns
        except OSError:
            return
        # a JSON landing inside the same timestamp tick as the last change leaves the
        # folder's mtime as it was, so walk anyway every 5 s
        if dm == self.dir_mtime and time.time() - self.last_walk < 5:
            return
        self.last_walk = time.time()
        try:
            ents = [e for e in os.scandir(self.meta_dir) if e.name.endswith(".json")
                    and self.seen.get(e.name) != e.stat().st_mtime_ns]
        except OSError:
            return
        self.dir_mtime = dm
        new = []
        for e in ents:
            try:
                st = e.stat()
                mt = st.st_mtime
                d = json.load(open(e.path))
            except (OSError, ValueError):
                self.dir_mtime = None  # half-written; walk again next pass
                continue
            self.seen[e.name] = st.st_mtime_ns
            dets = _jl(d.get("detections"))
            counts = [0] * len(SLOTS)
            for b in dets:
                counts[slot_of(int(b.get("class_id", -1)))] += 1
            scene = [(s.get("class_name", "?"), float(s.get("confidence", 0))) for s in _jl(d.get("scene"))]
            try:
                t = float(d.get("video.t_s", 0))
            except ValueError:
                t = 0.0
            rec = {"clip": d.get("video.clip", ""), "t": t, "counts": counts, "n": len(dets),
                   "scene": scene, "mtime": mt, "name": e.name}
            new.append(rec)
        if not new:
            return
        new.sort(key=lambda r: r["mtime"])
        with self.lock:
            for r in new:
                self.clips.setdefault(r["clip"], Clip()).add(r["t"], r)
                if r["mtime"] >= self.t_start:
                    self.session += 1
                    self.arrivals.append(r["mtime"])
                    self.last_arrival = max(self.last_arrival, time.time())
                    self.latest_clip = r["clip"]
                    self.latest_rec = r
            del self.arrivals[:-40]

    def _prune_out(self):
        """The agent's annotated JPEGs from this run, past out_keep_s."""
        if not self.out_dir or self.out_keep_s <= 0:
            return
        cut, gone = time.time() - self.out_keep_s, 0
        try:
            for e in os.scandir(self.out_dir):
                if not e.name.endswith(".jpg"):
                    continue
                mt = e.stat().st_mtime
                if self.t_start <= mt < cut:
                    os.remove(e.path)
                    gone += 1
        except OSError:
            pass
        if gone:
            self.log(f"minifi: pruned {gone} annotated JPEGs older than {self.out_keep_s // 60:.0f} min from {self.out_dir}")

    # --- where the agent runs ---
    def place(self, cores, nice=15):
        """Hold every thread of the agent to these cores, at no less than this
        nice. The renderer gives it every core while the picture waits on its
        scores, and the stream's spare cores while a clip plays: a decode burst
        on all six cores starves the renderer. Re-applied every 2 s, so threads
        the agent starts later are held too. nice only ever goes up, which needs
        no privilege."""
        self.placement = (set(cores), nice)
        self._apply_placement()

    def _apply_placement(self):
        pid = self.agent["pid"]
        if not pid or not self.placement:
            return
        cores, nice = self.placement
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except OSError:
            return
        for tid in tids:
            try:
                if os.sched_getaffinity(tid) != cores:
                    os.sched_setaffinity(tid, cores)
                if os.getpriority(os.PRIO_PROCESS, tid) < nice:
                    os.setpriority(os.PRIO_PROCESS, tid, nice)
            except OSError:
                pass                    # a thread that ended between the listing and here

    def _run(self):
        n = 0
        while True:
            self._scan()
            if n % 2 == 0:
                self._agent_stats()
            if n % 4 == 0:
                self._apply_placement()
            if n % 120 == 0:
                self._prune_out()
            n += 1
            time.sleep(0.5)

    def scored(self, clip):
        """How many of this clip's frames the agent has written a JSON for."""
        with self.lock:
            c = self.clips.get(clip)
            return len(c.ts) if c else 0

    def s_per_image(self):
        a = self.arrivals
        gaps = [b - x for x, b in zip(a, a[1:]) if 0 < b - x < 30]
        if not gaps:
            return None
        gaps.sort()
        return gaps[len(gaps) // 2]

    def view(self, clip, t):
        with self.lock:
            c = self.clips.get(clip)
            rec, i, cum = c.at(t) if c else (None, -1, [0] * len(SLOTS))
            other = False
            if rec is None and self.latest_rec is not None:
                # no frame of this clip scored yet: what the agent scored last, labelled
                rec, other = self.latest_rec, True
            return {"rec": rec, "idx": i, "cum": cum, "ticks": list(c.ts) if c else [],
                    "ahead": None if rec is None or other else rec["t"] - t, "other": other,
                    "session": self.session, "latest_clip": self.latest_clip,
                    "pulse": time.time() - self.last_arrival}


class MiniFiBar:
    """Draws the bar. The bar is rendered at 1280 wide into a cache and redrawn
    only when what it shows changes (about 10 times a second at most); every
    frame gets the cached bar copied in plus the playhead."""

    def __init__(self, feed):
        self.feed = feed
        self.cache = None
        self.key = None
        self.flash = {}            # slot -> time its count last changed
        self.last_counts = None
        self.scene_name = None
        self.scene_t = 0.0

    def draw(self, frame, clip, t, duration):
        h, w = frame.shape[:2]
        v = self.feed.view(clip, t)
        a = self.feed.agent
        now = time.time()
        rec = v["rec"]
        if rec is not None and rec["counts"] != self.last_counts:
            if self.last_counts is not None:
                for k, (x, y) in enumerate(zip(rec["counts"], self.last_counts)):
                    if x > y:               # a new sighting flashes; a drop does not
                        self.flash[k] = now
            self.last_counts = rec["counts"]
        if rec is not None and rec["scene"] and rec["scene"][0][0] != self.scene_name:
            self.scene_name, self.scene_t = rec["scene"][0][0], now
        key = (clip, v["idx"], len(v["ticks"]), v["session"], int(now * 10) if v["pulse"] < 1.0 or
               any(now - f < 0.6 for f in self.flash.values()) or now - self.scene_t < 0.8 or rec is None
               else int(now), a["pid"])
        if key != self.key or self.cache is None:
            self.cache = self._render(v, a, clip, t, duration, now)
            self.key = key
        bar = self.cache
        bh = BAR_H
        if w != DESIGN_W:
            bh = int(round(BAR_H * w / DESIGN_W))
            bar = cv2.resize(self.cache, (w, bh), interpolation=cv2.INTER_AREA)
        y0 = h - bh
        frame[y0:h] = bar
        # the timeline along the top edge: scored ticks are in the cache; the playhead is per frame
        if duration and duration > 0:
            px = int(min(max(t / duration, 0), 1) * (w - 1))
            cv2.line(frame, (0, y0), (px, y0), CYAN, 2)
            cv2.circle(frame, (px, y0), 4, WHITE, -1, AA)
        return frame

    # ---------------------------------------------------------------
    def _render(self, v, a, clip, t, duration, now):
        W, H = DESIGN_W, BAR_H
        img = np.empty((H, W, 3), np.uint8)
        # a soft vertical gradient, darker at the bottom
        for y in range(H):
            f = y / (H - 1)
            img[y] = [int(BG2[c] * (1 - f) + BG[c] * f) for c in range(3)]
        # the timeline track and MiNiFi's scored ticks
        cv2.line(img, (0, 0), (W, 0), CYAN_DIM, 2)
        if duration and duration > 0:
            for ts in v["ticks"]:
                x = int(min(ts / duration, 1) * (W - 1))
                cv2.line(img, (x, 0), (x, 4), CYAN, 1)
        rec = v["rec"]
        M, R = 440, 880                    # the three sections start at 0, M and R
        for x in (M, R):
            cv2.line(img, (x, 12), (x, H - 8), (60, 45, 30), 1)

        # ---- left: the agent ----
        cv2.putText(img, "MiNiFi", (14, 27), FD, 0.72, CYAN, 1, AA)
        cv2.putText(img, "Object Detection", (98, 27), FD, 0.6, WHITE, 1, AA)
        cv2.putText(img, "(CPU)", (240, 27), F, 0.5, GREY, 1, AA)
        # the pulse: bright on a fresh JSON, fading over a second
        p = v["pulse"]
        glow = max(0.0, 1.0 - p) if p < 1.0 else 0.0
        col = tuple(int(CYAN_DIM[c] + (CYAN[c] - CYAN_DIM[c]) * glow) for c in range(3))
        if a["pid"] is None:
            col = RED
        cv2.circle(img, (300, 21), 5, col, -1, AA)
        if glow > 0:
            cv2.circle(img, (300, 21), int(6 + 6 * (1 - glow)), CYAN, 1, AA)
        state, scol = self._state(v, a, clip, t)
        (sw, _), _ = cv2.getTextSize(state, F, 0.4, 1)
        cv2.rectangle(img, (316, 11), (326 + sw, 28), scol, 1)
        cv2.putText(img, state, (321, 24), F, 0.4, scol, 1, AA)
        spi = self.feed.s_per_image()
        cells = [("SCORED", f"{v['session']:,}"),
                 ("S/IMG", "--" if spi is None else f"{spi:.1f}"),
                 ("CPU", "--" if a["cpu_pct"] is None else f"{a['cpu_pct']:.0f}%"),
                 ("RSS", "--" if a["rss_mb"] is None else f"{a['rss_mb']:.0f}M")]
        x = 14
        for lab, val in cells:
            cv2.putText(img, lab, (x, 49), F, 0.36, GREY, 1, AA)
            (lw, _), _ = cv2.getTextSize(lab, F, 0.36, 1)
            cv2.putText(img, val, (x + lw + 5, 49), F, 0.46, WHITE, 1, AA)
            (vw, _), _ = cv2.getTextSize(val, F, 0.46, 1)
            x += lw + vw + 16
        # the agent's CPU as a sparkline at the end of the row
        hist = a["hist"][-60:]
        if len(hist) > 1 and x < M - 40:
            sx0, sx1, sy0, sy1 = x, M - 14, 38, 52
            pts = np.array([[int(sx0 + i * (sx1 - sx0) / (len(hist) - 1)),
                             int(sy1 - min(max(hv, 0), 100) / 100 * (sy1 - sy0))] for i, hv in enumerate(hist)], np.int32)
            cv2.polylines(img, [pts], False, CYAN, 1, AA)

        # ---- middle: the counters ----
        sw_ = (R - M - 16) // len(SLOTS)
        counts = rec["counts"] if rec else [0] * len(SLOTS)
        for k, (lab, _, c) in enumerate(SLOTS):
            x0 = M + 8 + k * sw_
            n, tot = counts[k], v["cum"][k]
            fl = now - self.flash.get(k, 0)
            if fl < 0.6:                    # a changed count flashes its cell
                a_ = 1 - fl / 0.6
                cv2.rectangle(img, (x0 + 2, 8), (x0 + sw_ - 2, H - 6),
                              tuple(int(BG2[i] + (c[i] - BG2[i]) * 0.35 * a_) for i in range(3)), -1)
            on = n > 0
            txt = str(n)
            (tw, _), _ = cv2.getTextSize(txt, FD, 0.74, 1)
            cv2.putText(img, txt, (x0 + (sw_ - tw) // 2, 30), FD, 0.74, c if on else DIM, 1, AA)
            (lw, _), _ = cv2.getTextSize(lab, F, 0.34, 1)
            cv2.putText(img, lab, (x0 + (sw_ - lw) // 2, 43), F, 0.34, GREY if on else DIM, 1, AA)
            tt = f"{tot}" if tot else ""
            if tt:
                (ttw, _), _ = cv2.getTextSize(tt, F, 0.38, 1)
                cv2.putText(img, tt, (x0 + (sw_ - ttw) // 2, 56), F, 0.38, CYAN_DIM if not on else CYAN, 1, AA)
            if on:
                cv2.line(img, (x0 + 10, 9), (x0 + sw_ - 10, 9), c, 2)

        # ---- right: the scene ----
        cv2.putText(img, "SCENE", (R + 14, 22), F, 0.36, GREY, 1, AA)
        cv2.putText(img, "Places365", (R + 14, 36), F, 0.32, DIM, 1, AA)
        if rec and rec["scene"]:
            name, conf = rec["scene"][0]
            nm = name.replace("_", " ")
            fresh = now - self.scene_t
            col = WHITE if fresh > 0.8 else tuple(int(CYAN[i] + (WHITE[i] - CYAN[i]) * fresh / 0.8) for i in range(3))
            scale = 0.7
            while cv2.getTextSize(nm, FD, scale, 1)[0][0] > 250 and scale > 0.45:
                scale -= 0.05
            cv2.putText(img, nm, (R + 92, 28), FD, scale, col, 1, AA)
            pct = f"{conf * 100:.0f}%"
            cv2.putText(img, pct, (W - 14 - cv2.getTextSize(pct, FD, 0.6, 1)[0][0], 28), FD, 0.6, CYAN, 1, AA)
            # the confidence bar
            bx0, bx1 = R + 92, W - 14
            cv2.rectangle(img, (bx0, 34), (bx1, 37), (60, 45, 30), -1)
            cv2.rectangle(img, (bx0, 34), (bx0 + int((bx1 - bx0) * min(conf, 1)), 37), CYAN, -1)
            # runners-up
            x = R + 92
            for k, (n2, c2) in enumerate(rec["scene"][1:3], start=2):
                s = n2.replace("_", " ")
                if len(s) > 17:
                    s = s[:16] + "."
                t2 = f"{k}  {s}  {c2 * 100:.0f}%"
                cv2.putText(img, t2, (x, 53), F, 0.36, GREY, 1, AA)
                x += 150
        else:
            msg = "waiting for MiNiFi"
            cv2.putText(img, msg, (R + 92, 30), F, 0.5, DIM, 1, AA)
            # a sweep while nothing is scored for this clip
            sx = R + 92 + int((now * 120) % 260)
            cv2.line(img, (R + 92, 42), (W - 14, 42), (50, 38, 26), 2)
            cv2.line(img, (sx, 42), (min(sx + 40, W - 14), 42), CYAN, 2)
        return img

    def _state(self, v, a, clip, t):
        if a["pid"] is None:
            return "AGENT DOWN", RED
        rec = v["rec"]
        if rec is None:
            return "QUEUED", AMBER
        if v["other"]:                  # this clip has nothing scored: the agent's latest, from another clip
            return ("PREV CLIP" if rec["clip"] < clip else "NEXT CLIP"), AMBER
        if v["ahead"] is not None and v["ahead"] > 0.05:
            return f"AHEAD {v['ahead']:.0f}s", CYAN
        lag = t - rec["t"]
        if lag > 15:
            return f"BEHIND {lag:.0f}s", AMBER
        return "IN SYNC", NANO_GREEN
