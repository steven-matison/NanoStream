#!/usr/bin/env python3
"""Walk dashcam video off a card or drive and run every sampled frame through
the Hailo daemon on the Jetson.

    python3 dashcam_walker.py --source /media/$USER/DASHCAM --out ~/dashcam-runs/run1
    python3 dashcam_walker.py --source /opt/nvidia/vpi4/samples/assets/dashcam.mp4 \
        --fps 2 --max-frames 40 --out /tmp/dc/run1 --crops

One JSON record per sampled frame goes to <out>/records.jsonl and, with
--post, to a MiNiFi ListenHTTP door; plate
crops and annotated keyframes land beside it. Needs only cv2: run it with the
venv install.sh recorded (`. ~/.nanostream/env && $NANOSTREAM_PY ...`); the
daemon holds the Hailo and OCR packages.

Reads nothing but the video; writes only under --out.
"""

import argparse
import datetime as dt
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".ts", ".mkv", ".m4v"}
# the device name written into every record
DEVICE = os.environ.get("DASHCAM_DEVICE", "nano")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def list_videos(source):
    """A file, a directory or mount point, or an http(s) directory index
    (`python3 -m http.server` on the box that holds the drive)."""
    if source.startswith(("http://", "https://")):
        return sorted(walk_http_index(source))
    if os.path.isfile(source):
        return [source]
    found = []
    for root, _dirs, files in os.walk(source):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXT and not f.startswith("."):
                found.append(os.path.join(root, f))
    return sorted(found)


def walk_http_index(url, depth=0):
    """Follow the <a href> links of a python http.server listing, two levels deep."""
    import html.parser

    class Links(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for k, v in attrs:
                    if k == "href" and v and not v.startswith(("?", "/", "..")):
                        self.hrefs.append(v)

    if not url.endswith("/"):
        return [url] if os.path.splitext(url)[1].lower() in VIDEO_EXT else []
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            page = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    p = Links()
    p.feed(page)
    out = []
    for href in p.hrefs:
        full = url + href
        if href.endswith("/") and depth < 2:
            out.extend(walk_http_index(full, depth + 1))
        elif os.path.splitext(href)[1].lower() in VIDEO_EXT and not href.startswith("."):
            out.append(full)
    return out


def fetch_to_cache(url, cache_dir, log):
    """Download one video to a local file. The stock `python3 -m http.server`
    has no byte-range support, and ffmpeg needs to seek inside an MP4 (the
    moov atom sits at the end), so streaming straight off it fails with
    'partial file'. A local copy is seekable and is deleted after the walk."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, urllib.parse.unquote(url.rsplit("/", 1)[-1]))
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(local, "wb") as f:
        n = 0
        while True:
            chunk = r.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
    secs = max(time.time() - t0, 0.001)
    log(f"fetched {n / 1e6:.0f} MB in {secs:.0f} s ({n * 8 / secs / 1e6:.0f} Mbit/s): {url}")
    return local


def source_mtime(path):
    if path.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(path, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                lm = r.headers.get("Last-Modified")
            if lm:
                return dt.datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
        return None
    return dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_json(url, data, content_type, timeout):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), r.status


def health(endpoint):
    base = endpoint.rsplit("/", 1)[0]
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def wait_cool(endpoint, max_temp, log):
    """The Hailo module throttles and then refuses streams when hot: pause at
    max_temp, go on once the die is 5 C under it."""
    paused = False
    while True:
        t = health(endpoint).get("chip_temp_c")
        if t is None or t < (max_temp - 5 if paused else max_temp):
            return
        if not paused:
            log(f"hailo die {t} C >= {max_temp:g}: pausing until it is under {max_temp - 5:g}")
        paused = True
        time.sleep(10)


# Burned-in telemetry overlays, as (x0, y0, x1, y1) fractions of the frame. The
# Corvette PDR puts speed top-left, the gear ring top-right, and the G-force and
# RPM readouts in a band across the bottom; left in, the OCR reads "LATERAL"
# and the detector sees kites and clocks. Blacked out before the POST only; the
# saved keyframes keep the picture.
HUD_MASKS = {
    "pdr": [(0.03, 0.02, 0.19, 0.22), (0.83, 0.02, 0.96, 0.22), (0.0, 0.76, 1.0, 1.0)],
    "none": [],
}


def mask_hud(frame, zones):
    if not zones:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    for fx0, fy0, fx1, fy1 in zones:
        out[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = 0
    return out


def annotate(frame, rec, scale):
    for v in rec.get("vehicles", []):
        x0, y0, x1, y1 = [int(round(c * scale)) for c in v["box"]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 140, 0), 2)
        cv2.putText(frame, f'{v["label"]} {v["score"]:.2f}', (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
        for p in v.get("plates", []):
            px0, py0, px1, py1 = [int(round(c * scale)) for c in p["box"]]
            cv2.rectangle(frame, (px0, py0), (px1, py1), (0, 220, 60), 2)
            if p.get("text"):
                cv2.putText(frame, p["text"], (px0, max(12, py0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 60), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="a video file, or a directory / mount point")
    ap.add_argument("--out", required=True, help="run directory for records, crops, keyframes")
    ap.add_argument("--endpoint", default="http://127.0.0.1:5911/plates")
    ap.add_argument("--post", help="MiNiFi door to POST each record to, e.g. http://127.0.0.1:8083/dashcam")
    ap.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of video")
    ap.add_argument("--width", type=int, default=1280, help="downscale wider frames to this before POSTing")
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many sampled frames (0 = all)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N videos after sorting")
    ap.add_argument("--start-at", help="start at the first video whose basename sorts >= this (e.g. 250123)")
    ap.add_argument("--resume", action="store_true",
                    help="skip videos already recorded in <out>/records.jsonl (continue, don't start over)")
    ap.add_argument("--max-temp", type=float, default=88.0, help="pause when the Hailo die reaches this, until it is 5 C under")
    ap.add_argument("--crops", action="store_true", help="ask for plate crops and save them")
    ap.add_argument("--keyframe-every", type=int, default=0,
                    help="also save an annotated keyframe every N sampled frames (0 = only when a plate is found)")
    ap.add_argument("--mask-hud", default="none", choices=sorted(HUD_MASKS),
                    help="black out a burned-in telemetry overlay before inference (pdr = Corvette PDR)")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "plates"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "keyframes"), exist_ok=True)
    records_path = os.path.join(a.out, "records.jsonl")
    log_path = os.path.join(a.out, "walker.log")

    def log(msg):
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    videos = list_videos(a.source)
    if not videos:
        log(f"no video under {a.source}")
        sys.exit(2)
    found = len(videos)
    if a.resume and os.path.exists(records_path):
        done = set()
        with open(records_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("source"))
                except json.JSONDecodeError:
                    pass
        videos = [v for v in videos if v not in done]
        log(f"resume: {len(done)} clip(s) already in records.jsonl, {len(videos)} of {found} left")
    if a.start_at:
        videos = [v for v in videos
                  if os.path.basename(urllib.parse.unquote(v)) >= a.start_at]
        log(f"start-at {a.start_at!r}: {len(videos)} of {found} remain")
    if a.skip:
        videos = videos[a.skip:]
        log(f"skip {a.skip}: {len(videos)} of {found} remain")
    if not videos:
        log("no video left after skip/start-at/resume")
        sys.exit(2)
    h = health(a.endpoint)
    if not h.get("ok"):
        log(f"daemon not healthy at {a.endpoint}")
        sys.exit(3)
    log(f"{len(videos)} video(s) under {a.source}; daemon models {h.get('models')} die {h.get('chip_temp_c')} C")

    url = a.endpoint + ("?crops=1&" if a.crops else "?") + f"min_score={a.min_score}"
    totals = {"files": 0, "frames": 0, "labels": {}, "plates_found": 0, "plates_read": 0,
              "texts": {}, "post_ok": 0, "post_failed": 0, "errors": 0}
    rtt = []
    started = time.time()
    stop = False

    with open(records_path, "a") as records:
        for path in videos:
            if stop:
                break
            local = path
            if path.startswith(("http://", "https://")):
                try:
                    local = fetch_to_cache(path, os.path.join(a.out, ".cache"), log)
                except Exception as e:
                    log(f"fetch failed {path}: {e}")
                    totals["errors"] += 1
                    continue
            cap = cv2.VideoCapture(local)
            if not cap.isOpened():
                log(f"cannot open {path}")
                totals["errors"] += 1
                continue
            vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(vfps / a.fps)))
            stem = os.path.splitext(os.path.basename(urllib.parse.unquote(path)))[0]
            mtime = source_mtime(path)
            log(f"{path}: {nframes} frames @ {vfps:.2f} fps, sampling every {step}")
            totals["files"] += 1
            i = -1
            while True:
                if not cap.grab():
                    break
                i += 1
                if i % step:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                scale = 1.0
                if frame.shape[1] > a.width:
                    scale = a.width / frame.shape[1]
                    frame = cv2.resize(frame, (a.width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", mask_hud(frame, HUD_MASKS[a.mask_hud]),
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                wait_cool(a.endpoint, a.max_temp, log)
                t0 = time.perf_counter()
                try:
                    res, _ = http_json(url, jpg.tobytes(), "image/jpeg", 120)
                # URLError, timeouts and a dropped connection are all OSError or HTTPException
                except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                    log(f"frame {i}: daemon error {e}")
                    totals["errors"] += 1
                    continue
                ms = (time.perf_counter() - t0) * 1000
                rtt.append(ms)
                if not res.get("ok"):
                    log(f"frame {i}: {res.get('error')}")
                    totals["errors"] += 1
                    continue

                vehicles = []
                for k, v in enumerate(res.get("vehicles", [])):
                    plates = []
                    for j, p in enumerate(v.get("plates", [])):
                        entry = {x: p.get(x) for x in ("box", "score", "width_px", "height_px",
                                                       "readable", "text", "text_conf", "ocr_pass",
                                                       "text_candidates", "lprnet_digits")}
                        if p.get("crop_b64"):
                            import base64
                            crop_name = f"{stem}-f{i:06d}-v{k}-p{j}.jpg"
                            with open(os.path.join(a.out, "plates", crop_name), "wb") as f:
                                f.write(base64.b64decode(p["crop_b64"]))
                            entry["crop"] = os.path.join("plates", crop_name)
                        plates.append(entry)
                        totals["plates_found"] += 1
                        if p.get("text"):
                            totals["plates_read"] += 1
                            totals["texts"][p["text"]] = totals["texts"].get(p["text"], 0) + 1
                    vehicles.append({"label": v["label"], "score": v["score"], "box": v["box"], "plates": plates})
                for lbl, n in res.get("counts", {}).items():
                    totals["labels"][lbl] = totals["labels"].get(lbl, 0) + n

                rec = {
                    "device": DEVICE, "ts_utc": utc_now(),
                    "source": path, "source_mtime_utc": mtime, "frame": i, "t_s": round(i / vfps, 2),
                    "image": res.get("image"),
                    "counts": res.get("counts", {}),
                    "vehicles": vehicles,
                    "plates_found": res.get("plates_found", 0),
                    "latency_ms": {"round_trip": round(ms, 1), "decode": res.get("decode_ms"),
                                   "detect": res.get("detect_ms"), "plates": res.get("plates_ms")},
                    "chip_temp_c": res.get("chip_temp_c"),
                    "accelerator": res.get("accelerator"), "models": res.get("models"),
                }
                records.write(json.dumps(rec) + "\n")
                records.flush()
                totals["frames"] += 1

                if a.post:
                    try:
                        http_json(a.post, json.dumps(rec).encode("utf-8"), "application/json", 10)
                        totals["post_ok"] += 1
                    except Exception as e:
                        totals["post_failed"] += 1
                        if totals["post_failed"] <= 3:
                            log(f"post failed: {e}")

                if rec["plates_found"] or (a.keyframe_every and totals["frames"] % a.keyframe_every == 0):
                    cv2.imwrite(os.path.join(a.out, "keyframes", f"{stem}-f{i:06d}.jpg"),
                                annotate(frame, res, 1.0), [cv2.IMWRITE_JPEG_QUALITY, 85])

                if totals["frames"] % 20 == 0:
                    log(f"{totals['frames']} frames, {totals['plates_found']} plates found, "
                        f"{totals['plates_read']} read, rtt {ms:.0f} ms, die {rec['chip_temp_c']} C")
                if a.max_frames and totals["frames"] >= a.max_frames:
                    stop = True
                    break
            cap.release()
            if local != path:
                try:
                    os.remove(local)
                except OSError:
                    pass

    rtt.sort()
    elapsed = time.time() - started
    summary = {
        **totals,
        "elapsed_s": round(elapsed, 1),
        "sampled_fps": round(totals["frames"] / elapsed, 2) if elapsed else None,
        "rtt_ms": {"p50": round(rtt[len(rtt) // 2], 1) if rtt else None,
                   "p95": round(rtt[int(len(rtt) * 0.95)], 1) if rtt else None},
        "source": a.source, "endpoint": a.endpoint, "post": a.post, "fps": a.fps,
        "mask_hud": a.mask_hud, "finished_utc": utc_now(),
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("summary " + json.dumps(summary))


if __name__ == "__main__":
    main()
