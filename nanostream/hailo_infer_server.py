"""Resident Hailo-8 vision daemon for the Jetson.

One process holds the models open on the Hailo-8 and serves them over loopback
HTTP, so the renderer and the walker stay thin clients and a model swap is
`systemctl --user restart hailo-infer`, never a stream restart.

Models, all public Hailo Model Zoo v2.20.0 builds for the Hailo-8 (HailoRT 4.24):

    yolov8s.hef                     COCO-80 detection, NMS on the chip
    tiny_yolov4_license_plates.hef  1-class plate detector, decoded on the host
    lprnet.hef                      plate reader, DIGITS ONLY (10 classes + blank,
                                    trained on Israeli plates) - kept as a
                                    cross-check field, not the answer

The plate TEXT comes from RapidOCR (PaddleOCR v4 recognizer on onnxruntime,
CPU) in the same venv, because the Hailo LPRNet cannot emit a letter.

    GET  /health            models, chip temperature, uptime, counters
    POST /detect            image -> COCO detections            ?min_score=0.35
    POST /plates            image -> vehicles -> plates -> text ?min_score=0.35&crops=1&full=1

Body: raw JPEG/PNG bytes, or JSON {"image_b64": "..."}.

Runs as the systemd *user* unit hailo-infer (install.sh writes it), from a venv
that holds HailoRT's Python wheel plus rapidocr-onnxruntime and
opencv-python-headless. HAILO_MODELS, HAILO_INFER_PORT and HAILO_INFER_LOG
override the defaults.
"""

import base64
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
try:
    from hailo_platform import (HEF, VDevice, Device, HailoStreamInterface, InferVStreams,
                                ConfigureParams, InputVStreamParams, OutputVStreamParams,
                                FormatType, HailoSchedulingAlgorithm)
except ImportError:          # the pure helpers below still import (tests, CI); serving needs HailoRT
    HEF = None

BIND_HOST = "127.0.0.1"
PORT = int(os.environ.get("HAILO_INFER_PORT", "5911"))
MODEL_DIR = os.environ.get("HAILO_MODELS", os.path.expanduser("~/hailo/models"))
LOG_PATH = os.path.expanduser(os.environ.get("HAILO_INFER_LOG", "~/.nanostream/hailo_infer_server.log"))
MAX_REQUEST_BYTES = 16 * 1024 * 1024

DETECT_HEF = os.path.join(MODEL_DIR, "yolov8s.hef")
PLATE_HEF = os.path.join(MODEL_DIR, "tiny_yolov4_license_plates.hef")
LPR_HEF = os.path.join(MODEL_DIR, "lprnet.hef")

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush").split()
VEHICLES = {"car", "truck", "bus", "motorcycle"}

# tiny_yolov4_license_plates.yaml (model zoo v2.20.0): strides 32 and 16, anchors
# per stride, yolo_v4 decoding with scale_x_y 1.05, score 0.1 / iou 0.3 defaults.
PLATE_STRIDES = (32, 16)
PLATE_ANCHORS = (((81, 82), (135, 169), (344, 319)), ((10, 14), (23, 27), (37, 58)))
PLATE_SCALE_XY = 1.05
PLATE_IOU = 0.3

# ocr_postprocessing.py (model zoo): CHARS = 0..9 then "-" as the CTC blank.
LPR_CHARS = "0123456789-"
MIN_OCR_WIDTH_PX = int(os.environ.get("HAILO_INFER_MIN_PLATE_PX", "40"))
OCR_SURE = float(os.environ.get("HAILO_INFER_OCR_SURE", "0.6"))   # below this the full OCR pass runs

STARTED_AT = time.time()
COUNTERS = {"detect": 0, "plates": 0, "errors": 0}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_TEMP_DEV = None
_TEMP_LOCK = threading.Lock()


def chip_temp():
    """Die temperature from the Hailo health monitor; None if the device is busy.

    One Device handle, opened on first use and kept. HailoRT writes two info lines to hailort.log
    every time a Device opens, whatever HAILORT_LOGGER_LEVEL says, and the renderer polls /health
    every few seconds: opening per call grew the log by about 1 MB an hour. A failed read drops
    the handle, and the next call opens a new one.
    """
    global _TEMP_DEV
    with _TEMP_LOCK:
        try:
            if _TEMP_DEV is None:
                _TEMP_DEV = Device()
            t = _TEMP_DEV.control.get_chip_temperature()
            return round(max(t.ts0_temperature, t.ts1_temperature), 1)
        except Exception:
            try:
                if _TEMP_DEV is not None:
                    _TEMP_DEV.release()
            except Exception:
                pass
            _TEMP_DEV = None
            return None


def letterbox(img, size, pad=114):
    """Fit a PIL image into size x size; return array, scale, pad x, pad y."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return np.ascontiguousarray(np.array(canvas, dtype=np.uint8)), scale, px, py


def nms(boxes, scores, iou_thresh):
    """Plain greedy NMS; boxes are [x0, y0, x1, y1]."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a + b - inter, 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


class HailoModel:
    """One configured network group with its vstreams held open."""

    def __init__(self, target, path):
        self.hef = HEF(path)
        self.name = self.hef.get_network_group_names()[0]
        cp = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = target.configure(self.hef, cp)[0]
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_infos = self.hef.get_output_vstream_infos()
        self.size_hw = tuple(self.in_info.shape[:2])
        ivp = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        ovp = OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)
        self.pipe = InferVStreams(self.ng, ivp, ovp)
        self.pipe.__enter__()

    def infer(self, frame_hwc):
        batch = np.expand_dims(frame_hwc, 0)
        return self.pipe.infer({self.in_info.name: batch})


class Vision:
    """yolov8s + plate detector + LPRNet on one VDevice under the scheduler, plus RapidOCR."""

    def __init__(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(params)
        self.lock = threading.Lock()  # one request on the chip at a time; the walker is serial anyway
        self.detect_m = HailoModel(self.target, DETECT_HEF)
        self.plate_m = HailoModel(self.target, PLATE_HEF)
        self.lpr_m = HailoModel(self.target, LPR_HEF)
        log(f"hailo models: {self.detect_m.name} {self.detect_m.size_hw}, "
            f"{self.plate_m.name} {self.plate_m.size_hw}, {self.lpr_m.name} {self.lpr_m.size_hw}")
        self.ocr = None
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            log("rapidocr ready (rec-only on plate crops)")
        except Exception as e:  # the daemon still detects without OCR
            log(f"rapidocr unavailable: {e}")

    # -- yolov8s, NMS on the chip ------------------------------------------------
    def detect(self, img, min_score):
        size = self.detect_m.size_hw[0]
        frame, scale, px, py = letterbox(img, size)
        with self.lock:
            res = self.detect_m.infer(frame)
        per_class = next(iter(res.values()))[0]
        w, h = img.size
        out = []
        for cls, boxes in enumerate(per_class):
            for y0, x0, y1, x1, score in np.asarray(boxes).reshape(-1, 5):
                if score < min_score:
                    continue
                bx = [(x0 * size - px) / scale, (y0 * size - py) / scale,
                      (x1 * size - px) / scale, (y1 * size - py) / scale]
                bx = [float(max(0.0, min(v, lim))) for v, lim in zip(bx, (w, h, w, h))]
                out.append({"label": COCO[cls] if cls < len(COCO) else str(cls),
                            "score": round(float(score), 3),
                            "box": [round(v, 1) for v in bx]})
        return sorted(out, key=lambda d: -d["score"])

    # -- tiny_yolov4 plates, decoded on the host ------------------------------------
    def plates(self, img, min_score):
        size = self.plate_m.size_hw[0]
        frame, scale, px, py = letterbox(img, size)
        with self.lock:
            res = self.plate_m.infer(frame)
        by_name = {k.split("/")[-1]: v[0] for k, v in res.items()}
        # conv19 is 13x13 (stride 32), conv21 is 26x26 (stride 16)
        branches = sorted(by_name.values(), key=lambda a: a.shape[0])
        boxes, scores = [], []
        for arr, stride, anchors in zip(branches, PLATE_STRIDES, PLATE_ANCHORS):
            g = arr.shape[0]
            t = arr.reshape(g, g, 3, 6).astype(np.float32)
            gy, gx = np.mgrid[0:g, 0:g]
            cx = (sigmoid(t[..., 0]) * PLATE_SCALE_XY - 0.5 * (PLATE_SCALE_XY - 1) + gx[..., None]) * stride
            cy = (sigmoid(t[..., 1]) * PLATE_SCALE_XY - 0.5 * (PLATE_SCALE_XY - 1) + gy[..., None]) * stride
            aw = np.array([a[0] for a in anchors], dtype=np.float32)
            ah = np.array([a[1] for a in anchors], dtype=np.float32)
            bw = np.exp(np.clip(t[..., 2], -10, 10)) * aw
            bh = np.exp(np.clip(t[..., 3], -10, 10)) * ah
            conf = sigmoid(t[..., 4]) * sigmoid(t[..., 5])
            m = conf >= min_score
            for x, y, w_, h_, c in zip(cx[m], cy[m], bw[m], bh[m], conf[m]):
                boxes.append([x - w_ / 2, y - h_ / 2, x + w_ / 2, y + h_ / 2])
                scores.append(float(c))
        keep = nms(boxes, scores, PLATE_IOU)
        w, h = img.size
        out = []
        for i in keep:
            x0, y0, x1, y1 = boxes[i]
            bx = [(x0 - px) / scale, (y0 - py) / scale, (x1 - px) / scale, (y1 - py) / scale]
            bx = [float(max(0.0, min(v, lim))) for v, lim in zip(bx, (w, h, w, h))]
            if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 2:
                continue
            out.append({"score": round(scores[i], 3), "box": [round(v, 1) for v in bx]})
        return sorted(out, key=lambda d: -d["score"])

    # -- LPRNet digits (cross-check only) -------------------------------------------
    def lpr_digits(self, crop):
        h, w = self.lpr_m.size_hw
        arr = np.ascontiguousarray(np.array(crop.resize((w, h), Image.BILINEAR), dtype=np.uint8))
        with self.lock:
            res = self.lpr_m.infer(arr)
        logits = next(iter(res.values()))[0]          # (1, 19, 11)
        seq = np.argmax(np.mean(logits, axis=0), axis=-1) if logits.ndim == 3 else np.argmax(logits, axis=-1)
        blank = len(LPR_CHARS) - 1
        out, prev = [], None
        for c in seq.tolist():
            if c != blank and c != prev:
                out.append(LPR_CHARS[c])
            prev = c
        return "".join(out)

    # -- RapidOCR on the crop ----------------------------------------------------------
    def _ocr_run(self, arr, det):
        try:
            result, _ = self.ocr(arr, use_det=det, use_cls=False, use_rec=True)
        except Exception as e:
            log(f"ocr failed: {e}")
            return []
        out = []
        for item in result or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[-2], str):
                out.append((item[-2], float(item[-1])))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append((str(item[1]), float(item[2])))
        return out

    @staticmethod
    def _clean(text):
        # ASCII letters and digits only: the recognizer's dictionary is CJK+Latin
        # and a stray CJK glyph on a tiny crop is noise, not a plate.
        text = "".join(ch for ch in text.upper() if ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in " -")
        return " ".join(text.split())

    def ocr_text(self, crop):
        """Recognizer alone first (~35 ms); when it is unsure, the full
        detect+recognize pass on a 2x upscale (~0.3-1.5 s), which read a plate the
        recognizer alone garbled. Returns
        (text, conf, pass)."""
        if self.ocr is None:
            return None, None, None
        w, h = crop.size
        if h < 64:
            k = 64.0 / h
            crop = crop.resize((max(16, int(w * k)), 64), Image.BICUBIC)
        arr = np.array(crop.convert("RGB"))[:, :, ::-1]   # RapidOCR takes BGR
        hits = [(self._clean(t), c) for t, c in self._ocr_run(arr, det=False)]
        hits = [(t, c) for t, c in hits if t]
        if hits and min(c for _, c in hits) >= OCR_SURE and len("".join(hits[0][0].split())) >= 4:
            cands = [{"text": t, "conf": round(c, 3)} for t, c in hits]
            return " ".join(t for t, _ in hits), round(min(c for _, c in hits), 3), "rec", cands
        up = crop.resize((crop.size[0] * 2, crop.size[1] * 2), Image.BICUBIC)
        arr2 = np.array(up.convert("RGB"))[:, :, ::-1]
        full = [(self._clean(t), c) for t, c in self._ocr_run(arr2, det=True)]
        full = [(t, c) for t, c in full if len(t.replace(" ", "")) >= 3]
        if full:
            # a plate number carries a digit; the state name on the plate header does
            # not. Then the longest run, then the recognizer's score. Every candidate
            # is returned beside the pick, so a vanity plate is not lost downstream.
            full.sort(key=lambda tc: (-any(ch.isdigit() for ch in tc[0]),
                                      -len(tc[0].replace(" ", "")), -tc[1]))
            cands = [{"text": t, "conf": round(c, 3)} for t, c in full]
            return full[0][0], round(full[0][1], 3), "det+rec", cands
        if hits:
            cands = [{"text": t, "conf": round(c, 3)} for t, c in hits]
            return " ".join(t for t, _ in hits), round(min(c for _, c in hits), 3), "rec", cands
        return None, None, None, []


def expand(box, w, h, fx=0.08, fy=0.08):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * fx, (y1 - y0) * fy
    return [max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy)]


def b64_jpeg(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_request(body, content_type):
    stripped = body.lstrip()
    if content_type.startswith("application/json") or stripped[:1] in (b"{", b"["):
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        b64 = payload.get("image_b64") or payload.get("image")
        if not b64:
            raise ValueError('JSON body needs "image_b64"')
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB"), payload
    return Image.open(io.BytesIO(body)).convert("RGB"), {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the journal quiet; the log file has what matters
        pass

    def _respond(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") not in ("/health", ""):
            self._respond(404, {"ok": False, "error": f"no such path: {self.path}"})
            return
        self._respond(200, {
            "ok": True,
            "accelerator": "hailo8",
            "hailort": __import__("hailo_platform").__version__,
            "models": {"detect": VISION.detect_m.name, "plates": VISION.plate_m.name,
                       "lpr": VISION.lpr_m.name, "ocr": "rapidocr-onnxruntime" if VISION.ocr else None},
            "chip_temp_c": chip_temp(),
            "uptime_s": round(time.time() - STARTED_AT, 1),
            "counters": dict(COUNTERS),
        })

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path.rstrip("/")
        if route not in ("/detect", "/plates"):
            self._respond(404, {"ok": False, "error": f"no such path: {self.path}"})
            return
        q = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_REQUEST_BYTES:
            self._respond(413, {"ok": False, "error": f"body exceeds {MAX_REQUEST_BYTES} bytes"})
            return
        body = self.rfile.read(length) if length else b""
        if not body:
            self._respond(400, {"ok": False, "error": "empty body"})
            return
        try:
            t0 = time.perf_counter()
            img, payload = decode_request(body, (self.headers.get("Content-Type") or "").lower())
            min_score = float(q.get("min_score", [payload.get("min_score", 0.35)])[0])
            t1 = time.perf_counter()
            if route == "/detect":
                dets = VISION.detect(img, min_score)
                t2 = time.perf_counter()
                COUNTERS["detect"] += 1
                self._respond(200, {
                    "ok": True, "accelerator": "hailo8", "model": VISION.detect_m.name,
                    "image": {"width": img.size[0], "height": img.size[1]},
                    "detections": dets,
                    "decode_ms": round((t1 - t0) * 1000, 2),
                    "inference_ms": round((t2 - t1) * 1000, 2),
                })
                return
            want_crops = q.get("crops", ["0"])[0] in ("1", "true")
            full_frame = q.get("full", ["0"])[0] in ("1", "true")
            self._respond(200, self.plates_pipeline(img, min_score, want_crops, full_frame, t0, t1))
        except (ValueError, json.JSONDecodeError, OSError) as e:
            COUNTERS["errors"] += 1
            self._respond(400, {"ok": False, "error": str(e)})
        except Exception as e:
            COUNTERS["errors"] += 1
            log(f"{route} FAILED: {e!r}")
            self._respond(500, {"ok": False, "error": repr(e)})

    @staticmethod
    def plates_pipeline(img, min_score, want_crops, full_frame, t0, t1):
        w, h = img.size
        dets = VISION.detect(img, min_score)
        t2 = time.perf_counter()
        vehicles = []
        n_plates = 0
        # the chip's NMS is per class, so one vehicle can come back as both
        # "truck" and "car"; keep the higher-scored box across vehicle classes
        vdets = [d for d in dets if d["label"] in VEHICLES]
        keep = nms([d["box"] for d in vdets], [d["score"] for d in vdets], 0.7)
        for d in (vdets[i] for i in keep):
            vbox = expand(d["box"], w, h)
            vcrop = img.crop(tuple(int(round(v)) for v in vbox))
            found = []
            if vcrop.size[0] >= 32 and vcrop.size[1] >= 24:
                for p in VISION.plates(vcrop, min_score):
                    pbox = [p["box"][0] + vbox[0], p["box"][1] + vbox[1],
                            p["box"][2] + vbox[0], p["box"][3] + vbox[1]]
                    found.append(Handler.read_plate(img, pbox, p["score"], want_crops))
            n_plates += len(found)
            vehicles.append({**d, "plates": found})
        extra = []
        if full_frame:
            for p in VISION.plates(img, min_score):
                extra.append(Handler.read_plate(img, p["box"], p["score"], want_crops))
            n_plates += len(extra)
        t3 = time.perf_counter()
        COUNTERS["plates"] += 1
        return {
            "ok": True, "accelerator": "hailo8",
            "models": {"detect": VISION.detect_m.name, "plates": VISION.plate_m.name,
                       "lpr": VISION.lpr_m.name, "ocr": "rapidocr-onnxruntime" if VISION.ocr else None},
            "image": {"width": w, "height": h},
            "counts": {lbl: sum(1 for d in dets if d["label"] == lbl) for lbl in sorted({d["label"] for d in dets})},
            "detections": dets,
            "vehicles": vehicles,
            "plates_outside_vehicles": extra,
            "plates_found": n_plates,
            "decode_ms": round((t1 - t0) * 1000, 2),
            "detect_ms": round((t2 - t1) * 1000, 2),
            "plates_ms": round((t3 - t2) * 1000, 2),
            "chip_temp_c": chip_temp(),
        }

    @staticmethod
    def read_plate(img, pbox, score, want_crops):
        w, h = img.size
        pb = expand(pbox, w, h, 0.06, 0.15)
        crop = img.crop(tuple(int(round(v)) for v in pb))
        # below ~40 px of plate width there are no characters to read; say so
        # instead of returning noise
        readable = crop.size[0] >= MIN_OCR_WIDTH_PX
        text, conf, ocr_pass, cands = VISION.ocr_text(crop) if readable else (None, None, None, [])
        entry = {"box": [round(v, 1) for v in pbox], "score": round(float(score), 3),
                 "width_px": int(crop.size[0]), "height_px": int(crop.size[1]),
                 "readable": readable,
                 "text": text, "text_conf": conf, "ocr_pass": ocr_pass,
                 "text_candidates": cands,
                 "lprnet_digits": VISION.lpr_digits(crop) if readable else None}
        if want_crops:
            entry["crop_b64"] = b64_jpeg(crop)
        return entry


if __name__ == "__main__":
    if HEF is None:
        sys.exit("hailo_platform is not importable from this python: install HailoRT's Python wheel "
                 "into the venv (see the README, 'The Hailo side')")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    VISION = Vision()
    log(f"serving on http://{BIND_HOST}:{PORT}  chip {chip_temp()} C")
    try:
        ThreadingHTTPServer((BIND_HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
