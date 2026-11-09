import numpy as np
from PIL import Image

import hailo_infer_server as h


def test_nms_keeps_the_best_of_an_overlap():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    assert sorted(h.nms(boxes, [0.9, 0.8, 0.7], 0.5)) == [0, 2]
    assert h.nms([], [], 0.5) == []


def test_letterbox_and_expand():
    arr, scale, px, py = h.letterbox(Image.new("RGB", (1280, 720)), 640)
    assert arr.shape == (640, 640, 3) and scale == 0.5 and px == 0 and py == 140
    assert h.expand([10, 10, 20, 20], 15, 100) == [9.2, 9.2, 15, 20.8]


def test_ocr_text_is_cleaned():
    assert h.Vision._clean("abc 12 東京-9") == "ABC 12 -9"


def test_lpr_charset():
    assert h.LPR_CHARS[-1] == "-" and len(h.LPR_CHARS) == 11
    assert isinstance(np.float32(1), np.floating)
