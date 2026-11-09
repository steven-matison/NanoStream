import os
import time

import numpy as np

from dashcam_walker import HUD_MASKS, mask_hud
from render_stream import prune_cache


def test_mask_hud_blacks_zones_on_a_copy():
    f = np.full((720, 1280, 3), 200, np.uint8)
    out = mask_hud(f, HUD_MASKS["pdr"])
    assert f.min() == 200                               # the picture keeps its overlay
    assert out[700, 640].max() == 0                      # the G and RPM band
    assert out[50, 100].max() == 0                       # the speed readout
    assert out[360, 640].min() == 200                    # the road
    assert mask_hud(f, HUD_MASKS["none"]) is f


def test_prune_cache(tmp_path):
    old, new, part = tmp_path / "old.mp4", tmp_path / "new.mp4", tmp_path / "x.mp4.part"
    for p in (old, new, part):
        p.write_bytes(b"x" * 10)
    t = time.time() - 30 * 3600
    os.utime(old, (t, t))
    logs = []
    prune_cache(str(tmp_path), 24, logs.append)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.mp4"]
    assert logs and "2 stale" in logs[0]
    prune_cache(str(tmp_path / "missing"), 24, logs.append)    # no cache yet: nothing to do
