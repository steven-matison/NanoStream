import json
import os
import time

import numpy as np

from minifi_bar import SLOTS, Clip, MiniFiBar, MiniFiFeed, slot_of


def rec(name, counts):
    return {"name": name, "counts": counts}


def test_clip_at_and_running_totals():
    c = Clip()
    one = [0] * len(SLOTS)
    one[1] = 2
    c.add(8.0, rec("b", one))
    c.add(4.0, rec("a", one))
    r, i, cum = c.at(9.0)
    assert r["name"] == "b" and i == 1 and cum[1] == 4
    r, i, cum = c.at(1.0)                               # nothing behind the playhead: the first one ahead
    assert r["name"] == "a" and i == 0 and cum[1] == 0
    c.add(8.0, rec("b", [0] * len(SLOTS)))              # rewritten under the same name: replaced
    assert len(c.recs) == 2 and c.at(9.0)[2][1] == 2


def test_slot_of():
    assert slot_of(0) == 0 and slot_of(2) == 1 and slot_of(3) == slot_of(1)
    assert slot_of(42) == len(SLOTS) - 1


def make_feed(tmp_path):
    meta, clips = tmp_path / "meta", tmp_path / "clips"
    meta.mkdir()
    logs = []
    return MiniFiFeed(str(meta), str(clips), logs.append), meta, clips, logs


def test_feed_links_and_releases(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    clip = cache / "c1.mp4"
    clip.write_bytes(b"v" * 100)
    feed, _, clips, _ = make_feed(tmp_path)
    feed.feed(str(clip))
    linked = clips / "c1.mp4"
    assert linked.exists() and os.path.samefile(clip, linked)
    assert (clips / ".nanostream-links").read_text() == "c1.mp4\n"
    feed.release(str(clip))
    assert not linked.exists()
    assert (clips / ".nanostream-links").read_text() == ""


def test_orphans_from_an_earlier_run(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (clips / "orphan.mp4").write_bytes(b"o")            # its cache copy is gone: link count 1
    (cache / "kept.mp4").write_bytes(b"k")
    os.link(cache / "kept.mp4", clips / "kept.mp4")     # still shared with the cache
    (clips / "mine.mp4").write_bytes(b"m")              # put there by hand, not in the manifest
    (clips / ".nanostream-links").write_text("orphan.mp4\nkept.mp4\n")
    feed, _, _, logs = make_feed(tmp_path)
    assert sorted(p.name for p in clips.iterdir()) == [".nanostream-links", "kept.mp4", "mine.mp4"]
    assert str(clips / "kept.mp4") in feed.links       # adopted, so it is released after play
    assert any("orphaned" in m for m in logs)


def test_scan_and_draw(tmp_path):
    feed, meta, _, _ = make_feed(tmp_path)
    d = {"video.clip": "c1", "video.t_s": "4.0",
         "detections": json.dumps([{"class_id": 2}, {"class_id": 2}, {"class_id": 0}]),
         "scene": json.dumps([{"class_name": "highway", "confidence": 0.5}])}
    (meta / "c1__000001.json").write_text(json.dumps(d))
    for _ in range(80):                                 # a full walk runs at least every 5 s
        feed._scan()
        v = feed.view("c1", 5.0)
        if v["rec"] is not None:
            break
        time.sleep(0.1)
    assert feed.scored("c1") == 1 and feed.scored("other") == 0
    assert v["rec"]["counts"][1] == 2 and v["rec"]["counts"][0] == 1
    assert v["rec"]["scene"][0] == ("highway", 0.5)
    frame = np.zeros((720, 1280, 3), np.uint8)
    MiniFiBar(feed).draw(frame, "c1", 5.0, 60.0)
    assert frame[-30:].any() and not frame[:600].any()  # drawn along the bottom only


def test_place_holds_every_thread(tmp_path):
    import subprocess
    import time as _t
    script = ("import threading, time\n"
              "for _ in range(3):\n"
              "    threading.Thread(target=time.sleep, args=(30,), daemon=True).start()\n"
              "time.sleep(30)\n")
    p = subprocess.Popen(["python3", "-c", script])
    try:
        _t.sleep(0.5)
        feed, _, _, _ = make_feed(tmp_path)
        feed.agent["pid"] = p.pid
        cores = sorted(os.sched_getaffinity(0))[:1]
        feed.place(cores, 10)
        tids = [int(t) for t in os.listdir(f"/proc/{p.pid}/task")]
        assert len(tids) >= 4
        for tid in tids:
            assert os.sched_getaffinity(tid) == set(cores)
            assert os.getpriority(os.PRIO_PROCESS, tid) >= 10
        feed.place(sorted(os.sched_getaffinity(0)), 5)      # widening the cores never lowers the nice
        assert all(os.getpriority(os.PRIO_PROCESS, t) >= 10 for t in tids)
    finally:
        p.kill()
