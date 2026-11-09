from render_stream import TOP_SLOTS, PLATE_SLOT, HailoTally, _top_slot

CAR = TOP_SLOTS.index(next(s for s in TOP_SLOTS if s[0] == "CAR"))


def det(label, box, score=0.9):
    return {"label": label, "box": box, "score": score}


def test_slots():
    assert _top_slot("car") == CAR
    assert _top_slot("motorcycle") == _top_slot("bicycle")
    assert _top_slot("kite") == PLATE_SLOT - 1          # OTHER


def test_a_track_counts_after_three_passes():
    t = HailoTally(confirm=3, ttl=0.8)
    for i in range(2):
        t.update([det("car", [100 + i, 100, 200 + i, 160])], now=i * 0.1)
    on, total, _ = t.view()
    assert on[CAR] == 1 and total[CAR] == 0
    t.update([det("car", [102, 100, 202, 160])], now=0.2)
    t.update([det("car", [103, 100, 203, 160])], now=0.3)
    assert t.view()[1][CAR] == 1                        # counted once, not per pass


def test_a_lost_track_is_a_new_object():
    t = HailoTally(confirm=2, ttl=0.5)
    t.update([det("car", [0, 0, 50, 50])], now=0.0)
    t.update([det("car", [0, 0, 50, 50])], now=0.1)
    t.update([], now=1.0)                               # unseen past the ttl: the track is dropped
    t.update([det("car", [0, 0, 50, 50])], now=1.1)
    t.update([det("car", [0, 0, 50, 50])], now=1.2)
    assert t.view()[1][CAR] == 2


def test_plates_count_distinct_texts():
    t = HailoTally()
    t.update([], plates=[{"text": "ABC123"}, {"text": None}], now=0)
    t.update([], plates=[{"text": "ABC123"}, {"text": "XYZ9"}], now=1)
    on, total, _ = t.view()
    assert on[PLATE_SLOT] == 2 and total[PLATE_SLOT] == 2
    t.update([], plates=None, now=2)                    # a detect-only pass keeps the plate count
    assert t.view()[0][PLATE_SLOT] == 2
