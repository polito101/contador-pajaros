"""v0.4.0 appearance-confirmation events — the line-free feeder contract.

One event per confirmed track, emitted at the exact add() where the track
reaches min_frames ("the model is sure"), carrying video-time `at` and the
confirmation frame's normalized centroid when the frame size is known.
len(events) == total() by construction.
"""
from __future__ import annotations

from contador_pajaros.contar import BIRD_CLASSES, BirdCounter


def test_event_emitted_at_confirmation_frame_only():
    c = BirdCounter(min_frames=3, fps=10.0)
    c.add(track_id=1, class_id=14, frame_index=4)
    c.add(track_id=1, class_id=14, frame_index=5)
    assert c.events == []          # not confirmed yet
    c.add(track_id=1, class_id=14, frame_index=6)
    assert len(c.events) == 1      # confirmation moment
    assert c.events[0]["at"] == 0.6  # frame 6 / 10 fps
    assert c.events[0]["class"] == "bird"
    assert c.events[0]["track_id"] == 1
    c.add(track_id=1, class_id=14, frame_index=7)
    assert len(c.events) == 1      # never a second event for the same track


def test_events_reconcile_with_total_and_unconfirmed_emit_nothing():
    c = BirdCounter(min_frames=5, fps=25.0)
    for f in range(5):
        c.add(track_id=1, class_id=14, frame_index=f)      # confirmed
    for f in range(3):
        c.add(track_id=2, class_id=14, frame_index=f)      # below threshold
    assert c.total() == 1
    assert len(c.events) == c.total()


def test_squirrel_proxies_share_one_bucket_one_event():
    """A class-flickering squirrel track (cat->dog->bear) accumulates in the
    single 'squirrel' bucket and emits exactly ONE event."""
    c = BirdCounter(min_frames=3, fps=10.0)
    c.add(track_id=7, class_id=15, frame_index=0)  # cat
    c.add(track_id=7, class_id=16, frame_index=1)  # dog
    c.add(track_id=7, class_id=21, frame_index=2)  # bear -> 3rd frame, confirm
    assert c.total() == 1
    assert c.breakdown() == {"bird": 0, "squirrel": 1}
    assert len(c.events) == 1
    assert c.events[0]["class"] == "squirrel"


def test_event_carries_normalized_xy_when_frame_size_known():
    c = BirdCounter(min_frames=2, fps=10.0, frame_size=(640, 360))
    c.add(track_id=1, class_id=14, cx=320, cy=90, frame_index=0)
    c.add(track_id=1, class_id=14, cx=322, cy=92, frame_index=1)
    (ev,) = c.events
    # the CONFIRMATION frame's centroid is the recorded point
    assert ev["x"] == round(322 / 640, 4)
    assert ev["y"] == round(92 / 360, 4)


def test_event_omits_xy_without_frame_size_or_centroid():
    c = BirdCounter(min_frames=1, fps=10.0)
    c.add(track_id=1, class_id=14, frame_index=0)
    (ev,) = c.events
    assert "x" not in ev and "y" not in ev
    c2 = BirdCounter(min_frames=1, fps=10.0, frame_size=(640, 360))
    c2.add(track_id=2, class_id=14, frame_index=0)  # no centroid passed
    assert "x" not in c2.events[0] and "y" not in c2.events[0]


def test_class_table_squirrel_proxies():
    assert BIRD_CLASSES[14] == "bird"
    assert BIRD_CLASSES[15] == BIRD_CLASSES[16] == BIRD_CLASSES[21] == "squirrel"


def test_cross_label_flicker_counts_once():
    """One track confirmed as squirrel must NOT confirm again under bird:
    one animal, one event, total()==1 (review find, 2026-06-12)."""
    c = BirdCounter(min_frames=3, fps=10.0)
    for f in range(3):
        c.add(track_id=7, class_id=15, frame_index=f)   # squirrel confirms
    for f in range(3, 9):
        c.add(track_id=7, class_id=14, frame_index=f)   # stray bird misfires
    assert c.total() == 1
    assert c.breakdown() == {"bird": 0, "squirrel": 1}
    assert len(c.events) == 1
    assert c.events[0]["class"] == "squirrel"


def test_min_frames_clamped_to_one():
    c = BirdCounter(min_frames=0, fps=10.0)
    c.add(track_id=1, class_id=14, frame_index=0)
    assert c.total() == 1 and len(c.events) == 1
