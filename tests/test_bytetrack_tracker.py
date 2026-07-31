from __future__ import annotations

import pytest

from app.bytetrack_tracker import ByteTrackPersonTracker
from app.config import load_config
from app.models import Detection, DetectionResult, TrackingConfig


def result(completed_at: float, boxes=(), class_id: int = 0) -> DetectionResult:
    return DetectionResult(
        "office",
        tuple(Detection("office", box, 0.9, class_id) for box in boxes),
        completed_at - 0.2,
        completed_at - 0.1,
        completed_at,
        5,
        80,
        5,
        int(completed_at),
        640,
        480,
    )


def test_real_ultralytics_bytetrack_preserves_matching_id():
    tracker = ByteTrackPersonTracker("office", TrackingConfig())
    first = tracker.update(result(1, ((10, 10, 100, 200),)))
    second = tracker.update(result(2, ((12, 10, 102, 200),)))
    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].track_state == "active"
    assert tracker.implementation == "ultralytics-8.3.0-bytetrack-dt-adapter"


def test_one_missed_update_holds_lost_track_then_wall_clock_removes_it():
    tracker = ByteTrackPersonTracker(
        "office", TrackingConfig(track_buffer=2, hold_box_ms=1500, remove_track_ms=2000)
    )
    active = tracker.update(result(1, ((10, 10, 100, 200),)))
    missed = tracker.update(result(2, ()))
    removed = tracker.update(result(3, ()))
    assert missed.tracks[0].track_id == active.tracks[0].track_id
    assert missed.tracks[0].track_state == "lost"
    assert removed.tracks == ()
    assert removed.removed_track_count == 1


def test_bytetrack_rejects_every_non_person_detection():
    tracker = ByteTrackPersonTracker("office", TrackingConfig())
    with pytest.raises(ValueError, match="person class 0"):
        tracker.update(result(1, ((1, 2, 3, 4),), class_id=2))


def test_tracker_reset_discards_ids_and_state():
    tracker = ByteTrackPersonTracker("office", TrackingConfig())
    tracker.update(result(1, ((10, 10, 100, 200),)))
    tracker.reset()
    assert tracker.update(result(2, ())).tracks == ()


def test_prediction_only_preserves_id_confidence_and_never_creates_track():
    tracker = ByteTrackPersonTracker("office", TrackingConfig())
    assert tracker.predict(0.2) is None
    correction = tracker.update(result(1.0, ((10, 10, 100, 200),)))
    predicted = tracker.predict(1.2)
    assert predicted is not None and predicted.is_prediction
    assert predicted.tracks[0].track_id == correction.tracks[0].track_id
    assert predicted.tracks[0].confidence == correction.tracks[0].confidence
    assert predicted.tracks[0].source == "bytetrack_prediction"
    assert predicted.tracks[0].track_state == "predicted"
    assert predicted.tracks[0].predicted is True
    assert predicted.predicted_track_count == 1


def test_five_short_prediction_steps_are_not_five_one_second_steps():
    config = TrackingConfig(prediction_deadzone_norm=0, max_prediction_displacement_norm_per_second=1)
    short = ByteTrackPersonTracker("office", config)
    long = ByteTrackPersonTracker("office", config)
    first = result(1.0, ((10, 10, 100, 200),))
    second = result(2.0, ((30, 10, 120, 200),))
    short.update(first)
    long.update(first)
    # A second correction establishes a one-second velocity estimate.
    short.update(second)
    long.update(second)
    for instant in (2.2, 2.4, 2.6, 2.8, 3.0):
        short.predict(instant)
    long.predict(3.0)
    short_box = short.predict(3.0).tracks[0].xyxy
    long_box = long.predict(3.0).tracks[0].xyxy
    assert short_box[0] == pytest.approx(long_box[0], abs=1.0)
    # A five-second accidental transition would have crossed almost the frame.
    assert short_box[0] < 200


def test_equivalent_elapsed_time_has_equivalent_prediction_at_multiple_rates():
    def position(prediction_fps: float) -> float:
        tracker = ByteTrackPersonTracker(
            "office", TrackingConfig(prediction_fps=prediction_fps,
                                      prediction_deadzone_norm=0,
                                      max_prediction_displacement_norm_per_second=1)
        )
        tracker.update(result(1.0, ((10, 10, 100, 200),)))
        tracker.update(result(2.0, ((30, 10, 120, 200),)))
        for tick in range(1, round(prediction_fps) + 1):
            tracker.predict(2.0 + tick / prediction_fps)
        return tracker.predict(3.0).tracks[0].xyxy[0]

    positions = [position(rate) for rate in (2, 5, 10)]
    assert max(positions) - min(positions) < 1.0


def test_prediction_stops_display_then_removes_backend_state():
    tracker = ByteTrackPersonTracker(
        "office", TrackingConfig(hold_box_ms=1500, remove_track_ms=2000)
    )
    tracker.update(result(1.0, ((10, 10, 100, 200),)))
    assert tracker.predict(2.49).tracks
    assert tracker.predict(2.51).tracks == ()
    removed = tracker.predict(3.01)
    assert removed.tracks == () and removed.removed_track_count == 1


def test_fresh_yolo_correction_replaces_prediction_state():
    tracker = ByteTrackPersonTracker("office", TrackingConfig())
    first = tracker.update(result(1.0, ((10, 10, 100, 200),)))
    predicted = tracker.predict(1.2)
    corrected = tracker.update(result(2.0, ((30, 10, 120, 200),)))
    assert predicted.tracks[0].predicted
    assert corrected.tracks[0].track_id == first.tracks[0].track_id
    assert corrected.tracks[0].source == "yolo"
    assert corrected.tracks[0].predicted is False


def test_large_default_track_buffer_is_rejected(tmp_path):
    config = tmp_path / "large-buffer.yaml"
    config.write_text(
        "cameras:\n  - {id: office, name: Office, url: rtsp://example}\n"
        "tracking: {tracker: bytetrack, track_buffer: 30}\n"
    )
    with pytest.raises(ValueError, match="one and five"):
        load_config(config)
