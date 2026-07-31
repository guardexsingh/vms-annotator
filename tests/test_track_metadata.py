from __future__ import annotations

from app.metadata import LatestMetadataMailbox, track_message
from app.models import TrackResult, TrackedPerson


def tracked(completed_at: float = 10.0) -> TrackResult:
    return TrackResult(
        camera_id="office",
        tracks=(
            TrackedPerson(
                "office", 7, (64, 48, 320, 432), 0.91, "active", completed_at - 0.12
            ),
        ),
        source_captured_at=completed_at - 0.3,
        inference_started_at=completed_at - 0.2,
        completed_at=completed_at,
        source_sequence=9,
        frame_width=640,
        frame_height=480,
        active_track_count=1,
        lost_track_count=0,
        removed_track_count=0,
    )


def test_track_metadata_is_normalized_metadata_only():
    message = track_message(tracked(), actual_yolo_fps=1.0, now=10.1)
    assert message["type"] == "tracks"
    assert message["actual_yolo_fps"] == 1.0
    assert message["boxes"] == [{
        "track_id": 7,
        "x": 0.1,
        "y": 0.1,
        "width": 0.4,
        "height": 0.8,
        "confidence": 0.91,
        "class_id": 0,
        "label": "person",
        "source": "yolo",
        "track_state": "active",
        "predicted": False,
        "last_confirmed_at_monotonic_ms": 9880.0,
        "last_confirmed_age_ms": 220.0,
    }]
    encoded = str(message).lower()
    assert "rtsp://" not in encoded
    assert "image" not in encoded
    assert "tensor" not in encoded


def test_slow_client_keeps_only_latest_track_update_for_camera():
    mailbox = LatestMetadataMailbox()
    for sequence in range(100):
        message = track_message(tracked(10 + sequence), 1.0, now=110)
        message["sequence"] = sequence
        mailbox.put("office", message)
    assert mailbox.depth == 1
    assert mailbox.take()["sequence"] == 99


def test_old_prediction_cannot_replace_newer_yolo_correction():
    mailbox = LatestMetadataMailbox()
    correction = track_message(tracked(12.0), 1.0)
    correction["sequence"] = 20
    prediction = track_message(tracked(12.0), 1.0)
    prediction["sequence"] = 19
    prediction["prediction_only"] = True
    assert not mailbox.put("office", correction)
    assert mailbox.put("office", prediction) is False
    assert mailbox.take()["sequence"] == 20
