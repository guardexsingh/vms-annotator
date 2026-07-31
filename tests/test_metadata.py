from __future__ import annotations

import json
import time

from app.metadata import LatestMetadataMailbox, MetadataHub, detection_message
from app.metrics import Metrics
from app.models import Detection, DetectionResult


def result(*detections: Detection, camera_id: str = "cam01") -> DetectionResult:
    return DetectionResult(camera_id, tuple(detections), 10.0, 10.01, 10.27, 1, 258, 1,
                           source_sequence=1234, frame_width=1920, frame_height=1080)


def test_metadata_coordinates_are_normalized_and_clamped():
    message = detection_message(result(Detection("cam01", (-20, 108, 2100, 1200), 1.2)), now=10.3)
    assert message["sequence"] == 1234
    assert message["boxes"] == [{"x": 0.0, "y": 0.1, "width": 1.0, "height": 0.9,
                                 "confidence": 1.0, "class_id": 0, "label": "person"}]
    assert message["capture_to_result_latency_ms"] == 270


def test_empty_detections_are_valid_metadata_without_frames_or_credentials():
    message = detection_message(result(), now=10.3)
    encoded = json.dumps(message)
    assert message["type"] == "detections" and message["boxes"] == []
    assert "image" not in encoded and "rtsp" not in encoded and "credential" not in encoded


def test_latest_pending_metadata_replaces_older_result():
    mailbox = LatestMetadataMailbox()
    assert not mailbox.put("cam01", {"sequence": 1})
    assert mailbox.put("cam01", {"sequence": 2})
    assert mailbox.depth == 1
    assert mailbox.take()["sequence"] == 2
    assert mailbox.replaced == 1


def test_slow_metadata_client_never_blocks_publisher_or_grows_history():
    metrics = Metrics()
    metrics.camera("cam01")
    hub = MetadataHub(["cam01"], metrics)
    client = hub.add_client()
    hub.subscribe(client, ["cam01"])
    assert client.mailbox.take()["type"] == "active_camera"
    started = time.monotonic()
    for sequence in range(1000):
        item = result(camera_id="cam01")
        object.__setattr__(item, "source_sequence", sequence)
        hub.publish(item)
    assert time.monotonic() - started < 1.0
    assert client.mailbox.depth == 1
    assert client.mailbox.take()["sequence"] == 999


def test_invalid_camera_subscription_is_rejected():
    hub = MetadataHub(["cam01", "cam02"], Metrics())
    client = hub.add_client()
    try:
        hub.subscribe(client, ["cam01", "unknown"])
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("invalid subscription was accepted")


def test_backend_ttl_replaces_boxes_with_stale_status():
    metrics = Metrics()
    metrics.camera("cam01")
    hub = MetadataHub(["cam01"], metrics, ttl_ms=750)
    client = hub.add_client()
    hub.subscribe(client, ["cam01"])
    assert client.mailbox.take()["type"] == "active_camera"
    hub.publish(result())
    assert client.mailbox.take()["type"] == "detections"
    assert hub.expire_stale(now=11.1) == ["cam01"]
    stale = client.mailbox.take()
    assert stale["type"] == "detector_status" and stale["status"] == "stale"
