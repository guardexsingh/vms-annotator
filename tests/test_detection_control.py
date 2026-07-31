from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.detection_runtime as runtime_module
from app.detection_runtime import DetectionRuntime
from app.health import set_active_camera_from_payload
from app.metadata import MetadataHub
from app.metrics import Metrics
from app.models import (
    AppConfig,
    CameraConfig,
    Detection,
    DetectionConfig,
    DetectionResult,
    MetadataConfig,
    OutputConfig,
    PipelineConfig,
    ServiceConfig,
    TrackingConfig,
    VideoConfig,
)

PROJECT = Path(__file__).parents[1]


def app_config() -> AppConfig:
    cameras = (
        CameraConfig("lobby", "Lobby", "rtsp://redacted", detection_enabled=True),
        CameraConfig("office", "Office", "rtsp://redacted", detection_enabled=True),
        CameraConfig("disabled", "Disabled", "", enabled=False, detection_enabled=True),
        CameraConfig("video-only", "Video", "rtsp://redacted", detection_enabled=False),
    )
    return AppConfig(
        cameras=cameras,
        detection=DetectionConfig(
            target_fps_per_camera=1,
            capture_fps=1,
            inference_workers=1,
            batch_mode="serial",
        ),
        tracking=TrackingConfig(),
        output=OutputConfig(),
        video=VideoConfig(),
        pipeline=PipelineConfig(),
        metadata=MetadataConfig(),
        service=ServiceConfig(),
    )


def runtime(monkeypatch):
    sessions = []

    class FakeSession:
        def __init__(self, owner, camera, generation):
            self.owner, self.camera, self.generation = owner, camera, generation
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True
            sessions.append(self)

        def stop(self):
            self.stopped = True

        def track(self, result):
            raise AssertionError("obsolete inference must not reach tracking")

    monkeypatch.setattr(runtime_module, "_DetectionSession", FakeSession)
    config = app_config()
    metrics = Metrics()
    for camera in config.cameras:
        metrics.camera(camera.id)
    hub = MetadataHub((camera.id for camera in config.cameras if camera.enabled), metrics, 2000)
    controller = DetectionRuntime(config, metrics, hub)
    return controller, metrics, hub, sessions


def test_no_camera_or_ai_session_exists_at_startup(monkeypatch):
    controller, metrics, _, sessions = runtime(monkeypatch)
    controller.start()
    assert controller.state() == {"camera_id": None, "status": "disabled"}
    assert sessions == []
    assert metrics.snapshot()["detector"]["state"] == "disabled"
    assert all(
        metrics.camera(camera.id).ai_capture_status == "disabled"
        for camera in controller.cameras
    )


def test_one_active_camera_idempotence_switch_and_disable(monkeypatch):
    controller, _, _, sessions = runtime(monkeypatch)
    controller.start()
    assert controller.set_active_camera("lobby")["camera_id"] == "lobby"
    assert len(sessions) == 1 and sessions[0].started
    assert controller.set_active_camera("lobby")["camera_id"] == "lobby"
    assert len(sessions) == 1

    assert controller.set_active_camera("office")["camera_id"] == "office"
    assert sessions[0].stopped
    assert len(sessions) == 2 and sessions[1].started

    assert controller.set_active_camera(None) == {
        "camera_id": None,
        "status": "disabled",
    }
    assert sessions[1].stopped


def test_only_detection_capable_enabled_camera_ids_are_accepted(monkeypatch):
    controller, _, _, _ = runtime(monkeypatch)
    controller.start()
    for camera_id in ("missing", "disabled", "video-only"):
        with pytest.raises(ValueError, match="not enabled"):
            controller.set_active_camera(camera_id)


def test_api_validation_and_identical_requests_share_runtime(monkeypatch):
    controller, _, _, sessions = runtime(monkeypatch)
    controller.start()
    first = set_active_camera_from_payload(controller, {"camera_id": "office"})
    second = set_active_camera_from_payload(controller, {"camera_id": "office"})
    assert first["camera_id"] == second["camera_id"] == "office"
    assert len(sessions) == 1
    for invalid in ({}, {"camera_id": ""}, {"camera_id": 2}, {"camera_id": None, "extra": 1}):
        with pytest.raises(ValueError):
            set_active_camera_from_payload(controller, invalid)


def test_switch_clears_old_browser_state_and_rejects_inflight_result(monkeypatch):
    controller, metrics, hub, sessions = runtime(monkeypatch)
    client = hub.add_client()
    hub.subscribe(client, ["lobby", "office"])
    client.mailbox.take()
    controller.start()
    client.mailbox.take()
    controller.set_active_camera("lobby")
    old = sessions[0]
    while client.mailbox.take(timeout=0.01):
        pass
    controller.set_active_camera("office")
    messages = []
    while (message := client.mailbox.take(timeout=0.01)) is not None:
        messages.append(message)
    assert any(
        message["type"] == "clear_tracks" and message["camera_id"] == "lobby"
        for message in messages
    )
    result = DetectionResult(
        "lobby",
        (Detection("lobby", (10, 10, 100, 200), 0.9),),
        time.monotonic(),
        time.monotonic(),
        time.monotonic(),
        1,
        1,
        1,
        1,
        640,
        480,
    )
    controller._session_result(old, "lobby", result)
    assert metrics.camera("lobby").completed_inferences == 0


def test_frontend_toggle_does_not_restart_video_path():
    source = (PROJECT / "web" / "app.js").read_text()
    toggle_body = source.split("async function toggleDetection", 1)[1].split(
        "function connectMetadata", 1
    )[0]
    assert "/api/detection/active-camera" in toggle_body
    assert "players" not in toggle_body
    assert ".connect(" not in toggle_body
    assert "srcObject" not in toggle_body
    assert "location.reload" not in toggle_body


def test_runtime_contains_no_hybrid_or_visual_tracker():
    source = "\n".join(
        (PROJECT / relative).read_text().lower()
        for relative in (
            "app/detection_runtime.py",
            "app/bytetrack_tracker.py",
            "app/metadata.py",
        )
    )
    assert "optical flow" not in source
    assert "csrt" not in source
    assert "mosse" not in source
    assert "medianflow" not in source
    assert "kcf" not in source
    assert "annotated/" not in source
    assert "h264" not in source


def test_backend_readiness_details_can_repeat_selected_fields(monkeypatch):
    controller, metrics, _, sessions = runtime(monkeypatch)
    controller.start()
    controller.set_active_camera("lobby")
    controller._backend_ready(
        sessions[0],
        {"device": "reported-device", "backend": "pytorch", "precision": "fp32"},
        SimpleNamespace(
            requested="auto",
            selected="pytorch",
            device="cpu",
            precision="fp32",
            fallback_used=False,
            fallback_reason=None,
        ),
    )
    detector = metrics.snapshot()["detector"]
    assert detector["state"] == "ready"
    assert detector["device"] == "cpu"


def test_starting_active_and_safe_error_are_published(monkeypatch):
    controller, _, hub, sessions = runtime(monkeypatch)
    client = hub.add_client()
    hub.subscribe(client, ["lobby"])
    client.mailbox.take()
    controller.start()
    client.mailbox.take()
    assert controller.set_active_camera("lobby") == {
        "camera_id": "lobby", "status": "starting"
    }
    starting = [client.mailbox.take(), client.mailbox.take()]
    assert {message["type"] for message in starting} == {"active_camera", "detector_status"}
    assert all(message["status"] == "starting" for message in starting)
    controller._session_failed(sessions[0], "rtsp://secret@example.invalid must not reach browser")
    messages = [client.mailbox.take(), client.mailbox.take()]
    error = next(message for message in messages if message["type"] == "detector_status")
    assert error["status"] == "error"
    assert error["message"] == "Detector could not start or process frames"
    assert "secret" not in str(error)
