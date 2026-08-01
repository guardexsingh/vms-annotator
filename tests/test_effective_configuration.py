"""Regression tests for environment-resolved detection configuration wiring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.detection_runtime as runtime_module
from app.config import load_config
from app.detection_runtime import DetectionRuntime, _DetectionSession
from app.health import effective_detection_public
from app.metadata import MetadataHub
from app.metrics import Metrics
from app.models import (
    AppConfig,
    CameraConfig,
    DetectionConfig,
    MetadataConfig,
    OutputConfig,
    PipelineConfig,
    ServiceConfig,
    TrackingConfig,
    VideoConfig,
)


def _yaml(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(
        "cameras:\n"
        "  - {id: renamed-entry, name: Entry, url: rtsp://example/live}\n"
        "detection: {backend: pytorch, target_fps_per_camera: 1}\n"
        "tracking: {prediction_fps: 5}\n"
    )
    return path


def _effective_config() -> AppConfig:
    return AppConfig(
        cameras=(CameraConfig("renamed-entry", "Entry", "rtsp://redacted"),),
        detection=DetectionConfig(
            backend="onnx", backend_source="DETECTION_BACKEND",
            target_fps_per_camera=2.0, yaml_target_fps_per_camera=1.0,
            inference_fps_environment_override=2.0,
            inference_fps_source="YOLO_INFERENCE_FPS",
        ),
        tracking=TrackingConfig(
            prediction_fps=10.0, yaml_prediction_fps=5.0,
            prediction_fps_environment_override=10.0,
            prediction_fps_source="BYTETRACK_PREDICTION_FPS",
        ),
        output=OutputConfig(), video=VideoConfig(), pipeline=PipelineConfig(),
        metadata=MetadataConfig(), service=ServiceConfig(),
    )


def test_process_environment_has_precedence_over_yaml_for_all_detection_targets(tmp_path):
    config = load_config(_yaml(tmp_path), {
        "DETECTION_BACKEND": "onnx",
        "YOLO_INFERENCE_FPS": "2",
        "BYTETRACK_PREDICTION_FPS": "10",
    })
    assert config.detection.backend == "onnx"
    assert config.detection.target_fps_per_camera == 2.0
    assert config.tracking.prediction_fps == 10.0
    assert config.detection.yaml_target_fps_per_camera == 1.0
    assert config.tracking.yaml_prediction_fps == 5.0


def test_yaml_and_built_in_defaults_are_retained_when_environment_is_absent(tmp_path):
    config = load_config(_yaml(tmp_path), {})
    assert config.detection.target_fps_per_camera == 1.0
    assert config.tracking.prediction_fps == 5.0
    assert config.detection.inference_fps_source == "config/cameras.yaml"
    assert config.tracking.prediction_fps_source == "config/cameras.yaml"
    with pytest.raises(ValueError, match="Invalid YOLO_INFERENCE_FPS"):
        load_config(_yaml(tmp_path), {"YOLO_INFERENCE_FPS": "0"})


def test_metrics_and_health_keep_the_same_effective_config_when_detection_is_disabled():
    config = _effective_config()
    metrics = Metrics(config)
    assert metrics.config is config
    metric = metrics.camera("renamed-entry")
    metrics.reset_detection("renamed-entry")
    metrics.set_detector("disabled", active_camera_id=None)
    snapshot = metrics.snapshot()
    assert metric.requested_inference_fps == 2.0
    assert metric.requested_tracker_fps == 10.0
    assert metric.configured_bytetrack_prediction_fps == 10.0
    assert snapshot["detector"] == {
        "state": "disabled", "error": None, "requested_backend": "onnx",
        "backend_source": "DETECTION_BACKEND", "requested_inference_fps": 2.0,
        "requested_yolo_fps": 2.0, "yolo_fps_source": "YOLO_INFERENCE_FPS",
        "requested_ai_capture_fps": 1.0, "ai_capture_fps_source": "built-in default",
        "requested_precision": "auto", "precision_source": "built-in default",
        "configured_bytetrack_prediction_fps": 10.0, "prediction_fps": 10.0,
        "prediction_fps_source": "BYTETRACK_PREDICTION_FPS", "active_camera_id": None,
    }
    assert effective_detection_public(config) == {
        "requested_backend": "onnx", "requested_inference_fps": 2.0,
        "requested_ai_capture_fps": 1.0, "requested_precision": "auto",
        "configured_bytetrack_prediction_fps": 10.0,
    }


def test_detection_session_constructs_both_schedulers_from_the_effective_config(monkeypatch):
    config = _effective_config()
    metrics = Metrics(config)
    metrics.camera("renamed-entry")
    hub = MetadataHub(["renamed-entry"], metrics)
    runtime = DetectionRuntime(config, metrics, hub)
    captured: dict[str, object] = {}

    class FakeBackend:
        name = "onnx"

        def warmup(self):
            return {"execution_provider": "CPUExecutionProvider"}

        def close(self):
            pass

    class FakeCapture:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeInferenceScheduler:
        def __init__(self, _slots, _backend, _store, fps, **_kwargs):
            captured["inference_fps"] = fps
            self.interval = 1 / fps

        def start(self):
            pass

        def stop(self):
            pass

    class FakePredictionScheduler:
        def __init__(self, fps, **_kwargs):
            captured["prediction_fps"] = fps
            self.interval = 1 / fps

        def start(self):
            pass

        def stop(self):
            pass

    class StopAfterSetup:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, *_args):
            self.stopped = True
            return True

        def set(self):
            self.stopped = True

    monkeypatch.setattr(runtime_module, "select_backend", lambda detection: (
        FakeBackend(), SimpleNamespace(requested=detection.backend, selected="onnx", device="cpu",
                                       precision="fp32", execution_provider="CPUExecutionProvider",
                                       fallback_used=False, fallback_reason=None),
    ))
    monkeypatch.setattr(runtime_module, "AICaptureWorker", FakeCapture)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeInferenceScheduler)
    monkeypatch.setattr(runtime_module, "PredictionScheduler", FakePredictionScheduler)
    session = _DetectionSession(runtime, config.cameras[0], 1)
    session.stop_event = StopAfterSetup()
    runtime._session, runtime._active_camera_id, runtime._generation = session, "renamed-entry", 1
    session._run()

    assert captured == {"inference_fps": 2.0, "prediction_fps": 10.0}
    detector = metrics.snapshot()["detector"]
    assert detector["requested_backend"] == detector["active_backend"] == "onnx"
    assert detector["inference_scheduler_interval_ms"] == 500.0
    assert detector["prediction_scheduler_interval_ms"] == 100.0


def test_runtime_start_and_camera_reset_never_restore_legacy_one_or_five_fps_values():
    config = _effective_config()
    metrics = Metrics(config)
    metrics.camera("renamed-entry")
    runtime = DetectionRuntime(config, metrics, MetadataHub(["renamed-entry"], metrics))
    runtime.start()
    camera = metrics.snapshot()["cameras"]["renamed-entry"]
    detector = metrics.snapshot()["detector"]
    assert camera["requested_inference_fps"] == detector["requested_inference_fps"] == 2.0
    assert camera["configured_bytetrack_prediction_fps"] == detector["configured_bytetrack_prediction_fps"] == 10.0
