from __future__ import annotations

import numpy as np
import pytest

from app.config import load_config
from app.detection_backend import select_backend
from app.models import DetectionConfig, Frame
from app.onnx_detector import OnnxPersonDetector


def _config(tmp_path, body=""):
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live}\n" + body)
    return path


def test_environment_backend_and_yolo_fps_override_yaml(tmp_path):
    config = _config(tmp_path, "detection: {backend: pytorch, target_fps_per_camera: 1}\n")
    loaded = load_config(config, {"DETECTION_BACKEND": "onnx", "YOLO_INFERENCE_FPS": "2.5"})
    assert loaded.detection.backend == "onnx"
    assert loaded.detection.target_fps_per_camera == 2.5
    assert loaded.detection.backend_source == "DETECTION_BACKEND"
    assert loaded.detection.inference_fps_source == "YOLO_INFERENCE_FPS"


@pytest.mark.parametrize("value", ("", "0", "-1", "nan", "inf", "abc", "25.1"))
def test_invalid_yolo_fps_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="Invalid YOLO_INFERENCE_FPS"):
        load_config(_config(tmp_path), {"YOLO_INFERENCE_FPS": value})


def test_explicit_onnx_selects_cpu_provider_and_returns_person_structure():
    backend, selection = select_backend(DetectionConfig(backend="onnx"))
    assert selection.selected == "onnx"
    details = backend.warmup()
    assert details["execution_provider"] == "CPUExecutionProvider"
    result = backend.detect("entry", Frame(np.zeros((360, 640, 3), dtype=np.uint8), 1, 1))
    assert result.camera_id == "entry"
    assert all(detection.class_id == 0 for detection in result.detections)


def test_explicit_onnx_uses_the_validated_fp32_model():
    backend, selection = select_backend(DetectionConfig(backend="onnx", precision="fp32"))
    assert selection.selected == "onnx"
    assert selection.precision == backend.precision == "fp32"
    assert backend.model_path.name == "yolo11n_640.onnx"
    tensor, *_ = backend._prepare(np.zeros((360, 640, 3), dtype=np.uint8))
    assert tensor.dtype == np.float32


def test_explicit_onnx_failure_never_selects_pytorch(tmp_path):
    config = DetectionConfig(backend="onnx", onnx_model=str(tmp_path / "missing.onnx"))
    backend, selection = select_backend(config)
    assert isinstance(backend, OnnxPersonDetector) and selection.selected == "onnx"
    with pytest.raises(RuntimeError, match="ONNX model is missing"):
        backend.warmup()


def test_auto_falls_back_to_pytorch_when_tensorrt_warmup_fails(monkeypatch):
    class FakeTRT:
        name = "tensorrt"
        error_summary = "engine warm-up failed"

        def __init__(self, config):
            self.config = config

        def warmup(self):
            raise RuntimeError("engine warm-up failed")

        def close(self):
            return None

    monkeypatch.setattr("app.tensorrt_detector.TensorRTPersonDetector", FakeTRT)
    backend, selection = select_backend(DetectionConfig(backend="auto", allow_backend_fallback=True))
    assert backend.name == selection.selected == "pytorch"
    assert selection.fallback_used and selection.fallback_reason
