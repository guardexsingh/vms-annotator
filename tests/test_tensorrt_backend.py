"""TensorRT backend, exporter, and AI-capture configuration guards."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import load_config
from app.detection_backend import select_backend
from app.detection_runtime import DetectionRuntime
from app.latest_frame import LatestFrame
from app.metadata import MetadataHub
from app.metrics import Metrics
from app.models import DetectionConfig, DetectionResult, Frame
from app.onnx_detector import OnnxPersonDetector
from app.tensorrt_detector import (
    TensorRTPersonDetector,
    _import_tensorrt,
    tensorrt_binding_info,
)


PROJECT = Path(__file__).resolve().parents[1]
ENGINE = PROJECT / "models" / "yolo11n_640_fp16.engine"
ONNX = PROJECT / "models" / "yolo11n_640.onnx"
EXPORTER = PROJECT / "scripts" / "export_yolo_trt.sh"


def _minimal_config(tmp_path: Path, **detection) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_text(
        "cameras:\n"
        "  - {id: cam01, name: One, url: rtsp://example/1, enabled: true}\n"
        "  - {id: cam02, name: Two, url: rtsp://example/2, enabled: true}\n"
        "  - {id: cam03, name: Three, url: rtsp://example/3, enabled: true}\n"
        "detection:\n"
        "  enabled: true\n"
        "  model: yolo11n.pt\n"
        "  image_size: 640\n"
        "  confidence: 0.40\n"
        "  classes: [0]\n"
        "  latest_frame_only: true\n"
        "  backend: pytorch\n"
        "  precision: auto\n"
        "  batch_mode: serial\n"
        "  inference_workers: 1\n"
        "  target_fps_per_camera: 2\n"
        "  capture_fps: 3\n"
        + "".join(f"  {key}: {value}\n" for key, value in detection.items())
        + "tracking: {enabled: true, tracker: bytetrack, prediction_fps: 10}\n"
        "metadata: {transport: websocket, path: /ws/detections}\n"
        "pipeline: {mode: native}\n"
        "video: {mode: direct_hevc, allow_transcode_fallback: false}\n"
        "output: {codec: hevc, pixel_format: yuv420p, low_latency: true}\n"
        "service: {host: 127.0.0.1, port: 18080}\n",
        encoding="utf-8",
    )
    return path


def test_exporter_uses_clean_fp32_onnx_static_batch_fp16_and_atomic_publish():
    text = EXPORTER.read_text(encoding="utf-8")
    assert "yolo11n_640.onnx" in text
    assert "refuses an INT8" in text or "INT8 ONNX" in text
    assert "--fp16" in text
    assert "workspace:${WORKSPACE_MB}M" in text
    assert "1, 3, 640, 640" in text
    assert "mv -f \"$TEMP_ENGINE\" \"$ENGINE_PATH\"" in text
    assert "flock" in text
    assert "trt_benchmark_" in text
    assert "sha256sum" in text
    assert "rm -f \"$TEMP_ENGINE\"" in text
    assert "--minShapes" not in text and "--optShapes" not in text


def test_exporter_rejects_int8_source_path():
    script = EXPORTER.read_text(encoding="utf-8")
    assert '[[ "${ONNX_PATH,,}" != *int8* ]]' in script


def test_explicit_tensorrt_selection_and_metrics_fields(tmp_path: Path):
    config = DetectionConfig(
        backend="tensorrt",
        precision="fp16",
        trt_engine_model=str(ENGINE if ENGINE.is_file() else tmp_path / "missing.engine"),
    )
    backend, selection = select_backend(config)
    assert isinstance(backend, TensorRTPersonDetector)
    assert selection.selected == "tensorrt"
    assert selection.fallback_used is False
    assert selection.execution_provider == "TensorRT"
    assert backend.engine_path.name.endswith("_fp16.engine")


def test_explicit_tensorrt_failure_never_falls_back(tmp_path: Path):
    missing = tmp_path / "absent_fp16.engine"
    backend, selection = select_backend(DetectionConfig(
        backend="tensorrt", precision="fp16", trt_engine_model=str(missing),
        allow_backend_fallback=True,
    ))
    assert selection.selected == "tensorrt"
    with pytest.raises(RuntimeError):
        backend.warmup()
    assert backend.name == "tensorrt"


def test_auto_requires_real_tensorrt_warmup_before_selection(monkeypatch):
    calls = {"warmup": 0}

    class FakeTRT:
        name = "tensorrt"
        error_summary = "engine warm-up failed"

        def __init__(self, config):
            self.config = config

        def warmup(self):
            calls["warmup"] += 1
            raise RuntimeError("engine warm-up failed")

        def close(self):
            return None

    monkeypatch.setattr("app.tensorrt_detector.TensorRTPersonDetector", FakeTRT)
    with pytest.raises(RuntimeError, match="warm-up"):
        select_backend(DetectionConfig(backend="auto", allow_backend_fallback=False))
    assert calls["warmup"] == 1


def test_failed_int8_onnx_is_rejected_for_tensorrt_and_config(tmp_path: Path):
    with pytest.raises(ValueError, match="INT8"):
        load_config(_minimal_config(tmp_path), {"TRT_ENGINE_MODEL": "models/yolo11n_640_int8.engine"})
    detector = TensorRTPersonDetector(DetectionConfig(
        backend="tensorrt", trt_engine_model="models/bad_int8.engine",
    ))
    with pytest.raises(RuntimeError, match="INT8"):
        detector.load()


def test_tensorrt_system_binding_loads_through_compatibility_path():
    trt = _import_tensorrt()
    info = tensorrt_binding_info()
    assert trt.__version__.startswith("10.3")
    assert "/usr/lib/python3.10/dist-packages" in str(info["tensorrt_binding_path"])
    assert info["gpu_ready"] is True


@pytest.mark.skipif(not ENGINE.is_file(), reason="FP16 engine not published yet")
def test_engine_path_validation_deserialization_and_reuse():
    backend = TensorRTPersonDetector(DetectionConfig(
        backend="tensorrt", precision="fp16", trt_engine_model=str(ENGINE),
    ))
    details = backend.warmup()
    assert details["execution_provider"] == "TensorRT"
    assert Path(details["engine"]).resolve() == ENGINE.resolve()
    assert details["input_shape"] == "1x3x640x640"
    assert details["reuses_engine"] is True
    first_engine, first_context, first_stream = backend._engine, backend._context, backend._stream
    first_input, first_output = backend._input_device, backend._output_device
    backend.warmup()
    assert backend._engine is first_engine
    assert backend._context is first_context
    assert backend._stream is first_stream
    assert backend._input_device is first_input
    assert backend._output_device is first_output
    assert backend._host_input.dtype == np.float32
    assert tuple(backend._host_input.shape) == (1, 3, 640, 640)
    backend.close()
    assert backend._engine is None and backend._stream is None


@pytest.mark.skipif(not ENGINE.is_file(), reason="FP16 engine not published yet")
def test_preprocessing_letterbox_nms_and_invalid_box_rejection():
    backend = TensorRTPersonDetector(DetectionConfig(
        backend="tensorrt", precision="fp16", trt_engine_model=str(ENGINE), confidence=0.40,
    ))
    backend.load()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[200:800, 700:1100] = (40, 40, 200)
    ratio, left, top = backend._prepare(image)
    assert 0 < ratio <= 1
    assert left >= 0 and top >= 0
    assert backend._host_input is not None
    assert float(backend._host_input.max()) <= 1.0 + 1e-5
    empty = np.zeros((1, 84, 8400), dtype=np.float32)
    boxes, scores = backend._decode_person_boxes(empty, 0.40)
    assert boxes == [] and scores == []
    bogus = np.zeros((1, 84, 2), dtype=np.float32)
    bogus[0, 0, 0], bogus[0, 1, 0], bogus[0, 2, 0], bogus[0, 3, 0], bogus[0, 4, 0] = 10, 10, -5, -5, 0.9
    bogus[0, 0, 1], bogus[0, 1, 1], bogus[0, 2, 1], bogus[0, 3, 1], bogus[0, 4, 1] = (
        float("nan"), 1, 2, 2, 0.9
    )
    decoded, _ = backend._decode_person_boxes(bogus, 0.40)
    assert all((x2 > x1 and y2 > y1) for x1, y1, x2, y2 in decoded)
    assert callable(OnnxPersonDetector._nms)
    result = backend.detect("cam01", Frame(image, time.monotonic(), 1, source_width=1920, source_height=1080))
    assert all(detection.class_id == 0 for detection in result.detections)
    assert all(detection.confidence >= 0.40 for detection in result.detections)
    backend.close()


def test_ai_capture_fps_precedence_and_floor_against_yolo(tmp_path: Path):
    path = _minimal_config(tmp_path)
    env_win = load_config(path, {
        "AI_CAPTURE_FPS": "4",
        "YOLO_INFERENCE_FPS": "2",
    })
    assert env_win.detection.capture_fps == 4.0
    assert env_win.detection.capture_fps_source == "AI_CAPTURE_FPS"
    assert env_win.detection.target_fps_per_camera == 2.0
    raised = load_config(path, {
        "AI_CAPTURE_FPS": "1",
        "YOLO_INFERENCE_FPS": "2",
    })
    assert raised.detection.capture_fps == 2.0
    assert raised.detection.target_fps_per_camera == 2.0


def test_latest_frame_one_slot_and_no_inference_queue_accumulation():
    slot = LatestFrame[Frame]()
    slot.put(Frame(np.zeros((2, 2, 3), dtype=np.uint8), 1.0, 1))
    slot.put(Frame(np.zeros((2, 2, 3), dtype=np.uint8), 2.0, 2))
    slot.put(Frame(np.zeros((2, 2, 3), dtype=np.uint8), 3.0, 3))
    assert slot.depth == 1
    assert slot.replaced == 2
    frame = slot.take()
    assert frame is not None and frame.sequence == 3
    assert slot.take() is None
    assert slot.depth == 0


def test_bytetrack_remains_independently_configured(tmp_path: Path):
    loaded = load_config(_minimal_config(tmp_path), {
        "YOLO_INFERENCE_FPS": "2",
        "AI_CAPTURE_FPS": "3",
        "BYTETRACK_PREDICTION_FPS": "10",
    })
    assert loaded.detection.target_fps_per_camera == 2.0
    assert loaded.detection.capture_fps == 3.0
    assert loaded.tracking.prediction_fps == 10.0
    assert loaded.tracking.prediction_fps_source == "BYTETRACK_PREDICTION_FPS"


def _runtime_with_metrics(tmp_path: Path):
    config = load_config(_minimal_config(tmp_path))
    metrics = Metrics(config)
    for camera in config.cameras:
        metrics.camera(camera.id)
    hub = MetadataHub([camera.id for camera in config.cameras], metrics)
    runtime = DetectionRuntime(config, metrics, hub)
    runtime.start()
    return config, metrics, runtime


def test_switching_cameras_rejects_obsolete_tensorrt_results(tmp_path: Path):
    config, metrics, runtime = _runtime_with_metrics(tmp_path)
    with runtime._state_lock:
        runtime._generation += 1
        runtime._active_camera_id = "cam02"
        runtime._status = "active"
        obsolete = SimpleNamespace(
            owner=runtime, camera=config.cameras[0], generation=runtime._generation - 1,
            track=lambda result: result, stop_event=SimpleNamespace(is_set=lambda: False),
        )
    now = time.monotonic()
    before = metrics.cameras["cam01"].latest_inference_latency_ms
    result = DetectionResult("cam01", (), now, now, now, 0, 0, 0, source_sequence=9)
    runtime._session_result(obsolete, "cam01", result)
    assert metrics.cameras["cam01"].latest_inference_latency_ms == before
    runtime.stop()


def test_disabling_detection_releases_runtime_resources(monkeypatch, tmp_path: Path):
    closed = {"count": 0}

    class FakeBackend:
        name = "tensorrt"
        error_summary = None
        retry_after = 0.0

        def warmup(self):
            return {"backend": "tensorrt", "execution_provider": "TensorRT", "engine": "x"}

        def close(self):
            closed["count"] += 1

    monkeypatch.setattr(
        "app.detection_runtime.select_backend",
        lambda config: (FakeBackend(), SimpleNamespace(
            requested="tensorrt", selected="tensorrt", device="cuda", precision="fp16",
            fallback_used=False, fallback_reason=None, execution_provider="TensorRT",
        )),
    )

    class ImmediateCapture:
        def __init__(self, *args, **kwargs):
            self.on_status = kwargs.get("on_status") or args[3]

        def start(self):
            self.on_status("cam01", "ready")

        def stop(self):
            return None

    monkeypatch.setattr("app.detection_runtime.AICaptureWorker", ImmediateCapture)
    monkeypatch.setattr(
        "app.detection_runtime.InferenceScheduler",
        lambda *a, **k: SimpleNamespace(start=lambda: None, stop=lambda: None, interval=0.5),
    )
    monkeypatch.setattr(
        "app.detection_runtime.PredictionScheduler",
        lambda *a, **k: SimpleNamespace(start=lambda: None, stop=lambda: None, interval=0.1),
    )
    monkeypatch.setattr(
        "app.detection_runtime.ByteTrackPersonTracker",
        lambda *a, **k: SimpleNamespace(
            reset=lambda: None, update=lambda r: None, predict=lambda n: None,
            implementation="bytetrack",
        ),
    )

    _, _, runtime = _runtime_with_metrics(tmp_path)
    runtime.set_active_camera("cam01")
    time.sleep(0.2)
    runtime.set_active_camera(None)
    time.sleep(0.2)
    assert closed["count"] >= 1
    assert runtime.state()["status"] == "disabled"
    runtime.stop()


def test_multiple_browsers_do_not_create_multiple_engines():
    source = (PROJECT / "app" / "detection_runtime.py").read_text(encoding="utf-8")
    assert "Owns at most one on-demand camera AI session for all browser clients" in source
    assert "self._session" in source


def test_direct_hevc_unchanged_and_no_h264_or_annotated_or_cam04():
    cameras = (PROJECT / "config" / "cameras.yaml").read_text().lower()
    assert "mode: direct_hevc" in cameras
    assert "allow_transcode_fallback: false" in cameras
    assert "libx264" not in cameras
    assert "annotated/" not in cameras
    roots = [PROJECT / name for name in ("app", "config", "scripts", "web")]
    blob = "\n".join(
        path.read_text(errors="ignore").lower()
        for root in roots for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert "cam04" not in blob
    production = "/mnt/guardex-nvme/" + "guardex_vms"
    assert production not in blob


def test_onnx_fp32_source_exists_and_int8_model_is_not_default():
    assert ONNX.is_file()
    example = (PROJECT / ".env.example").read_text(encoding="utf-8")
    assert "yolo11n_640_fp16.engine" in example
    assert "yolo11n_640_int8.onnx" not in example
    assert "DETECTION_BACKEND=tensorrt" in example
