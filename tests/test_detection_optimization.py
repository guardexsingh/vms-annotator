from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.detection_backend import select_backend
from app.detection_store import DetectionStore
from app.health import health_payload
from app.inference_scheduler import InferenceScheduler
from app.latest_frame import LatestFrame
from app.metrics import Metrics
from app.models import DetectionConfig, DetectionResult, Frame


PROJECT = Path(__file__).resolve().parents[1]


class RecordingBatchDetector:
    retry_after = 0.0

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, Frame]]] = []

    @staticmethod
    def _result(camera_id: str, frame: Frame) -> DetectionResult:
        now = time.monotonic()
        return DetectionResult(camera_id, (), frame.captured_at, now, now, 0, 0, 0,
                               source_sequence=frame.sequence)

    def detect(self, camera_id: str, frame: Frame) -> DetectionResult:
        self.calls.append([(camera_id, frame)])
        return self._result(camera_id, frame)

    def detect_batch(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        self.calls.append(list(frames))
        return [self._result(camera_id, frame) for camera_id, frame in frames]


def _frame(sequence: int, age_seconds: float = 0.0) -> Frame:
    return Frame(np.zeros((2, 2, 3), dtype=np.uint8),
                 time.monotonic() - age_seconds, sequence)


def test_explicit_tensorrt_selection_never_silently_falls_back():
    backend, selection = select_backend(DetectionConfig(backend="tensorrt"))
    assert backend.name == "tensorrt"
    assert selection.selected == "tensorrt"
    assert selection.fallback_used is False


def test_fallback_detail_is_visible_in_health_and_metrics(monkeypatch):
    class ReadyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr("app.health.urllib.request.urlopen", lambda *_a, **_k: ReadyResponse())
    metrics = Metrics()
    metrics.set_detector("ready", selected_backend="pytorch", fallback_used=True,
                         fallback_reason="TensorRT CUDA initialization is unavailable")
    status, payload = health_payload(metrics)
    assert status == 200
    assert payload["detector"]["fallback_used"] is True
    assert "CUDA" in payload["detector"]["fallback_reason"]


def test_batch_uses_one_latest_frame_per_camera_and_maps_results():
    slots = {camera_id: LatestFrame[Frame]() for camera_id in ("renamed-a", "renamed-b")}
    slots["renamed-a"].put(_frame(1))
    slots["renamed-a"].put(_frame(2))
    slots["renamed-b"].put(_frame(8))
    detector = RecordingBatchDetector()
    published = []
    scheduler = InferenceScheduler(
        slots, detector, DetectionStore(750), 50, on_result=lambda camera_id, result:
        published.append((camera_id, result.source_sequence)), batch_mode="batch",
    )
    scheduler.start()
    time.sleep(.08)
    scheduler.stop()
    assert detector.calls
    first = detector.calls[0]
    assert {camera_id for camera_id, _ in first} == {"renamed-a", "renamed-b"}
    assert len({camera_id for camera_id, _ in first}) == len(first)
    assert ("renamed-a", 2) in published
    assert ("renamed-b", 8) in published


def test_offline_and_stale_camera_do_not_block_or_fill_batch():
    slots = {camera_id: LatestFrame[Frame]() for camera_id in ("fresh", "offline", "stale")}
    slots["fresh"].put(_frame(10))
    slots["stale"].put(_frame(20, age_seconds=2))
    detector = RecordingBatchDetector()
    scheduler = InferenceScheduler(
        slots, detector, DetectionStore(750), 50, batch_mode="opportunistic",
        max_batch_wait_ms=5, max_frame_age_ms=750,
    )
    scheduler.start()
    time.sleep(.06)
    scheduler.stop()
    assert detector.calls
    assert [camera_id for camera_id, _ in detector.calls[0]] == ["fresh"]


def test_older_sequence_cannot_replace_newer_detection():
    store = DetectionStore(750)
    now = time.monotonic()
    newer = DetectionResult("renamed", (), now, now, now, 0, 0, 0, source_sequence=2)
    older = DetectionResult("renamed", (), now, now, now, 0, 0, 0, source_sequence=1)
    assert store.put(newer)
    assert not store.put(older)
    assert store.valid("renamed")[0].source_sequence == 2


def test_selected_config_never_changes_frozen_video_or_adds_removed_camera():
    normal = (PROJECT / "config" / "cameras.yaml").read_text().lower()
    assert "mode: direct_hevc" in normal
    assert "allow_transcode_fallback: false" in normal
    assert "inference_workers: 1" in normal
    assert "target_fps_per_camera: 3" in normal
    assert "torch_threads: 1" in normal
    assert "capture_fps: 3" in normal
    roots = [PROJECT / name for name in ("app", "config", "scripts", "web")]
    searchable = "\n".join(
        path.read_text(errors="ignore").lower()
        for root in roots for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    assert "cam" + "04" not in searchable
    start = (PROJECT / "scripts" / "start.sh").read_text()
    assert "nice -n 5 \"$VENV_PYTHON\" -m app.main" in start
    assert "libx264" not in normal
    assert "annotated/" not in normal


def test_experiment_code_has_no_production_vms_control_path():
    production = "/mnt/guardex-nvme/" + "guardex_vms"
    for root_name in ("app", "config", "scripts", "web"):
        for path in (PROJECT / root_name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                assert production not in path.read_text(errors="ignore")


def test_thread_limit_is_reapplied_after_predictor_initialization(monkeypatch):
    import torch

    events = []

    class Model:
        def predict(self, *_args, **_kwargs):
            events.append("predict")
            return []

    detector, _ = select_backend(DetectionConfig(torch_threads=1, torch_interop_threads=1))
    detector._model = Model()
    detector._load = lambda: None
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: events.append(("interop", value)))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: events.append(("threads", value)))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    detector.preflight()
    assert events.index("predict") < events.index(("threads", 1))
