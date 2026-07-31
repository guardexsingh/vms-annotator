from __future__ import annotations

import re
import threading
import time
from typing import Any

import numpy as np

from .models import Detection, DetectionResult, Frame
from .detection_backend import DetectionBackend


class DetectorUnavailable(RuntimeError):
    """The shared detector is unavailable; callers must not spin-retry it."""


def _redact_error(error: BaseException) -> str:
    text = re.sub(r"(?:rtsp|http|https)://[^\s'\"]+", "<redacted-url>", str(error))
    return f"{type(error).__name__}: {text[:240]}"


class YoloPersonDetector(DetectionBackend):
    """One shared model with bounded retry after a model/import failure."""

    name = "pytorch"
    precision = "fp32"

    def __init__(self, model: str, image_size: int, confidence: float,
                 torch_threads: int = 0, torch_interop_threads: int = 1) -> None:
        self.model_path, self.image_size, self.confidence = model, image_size, confidence
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.model_initialization_ms: float | None = None
        self.warmup_ms: float | None = None
        self.device: str | None = None
        self.torch_threads = torch_threads
        self.torch_interop_threads = torch_interop_threads
        self.state = "loading"
        self.error_summary: str | None = None
        self._next_retry = 0.0
        self._failures = 0

    @property
    def retry_after(self) -> float:
        return self._next_retry

    def _mark_failed(self, error: BaseException) -> None:
        self._model = None
        self.state = "failed"
        self.error_summary = _redact_error(error)
        self._failures += 1
        self._next_retry = time.monotonic() + min(60.0, 2.0 ** min(self._failures, 5))

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if self.state == "failed" and time.monotonic() < self._next_retry:
                raise DetectorUnavailable(self.error_summary or "detector is in backoff")
            self.state, self.error_summary = "loading", None
            try:
                started = time.monotonic()
                # Keep this import lazy: relay-only mode must never import Ultralytics.
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
                self.model_initialization_ms = (time.monotonic() - started) * 1000
                try:
                    self.device = str(next(self._model.model.parameters()).device)
                except (AttributeError, StopIteration):
                    self.device = "unknown"
                self.state, self._failures = "ready", 0
            except Exception as error:
                self._mark_failed(error)
                raise DetectorUnavailable(self.error_summary) from error

    def load(self) -> None:
        self._load()

    def preflight(self) -> dict[str, object]:
        """Import the complete stack, load once, and run a synthetic warm-up."""
        try:
            import numpy as preflight_numpy
            import matplotlib
            import torch
            from ultralytics import YOLO  # noqa: F401 - explicit preflight import order

            # NumPy is intentionally imported before Matplotlib and torch.
            versions = {
                "numpy_version": preflight_numpy.__version__, "numpy_file": preflight_numpy.__file__,
                "matplotlib_version": matplotlib.__version__, "matplotlib_file": matplotlib.__file__,
                "torch_version": torch.__version__, "torch_file": torch.__file__,
                "cuda_available": torch.cuda.is_available(),
            }
            if self.torch_interop_threads > 0:
                try:
                    torch.set_num_interop_threads(self.torch_interop_threads)
                except RuntimeError:
                    if torch.get_num_interop_threads() != self.torch_interop_threads:
                        raise
            self._load()
            started = time.monotonic()
            assert self._model is not None
            self._model.predict(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8),
                                imgsz=self.image_size, conf=self.confidence, classes=[0], verbose=False)
            self.warmup_ms = (time.monotonic() - started) * 1000
            if self.torch_threads > 0:
                torch.set_num_threads(self.torch_threads)
        except DetectorUnavailable:
            raise
        except Exception as error:
            with self._lock:
                self._mark_failed(error)
            raise DetectorUnavailable(self.error_summary) from error
        return {**versions, "model_load_ms": self.model_initialization_ms, "warmup_ms": self.warmup_ms,
                "device": self.device, "backend": self.name, "precision": self.precision,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads()}

    def warmup(self) -> dict[str, object]:
        return self.preflight()

    def detect(self, camera_id: str, frame: Frame) -> DetectionResult:
        return self.infer([(camera_id, frame)])[0]

    def infer(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        if not frames:
            return []
        self._load()
        started = time.monotonic()
        try:
            assert self._model is not None
            images = [frame.image for _, frame in frames]
            source = images[0] if len(images) == 1 else images
            predictions = self._model.predict(source, imgsz=self.image_size, conf=self.confidence,
                                              classes=[0], verbose=False)
        except Exception as error:
            with self._lock:
                self._mark_failed(error)
            raise DetectorUnavailable(self.error_summary) from error
        inference_done = time.monotonic()
        completed = time.monotonic()
        results: list[DetectionResult] = []
        for (camera_id, frame), result in zip(frames, predictions):
            detections: list[Detection] = []
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    image_height, image_width = frame.image.shape[:2]
                    source_width = frame.source_width or image_width
                    source_height = frame.source_height or image_height
                    x_scale, y_scale = source_width / image_width, source_height / image_height
                    detections.append(Detection(
                        camera_id, (x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale),
                        float(box.conf[0]), 0,
                    ))
            speeds = getattr(result, "speed", {}) or {}
            height, width = frame.image.shape[:2]
            source_width, source_height = frame.source_width or width, frame.source_height or height
            results.append(DetectionResult(
                camera_id, tuple(detections), frame.captured_at, started, completed,
                float(speeds.get("preprocess", 0.0)),
                (inference_done - started) * 1000 / len(frames),
                float(speeds.get("postprocess", (completed - inference_done) * 1000 / len(frames))),
                frame.sequence, source_width, source_height,
            ))
        return results

    def detect_batch(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        return self.infer(frames)

    def close(self) -> None:
        self._model = None
