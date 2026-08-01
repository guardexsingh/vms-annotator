from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .detection_store import DetectionStore
from .latest_frame import LatestFrame
from .models import Frame

LOG = logging.getLogger(__name__)


class InferenceScheduler:
    """Freshness-first serial or batched scheduler with one slot per camera."""

    def __init__(self, slots: dict[str, LatestFrame[Frame]], detector, store: DetectionStore,
                 fps_per_camera: float, on_result: Callable[[str, object], None] | None = None,
                 on_error: Callable[[str, BaseException], None] | None = None,
                 batch_mode: str = "serial", max_frame_age_ms: int = 750,
                 max_batch_wait_ms: int = 20,
                 on_consumed: Callable[[str, Frame], None] | None = None,
                 on_stale: Callable[[str], None] | None = None,
                 on_compute: Callable[[str, float], None] | None = None) -> None:
        self.slots, self.detector, self.store, self.fps = slots, detector, store, fps_per_camera
        self.interval = 1.0 / fps_per_camera
        self.on_result, self.on_error = on_result, on_error
        self.batch_mode = batch_mode
        self.max_frame_age = max_frame_age_ms / 1000
        self.max_batch_wait = max_batch_wait_ms / 1000
        self.on_consumed, self.on_stale = on_consumed, on_stale
        self.on_compute = on_compute
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._order = list(slots)
        self._next_due: dict[str, float] = {}
        self.completed: dict[str, int] = {camera_id: 0 for camera_id in slots}
        self._detector_blocked_until = 0.0
        self._last_frame: dict[str, tuple[int, float]] = {}

    def start(self) -> None:
        base, interval = time.monotonic(), self.interval
        batched = self.batch_mode in {"batch", "opportunistic"}
        self._next_due = {
            camera_id: base + (0 if batched else index * interval / max(1, len(self._order)))
            for index, camera_id in enumerate(self._order)
        }
        self._thread = threading.Thread(target=self._run, name="inference-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        if self.batch_mode in {"batch", "opportunistic"}:
            self._run_batch()
        else:
            self._run_serial()

    def _accept_frame(self, camera_id: str, frame: Frame, now: float) -> bool:
        identity = (frame.sequence, frame.captured_at)
        if self._last_frame.get(camera_id) == identity:
            return False
        self._last_frame[camera_id] = identity
        if now - frame.captured_at > self.max_frame_age:
            if self.on_stale:
                self.on_stale(camera_id)
            return False
        if self.on_consumed:
            self.on_consumed(camera_id, frame)
        return True

    def _publish(self, camera_id: str, result) -> None:
        if result.completed_at - result.source_captured_at > self.max_frame_age:
            if self.on_stale:
                self.on_stale(camera_id)
            return
        if not self.store.put(result):
            if self.on_stale:
                self.on_stale(camera_id)
            return
        self.completed[camera_id] += 1
        if self.on_result:
            self.on_result(camera_id, result)

    def _inference_failed(self, camera_id: str, error: BaseException, now: float) -> None:
        retry_after = getattr(self.detector, "retry_after", now + 2.0)
        self._detector_blocked_until = max(now + 0.25, retry_after)
        LOG.exception("Inference failed for %s: %s", camera_id, error)
        if self.on_error:
            self.on_error(camera_id, error)

    def _run_serial(self) -> None:
        interval, cursor = self.interval, 0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < self._detector_blocked_until:
                self._stop.wait(min(0.25, self._detector_blocked_until - now))
                continue
            eligible = [camera_id for camera_id in self._order if now >= self._next_due[camera_id]]
            if not eligible:
                wait = max(0.001, min(self._next_due.values()) - now)
                self._stop.wait(min(wait, 0.02))
                continue
            camera_id = next((
                self._order[(cursor + offset) % len(self._order)]
                for offset in range(len(self._order))
                if self._order[(cursor + offset) % len(self._order)] in eligible
            ), eligible[0])
            cursor = (self._order.index(camera_id) + 1) % len(self._order)
            while self._next_due[camera_id] <= now:
                self._next_due[camera_id] += interval
            frame = self.slots[camera_id].take()
            if frame is None or not self._accept_frame(camera_id, frame, time.monotonic()):
                continue
            try:
                cpu_started = time.thread_time()
                detected = self.detector.detect(camera_id, frame)
                if self.on_compute:
                    self.on_compute(camera_id, (time.thread_time() - cpu_started) * 1000)
                self._publish(camera_id, detected)
            except Exception as error:
                self._inference_failed(camera_id, error, now)

    def _run_batch(self) -> None:
        interval = self.interval
        while not self._stop.is_set():
            now = time.monotonic()
            if now < self._detector_blocked_until:
                self._stop.wait(min(.25, self._detector_blocked_until - now))
                continue
            due = [camera_id for camera_id in self._order if now >= self._next_due[camera_id]]
            if not due:
                wait = max(.001, min(self._next_due.values()) - now)
                self._stop.wait(min(.02, wait))
                continue
            for camera_id in due:
                while self._next_due[camera_id] <= now:
                    self._next_due[camera_id] += interval
            deadline, pending = now + self.max_batch_wait, set(due)
            frames: list[tuple[str, Frame]] = []
            while pending and not self._stop.is_set():
                for camera_id in tuple(pending):
                    frame = self.slots[camera_id].take()
                    if frame is not None:
                        pending.remove(camera_id)
                        if self._accept_frame(camera_id, frame, time.monotonic()):
                            frames.append((camera_id, frame))
                if not pending or time.monotonic() >= deadline:
                    break
                self._stop.wait(.002)
            if not frames:
                continue
            try:
                cpu_started = time.thread_time()
                results = self.detector.detect_batch(frames)
                cpu_ms = (time.thread_time() - cpu_started) * 1000
                for result in results:
                    if self.on_compute:
                        self.on_compute(result.camera_id, cpu_ms / max(1, len(frames)))
                    self._publish(result.camera_id, result)
            except Exception as error:
                self._inference_failed(frames[0][0], error, now)
