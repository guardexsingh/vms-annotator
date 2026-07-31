from __future__ import annotations

import threading
import time

from .models import DetectionResult


class DetectionStore:
    def __init__(self, ttl_ms: int) -> None:
        self._ttl_seconds = ttl_ms / 1000.0
        self._values: dict[str, DetectionResult] = {}
        self._stale_marked: set[tuple[str, float]] = set()
        self._lock = threading.Lock()

    def put(self, result: DetectionResult) -> bool:
        with self._lock:
            current = self._values.get(result.camera_id)
            if current is not None and (
                result.source_captured_at < current.source_captured_at
                or (result.source_captured_at == current.source_captured_at
                    and result.source_sequence <= current.source_sequence)
            ):
                return False
            self._values[result.camera_id] = result
            return True

    def valid(self, camera_id: str, now: float | None = None) -> tuple[DetectionResult | None, bool]:
        now = time.monotonic() if now is None else now
        with self._lock:
            result = self._values.get(camera_id)
            if result is None:
                return None, False
            stale = now - result.completed_at > self._ttl_seconds
            if stale:
                key = (camera_id, result.completed_at)
                newly_stale = key not in self._stale_marked
                self._stale_marked.add(key)
                return None, newly_stale
            return result, False

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._values.pop(camera_id, None)
