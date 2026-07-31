from __future__ import annotations

import time

import cv2

from .detection_store import DetectionStore
from .models import Frame


class Compositor:
    def __init__(self, store: DetectionStore) -> None:
        self.store = store

    def compose(self, camera_id: str, frame: Frame) -> tuple[object, int, float | None, bool]:
        image = frame.image.copy()
        result, newly_stale = self.store.valid(camera_id)
        if result is None:
            return image, 0, None, newly_stale
        age_ms = (time.monotonic() - result.completed_at) * 1000
        for detection in result.detections:
            x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(image, f"person {detection.confidence:.2f}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
        return image, len(result.detections), age_ms, newly_stale
