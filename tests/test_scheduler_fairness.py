import time

import numpy as np

from app.detection_store import DetectionStore
from app.inference_scheduler import InferenceScheduler
from app.latest_frame import LatestFrame
from app.models import DetectionResult, Frame


class FakeDetector:
    def detect(self, camera_id, frame):
        now = time.monotonic()
        return DetectionResult(camera_id, (), frame.captured_at, now, now, 0, 0, 0)


def test_round_robin_gives_each_camera_a_turn():
    slots = {key: LatestFrame() for key in ("a", "b", "c", "d")}
    for slot in slots.values(): slot.put(Frame(np.zeros((1, 1, 3), dtype=np.uint8), time.monotonic(), 1))
    scheduler = InferenceScheduler(slots, FakeDetector(), DetectionStore(500), 20)
    scheduler.start(); time.sleep(.25); scheduler.stop()
    assert all(scheduler.completed[key] >= 1 for key in slots)
