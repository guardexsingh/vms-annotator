import numpy as np

from app.metadata import detection_message
from app.models import Frame
from app.yolo_detector import YoloPersonDetector


class FakeBox:
    cls = np.array([0])
    conf = np.array([0.9])
    xyxy = np.array([[64, 36, 320, 180]])


class FakePrediction:
    boxes = [FakeBox()]
    speed = {}


class FakeModel:
    def predict(self, *args, **kwargs):
        return [FakePrediction()]


def test_scaled_ai_frame_maps_boxes_back_to_original_source_dimensions():
    detector = YoloPersonDetector("unused.pt", 640, 0.4)
    detector._model = FakeModel()
    detector.state = "ready"
    frame = Frame(np.zeros((360, 640, 3), dtype=np.uint8), 1.0, 7, source_width=1920, source_height=1080)
    result = detector.detect("cam01", frame)
    message = detection_message(result, now=result.completed_at)
    assert message["frame_width"] == 1920 and message["frame_height"] == 1080
    assert message["boxes"][0]["x"] == 0.1
    assert message["boxes"][0]["y"] == 0.1
    assert message["boxes"][0]["width"] == 0.4
    assert message["boxes"][0]["height"] == 0.4
