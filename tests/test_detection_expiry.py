from app.detection_store import DetectionStore
from app.models import DetectionResult


def test_expired_result_is_not_returned_twice_as_newly_stale():
    store = DetectionStore(500)
    result = DetectionResult("cam01", (), 1.0, 1.0, 1.1, 0, 0, 0)
    store.put(result)
    assert store.valid("cam01", 1.59) == (result, False)
    assert store.valid("cam01", 1.61) == (None, True)
    assert store.valid("cam01", 1.62) == (None, False)
