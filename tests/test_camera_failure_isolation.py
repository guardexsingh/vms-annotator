from app.latest_frame import LatestFrame


def test_one_camera_slot_failure_does_not_touch_another_camera_slot():
    failed, healthy = LatestFrame(), LatestFrame()
    failed.put("old")
    healthy.put("fresh")
    failed.take()
    assert healthy.take() == "fresh"
