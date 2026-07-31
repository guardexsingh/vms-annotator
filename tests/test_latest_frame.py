from app.latest_frame import LatestFrame


def test_waiting_frame_is_replaced_not_queued():
    slot = LatestFrame[int]()
    assert not slot.put(100); assert slot.put(101); assert slot.put(102); assert slot.put(103)
    assert slot.take() == 103
    assert slot.take() is None
    assert slot.replaced == 3


def test_slot_has_no_queue_growth():
    slot = LatestFrame[int]()
    for value in range(10_000): slot.put(value)
    assert slot.take() == 9_999
    assert slot.take() is None
