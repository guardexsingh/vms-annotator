from __future__ import annotations

import threading
import pytest

import app.prediction_scheduler as scheduler_module
from app.prediction_scheduler import PredictionScheduler


def test_prediction_scheduler_targets_five_hz_without_a_backlog():
    ticks: list[float] = []
    done = threading.Event()

    def on_tick(now: float) -> None:
        ticks.append(now)
        if len(ticks) == 5:
            done.set()

    scheduler = PredictionScheduler(5, on_tick)
    scheduler.start()
    assert done.wait(1.5)
    scheduler.stop()
    intervals = [right - left for left, right in zip(ticks, ticks[1:])]
    assert all(.12 <= interval <= .35 for interval in intervals)


def test_late_prediction_tick_is_counted_and_never_caught_up(monkeypatch):
    ticks, skipped = [], []
    scheduler = PredictionScheduler(5, ticks.append, skipped.append)
    clock = iter((0.0, 0.8))  # Due at .2: .4, .6 and .8 are not queued.
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: next(clock))

    def stop_after_one(now: float) -> None:
        ticks.append(now)
        scheduler._stop.set()

    scheduler.on_tick = stop_after_one
    scheduler._run()
    assert ticks == [0.8]
    assert skipped == [3]


@pytest.mark.parametrize(("fps", "interval"), ((5, .2), (10, .1), (2.5, .4), (25, .04)))
def test_scheduler_interval_is_derived_from_effective_fps(fps, interval):
    assert PredictionScheduler(fps, lambda _: None).interval == pytest.approx(interval)
