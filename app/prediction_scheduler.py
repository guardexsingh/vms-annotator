"""Latest-state, monotonic ByteTrack prediction ticker."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


class PredictionScheduler:
    """Runs one prediction callback at most once per due monotonic tick.

    Late time is counted as skipped ticks and never converted into a burst of
    catch-up callbacks.  It owns no camera or detector; the exclusive runtime
    gives it one selected session only.
    """

    def __init__(self, prediction_fps: float, on_tick: Callable[[float], None],
                 on_skipped: Callable[[int], None] | None = None) -> None:
        self.interval = 1.0 / prediction_fps
        self.on_tick, self.on_skipped = on_tick, on_skipped
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="bytetrack-prediction", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        next_due = time.monotonic() + self.interval
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_due:
                self._stop.wait(min(.05, next_due - now))
                continue
            overdue = int((now - next_due) // self.interval)
            if overdue and self.on_skipped:
                self.on_skipped(overdue)
            # Start the next period from now: do not queue overdue ticks.
            next_due = now + self.interval
            self.on_tick(now)
