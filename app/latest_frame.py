from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestFrame(Generic[T]):
    """A one-item replaceable mailbox. It can never grow into a frame queue."""
    def __init__(self) -> None:
        self._value: T | None = None
        self._replaced = 0
        self._lock = threading.Lock()

    def put(self, value: T) -> bool:
        """Store value, replacing an unconsumed one. Returns whether replacement occurred."""
        with self._lock:
            replaced = self._value is not None
            if replaced:
                self._replaced += 1
            self._value = value
            return replaced

    def take(self) -> T | None:
        with self._lock:
            value, self._value = self._value, None
            return value

    def peek(self) -> T | None:
        with self._lock:
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None

    @property
    def replaced(self) -> int:
        with self._lock:
            return self._replaced

    @property
    def depth(self) -> int:
        with self._lock:
            return int(self._value is not None)
