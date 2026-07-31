"""Normalized, latest-only detection metadata fan-out."""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from typing import Any

from .metrics import Metrics
from .models import DetectionResult, TrackResult


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def detection_message(result: DetectionResult, now: float | None = None) -> dict[str, Any]:
    """Convert original-frame pixel boxes to clamped normalized metadata."""
    now = time.monotonic() if now is None else now
    wall_clock_offset_ms = time.time() * 1000 - time.monotonic() * 1000
    width, height = max(1, result.frame_width), max(1, result.frame_height)
    boxes = []
    for detection in result.detections:
        x1, y1, x2, y2 = detection.xyxy
        left, top = _clamp(x1 / width), _clamp(y1 / height)
        right, bottom = _clamp(x2 / width), _clamp(y2 / height)
        boxes.append({"x": left, "y": top, "width": _clamp(right - left), "height": _clamp(bottom - top),
                      "confidence": _clamp(detection.confidence), "class_id": 0, "label": "person"})
    return {
        "type": "detections", "camera_id": result.camera_id, "sequence": result.source_sequence,
        "frame_width": result.frame_width, "frame_height": result.frame_height,
        "captured_at_monotonic_ms": round(result.source_captured_at * 1000, 3),
        "inference_started_at_monotonic_ms": round(result.inference_started_at * 1000, 3),
        "completed_at_monotonic_ms": round(result.completed_at * 1000, 3),
        "completed_at_unix_ms": round(result.completed_at * 1000 + wall_clock_offset_ms, 3),
        "websocket_sent_at_monotonic_ms": round(now * 1000, 3),
        "websocket_sent_at_unix_ms": round(now * 1000 + wall_clock_offset_ms, 3),
        "capture_to_inference_start_ms": round(
            (result.inference_started_at - result.source_captured_at) * 1000, 3
        ),
        "inference_latency_ms": round((result.completed_at - result.inference_started_at) * 1000, 3),
        "capture_to_result_latency_ms": round((result.completed_at - result.source_captured_at) * 1000, 3),
        "metadata_age_ms": round(max(0.0, now - result.completed_at) * 1000, 3),
        "boxes": boxes,
    }


def track_message(result: TrackResult, actual_yolo_fps: float,
                  actual_tracker_fps: float = 0.0,
                  configured_bytetrack_prediction_fps: float = 5.0,
                  now: float | None = None) -> dict[str, Any]:
    """Convert ByteTrack output to normalized, metadata-only boxes."""
    now = time.monotonic() if now is None else now
    wall_clock_offset_ms = time.time() * 1000 - time.monotonic() * 1000
    width, height = max(1, result.frame_width), max(1, result.frame_height)
    boxes = []
    for track in result.tracks:
        x1, y1, x2, y2 = track.xyxy
        left, top = _clamp(x1 / width), _clamp(y1 / height)
        right, bottom = _clamp(x2 / width), _clamp(y2 / height)
        boxes.append({
            "track_id": track.track_id,
            "x": left,
            "y": top,
            "width": _clamp(right - left),
            "height": _clamp(bottom - top),
            "confidence": _clamp(track.confidence),
            "class_id": 0,
            "label": "person",
            "source": track.source,
            "track_state": track.track_state,
            "predicted": track.predicted,
            "last_confirmed_at_monotonic_ms": round(track.last_confirmed_at * 1000, 3),
            "last_confirmed_age_ms": round(max(0.0, now - track.last_confirmed_at) * 1000, 3),
        })
    return {
        "type": "tracks",
        "camera_id": result.camera_id,
        "sequence": result.source_sequence,
        "frame_width": result.frame_width,
        "frame_height": result.frame_height,
        "captured_at_monotonic_ms": round(result.source_captured_at * 1000, 3),
        "inference_started_at_monotonic_ms": round(result.inference_started_at * 1000, 3),
        "completed_at_monotonic_ms": round(result.completed_at * 1000, 3),
        "completed_at_unix_ms": round(result.completed_at * 1000 + wall_clock_offset_ms, 3),
        "websocket_sent_at_monotonic_ms": round(now * 1000, 3),
        "websocket_sent_at_unix_ms": round(now * 1000 + wall_clock_offset_ms, 3),
        "capture_to_inference_start_ms": round(
            (result.inference_started_at - result.source_captured_at) * 1000, 3
        ),
        "inference_latency_ms": round((result.completed_at - result.inference_started_at) * 1000, 3),
        "capture_to_result_latency_ms": round(
            (result.completed_at - result.source_captured_at) * 1000, 3
        ),
        "last_yolo_age_ms": round(max(0.0, now - (result.yolo_completed_at or result.completed_at)) * 1000, 3),
        "last_track_update_age_ms": round(max(0.0, now - result.completed_at) * 1000, 3),
        "actual_yolo_fps": actual_yolo_fps,
        "actual_tracker_fps": actual_tracker_fps,
        "configured_bytetrack_prediction_fps": configured_bytetrack_prediction_fps,
        "active_track_count": result.active_track_count,
        "lost_track_count": result.lost_track_count,
        "removed_track_count": result.removed_track_count,
        "predicted_track_count": result.predicted_track_count,
        "prediction_only": result.is_prediction,
        "boxes": boxes,
    }


class LatestMetadataMailbox:
    """At most one pending message per camera, regardless of client speed."""

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._closed = False
        self.replaced = 0

    def put(self, camera_id: str, message: dict[str, Any]) -> bool:
        with self._condition:
            if self._closed:
                return False
            current = self._pending.get(camera_id)
            if current and current.get("type") == message.get("type") \
                    and message.get("type") in {"detections", "tracks"}:
                current_time = float(current.get("captured_at_monotonic_ms", 0))
                new_time = float(message.get("captured_at_monotonic_ms", 0))
                if new_time < current_time or (
                    new_time == current_time
                    and int(message.get("sequence", 0)) <= int(current.get("sequence", 0))
                ):
                    return False
            replaced = camera_id in self._pending
            if replaced:
                self.replaced += 1
            self._pending[camera_id] = message
            self._condition.notify()
            return replaced

    def take(self, timeout: float = 1.0) -> dict[str, Any] | None:
        with self._condition:
            if not self._pending and not self._closed:
                self._condition.wait(timeout)
            if not self._pending:
                return None
            camera_id = next(iter(self._pending))
            return self._pending.pop(camera_id)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._pending)


class MetadataClient:
    def __init__(self) -> None:
        self.subscriptions: frozenset[str] = frozenset()
        self.mailbox = LatestMetadataMailbox()


class MetadataHub:
    """Non-blocking publisher shared by inference and WebSocket sessions."""

    def __init__(self, camera_ids: Iterable[str], metrics: Metrics, ttl_ms: int = 750) -> None:
        self.camera_ids = frozenset(camera_ids)
        self.metrics = metrics
        self.ttl_seconds = ttl_ms / 1000.0
        self._clients: set[MetadataClient] = set()
        self._latest: dict[str, dict[str, Any]] = {}
        self._active_message: dict[str, Any] = {
            "type": "active_camera", "camera_id": None, "status": "disabled"
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._expiry_loop, name="metadata-expiry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            self.remove_client(client)

    def add_client(self) -> MetadataClient:
        client = MetadataClient()
        with self._lock:
            self._clients.add(client)
            self._update_metrics_locked()
        return client

    def remove_client(self, client: MetadataClient) -> None:
        with self._lock:
            self._clients.discard(client)
            self._update_metrics_locked()
        client.mailbox.close()

    def subscribe(self, client: MetadataClient, camera_ids: Iterable[str]) -> None:
        requested = frozenset(camera_ids)
        invalid = sorted(requested - self.camera_ids)
        if invalid:
            raise ValueError(f"Unknown camera IDs: {', '.join(invalid)}")
        with self._lock:
            client.subscriptions = requested
            latest = [self._active_message]
            latest.extend(self._latest[camera_id] for camera_id in requested if camera_id in self._latest)
            self._update_metrics_locked()
        for message in latest:
            key = message.get("camera_id") or "__active_camera__"
            client.mailbox.put(key, message)

    def publish(self, result: DetectionResult) -> None:
        message = detection_message(result)
        with self._lock:
            current = self._latest.get(result.camera_id)
            if current and (
                float(message["captured_at_monotonic_ms"])
                < float(current.get("captured_at_monotonic_ms", 0))
                or (
                    float(message["captured_at_monotonic_ms"])
                    == float(current.get("captured_at_monotonic_ms", 0))
                    and int(message["sequence"]) <= int(current.get("sequence", 0))
                )
            ):
                self.metrics.camera(result.camera_id).stale_results += 1
                return
            self._latest[result.camera_id] = message
            targets = [client for client in self._clients if result.camera_id in client.subscriptions]
        for client in targets:
            replaced = client.mailbox.put(result.camera_id, message)
            self.metrics.record_metadata(result.camera_id, replaced)

    def publish_tracks(self, result: TrackResult, actual_yolo_fps: float,
                       actual_tracker_fps: float = 0.0,
                       configured_bytetrack_prediction_fps: float = 5.0) -> None:
        message = track_message(
            result, actual_yolo_fps, actual_tracker_fps,
            configured_bytetrack_prediction_fps,
        )
        with self._lock:
            current = self._latest.get(result.camera_id)
            if current and (
                float(message["captured_at_monotonic_ms"])
                < float(current.get("captured_at_monotonic_ms", 0))
                or (
                    float(message["captured_at_monotonic_ms"])
                    == float(current.get("captured_at_monotonic_ms", 0))
                    and int(message["sequence"]) <= int(current.get("sequence", 0))
                )
            ):
                self.metrics.camera(result.camera_id).stale_results += 1
                return
            self._latest[result.camera_id] = message
            targets = [client for client in self._clients if result.camera_id in client.subscriptions]
        for client in targets:
            replaced = client.mailbox.put(result.camera_id, message)
            self.metrics.record_metadata(result.camera_id, replaced)

    def clear_tracks(self, camera_id: str) -> None:
        if camera_id not in self.camera_ids:
            return
        message = {"type": "clear_tracks", "camera_id": camera_id}
        with self._lock:
            self._latest.pop(camera_id, None)
            targets = [client for client in self._clients if camera_id in client.subscriptions]
        for client in targets:
            replaced = client.mailbox.put(camera_id, message)
            self.metrics.record_metadata(camera_id, replaced)

    def active_camera(self, camera_id: str | None, status: str) -> None:
        if camera_id is not None and camera_id not in self.camera_ids:
            return
        if status not in {"disabled", "starting", "active", "stopping", "error"}:
            return
        message = {"type": "active_camera", "camera_id": camera_id, "status": status}
        with self._lock:
            self._active_message = message
            targets = list(self._clients)
        for client in targets:
            client.mailbox.put("__active_camera__", message)

    def expire_stale(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [
                camera_id for camera_id, message in self._latest.items()
                if "completed_at_monotonic_ms" in message
                and now * 1000 - float(message["completed_at_monotonic_ms"]) > self.ttl_seconds * 1000
            ]
            for camera_id in stale:
                self._latest.pop(camera_id, None)
                self.metrics.camera(camera_id).stale_results += 1
        for camera_id in stale:
            self.detector_status(camera_id, "stale", detection_age_ms=self.ttl_seconds * 1000)
        return stale

    def _expiry_loop(self) -> None:
        while not self._stop.wait(min(0.1, max(0.02, self.ttl_seconds / 4))):
            self.expire_stale()

    def detector_status(self, camera_id: str, status: str, actual_fps: float = 0.0,
                        detection_age_ms: float | None = None, message: str | None = None) -> None:
        if camera_id not in self.camera_ids or status not in {
            "starting", "active", "stopping", "error", "disabled", "stale", "offline"
        }:
            return
        message = {"type": "detector_status", "camera_id": camera_id, "status": status,
                   "actual_fps": actual_fps, "detection_age_ms": detection_age_ms,
                   "message": message}
        with self._lock:
            if status in {"stale", "error", "offline", "disabled"}:
                self._latest.pop(camera_id, None)
            targets = [client for client in self._clients if camera_id in client.subscriptions]
        for client in targets:
            replaced = client.mailbox.put(camera_id, message)
            self.metrics.record_metadata(camera_id, replaced)

    def _update_metrics_locked(self) -> None:
        subscriptions = {camera_id: sum(camera_id in client.subscriptions for client in self._clients)
                         for camera_id in self.camera_ids}
        self.metrics.metadata_clients(len(self._clients), subscriptions)

    @staticmethod
    def encode(message: dict[str, Any]) -> str:
        # The schema contains only scalar metadata and boxes; serialization is
        # centralized so image/frame objects cannot accidentally cross it.
        current = dict(message)
        if current.get("type") == "detections" and "completed_at_monotonic_ms" in current:
            current["metadata_age_ms"] = round(max(float(current.get("metadata_age_ms", 0.0)),
                                                       time.monotonic() * 1000 - current["completed_at_monotonic_ms"]), 3)
        if current.get("type") == "tracks" and "completed_at_monotonic_ms" in current:
            now_ms = time.monotonic() * 1000
            age = max(0.0, now_ms - float(current["completed_at_monotonic_ms"]))
            current["last_yolo_age_ms"] = round(age, 3)
            current["last_track_update_age_ms"] = round(age, 3)
            current["boxes"] = [dict(box, last_confirmed_age_ms=round(
                max(0.0, now_ms - float(box["last_confirmed_at_monotonic_ms"])), 3
            )) for box in current.get("boxes", [])]
        return json.dumps(current, separators=(",", ":"), allow_nan=False)
