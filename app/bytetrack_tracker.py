"""Person-only ByteTrack adapter with elapsed-time prediction between YOLO updates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .models import DetectionResult, TrackResult, TrackedPerson, TrackingConfig


class _ByteResults:
    def __init__(self, result: DetectionResult) -> None:
        detections = result.detections
        self.conf = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
        self.cls = np.zeros(len(detections), dtype=np.float32)
        xywh = []
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            xywh.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape((-1, 4))


@dataclass
class _TrackRecord:
    person: TrackedPerson


def predict_kalman_with_dt(track, dt_seconds: float) -> None:
    """Advance Ultralytics' XYAH state with a real elapsed-time transition.

    Ultralytics' public ``STrack.predict()`` is a one-frame transition with
    ``dt=1``.  This small adapter uses its existing Kalman state/filter but
    supplies ``dt`` to the constant-velocity transition.  Process-noise is
    scaled by sqrt(dt), preserving the one-second covariance when five 200 ms
    prediction ticks span one second.  The installed package is not modified.
    """
    if track.mean is None or track.covariance is None or dt_seconds <= 0:
        return
    dt = float(min(1.0, dt_seconds))
    mean = track.mean.copy()
    # Match STrack.predict(): lost tracks do not keep changing height.
    if str(getattr(track, "state", "")) != "TrackState.Tracked" and getattr(track.state, "name", "") != "Tracked":
        mean[7] = 0
    kf = track.kalman_filter
    motion = np.eye(8, dtype=np.float32)
    for index in range(4):
        motion[index, index + 4] = dt
    scale = np.sqrt(dt)
    height = max(1.0, float(mean[3]))
    std_pos = [kf._std_weight_position * height * scale,
               kf._std_weight_position * height * scale,
               1e-2 * scale,
               kf._std_weight_position * height * scale]
    std_vel = [kf._std_weight_velocity * height * scale,
               kf._std_weight_velocity * height * scale,
               1e-5 * scale,
               kf._std_weight_velocity * height * scale]
    covariance = np.linalg.multi_dot((motion, track.covariance, motion.T))
    track.mean = motion @ mean
    track.covariance = covariance + np.diag(np.square(np.r_[std_pos, std_vel]))


class ByteTrackPersonTracker:
    """Ultralytics 8.3.0 ByteTrack with correction at 1 FPS and prediction at 5 FPS."""

    implementation = "ultralytics-8.3.0-bytetrack-dt-adapter"

    def __init__(self, camera_id: str, config: TrackingConfig) -> None:
        self.camera_id, self.config = camera_id, config
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            match_thresh=config.match_thresh,
            track_buffer=config.track_buffer,
            fuse_score=config.fuse_score,
        )
        # ByteTrack's lost buffer is intentionally in YOLO-correction cycles.
        self._tracker = BYTETracker(args, frame_rate=30)
        self._tracker.max_time_lost = config.track_buffer
        self._records: dict[int, _TrackRecord] = {}
        self._last_result: DetectionResult | None = None
        self._last_prediction_at: float | None = None
        self._update_sequence = 0
        self.removed_count = 0

    def reset(self) -> None:
        self._tracker.reset()
        self._records.clear()
        self._last_result = None
        self._last_prediction_at = None
        self._update_sequence = 0
        self.removed_count = 0

    def update(self, result: DetectionResult) -> TrackResult:
        """Apply one fresh YOLO correction; detections are never reused later."""
        if any(detection.class_id != 0 for detection in result.detections):
            raise ValueError("ByteTrack accepts person class 0 only")
        # Bring the state up to the correction timestamp, then suppress
        # BYTETracker.update()'s fixed dt=1 multi_predict call.  The local
        # elapsed-time adapter has already performed the needed transition.
        self._advance_tracks(result.completed_at)
        original_predict = self._tracker.multi_predict
        self._tracker.multi_predict = lambda _: None
        try:
            rows = self._tracker.update(_ByteResults(result))
        finally:
            self._tracker.multi_predict = original_predict
        self._last_result = result
        self._last_prediction_at = result.completed_at
        self._update_sequence += 1
        active_ids: set[int] = set()
        for row in rows:
            track_id = int(row[4])
            active_ids.add(track_id)
            self._records[track_id] = _TrackRecord(TrackedPerson(
                camera_id=self.camera_id,
                track_id=track_id,
                xyxy=self._clamped_xyxy(tuple(float(value) for value in row[:4]), result),
                confidence=float(row[5]),
                track_state="active",
                last_confirmed_at=result.completed_at,
                source="yolo",
                predicted=False,
            ))
        self._mark_lost_and_remove(result.completed_at)
        return self._result(result, result.completed_at, active_ids, is_prediction=False)

    def predict(self, now: float) -> TrackResult | None:
        """Publish prediction-only track state without invoking YOLO or association."""
        if self._last_result is None:
            return None
        elapsed = max(0.0, now - (self._last_prediction_at or now))
        self._advance_tracks(now)
        self._update_sequence += 1
        self._mark_lost_and_remove(now)
        result = self._last_result
        active_ids = {
            int(track.track_id) for track in self._tracker.tracked_stracks
            if int(track.track_id) in self._records
        }
        for track_id in active_ids:
            record = self._records[track_id]
            age_ms = (now - record.person.last_confirmed_at) * 1000
            if age_ms > self.config.hold_box_ms:
                continue
            track = self._underlying_track(track_id)
            if track is None:
                continue
            xyxy = self._clamped_xyxy(tuple(float(value) for value in track.xyxy), result)
            if xyxy is None:
                continue
            xyxy = self._bounded_motion(record.person.xyxy, xyxy, result, elapsed)
            if xyxy is None:
                continue
            record.person = replace(
                record.person, xyxy=xyxy, track_state="predicted",
                source="bytetrack_prediction", predicted=True,
            )
        self._last_prediction_at = now
        return self._result(result, now, active_ids, is_prediction=True)

    def _advance_tracks(self, now: float) -> None:
        if self._last_prediction_at is None:
            self._last_prediction_at = now
            return
        dt = max(0.0, now - self._last_prediction_at)
        if dt <= 0:
            return
        for track in self._tracker.tracked_stracks:
            if int(track.track_id) in self._records:
                predict_kalman_with_dt(track, dt)
        self._last_prediction_at = now

    def _underlying_track(self, track_id: int):
        return next((track for track in self._tracker.tracked_stracks if int(track.track_id) == track_id), None)

    def _mark_lost_and_remove(self, now: float) -> None:
        lost_ids = {int(track.track_id) for track in self._tracker.lost_stracks}
        removed_ids = {int(track.track_id) for track in self._tracker.removed_stracks}
        for track_id in lost_ids:
            record = self._records.get(track_id)
            if record is not None:
                record.person = replace(record.person, track_state="lost")
        expired = []
        for track_id, record in self._records.items():
            age_ms = (now - record.person.last_confirmed_at) * 1000
            if track_id in removed_ids or age_ms >= self.config.remove_track_ms:
                expired.append(track_id)
        if not expired:
            return
        for track_id in expired:
            self._records.pop(track_id, None)
            self.removed_count += 1
        expired_set = set(expired)
        self._tracker.tracked_stracks = [track for track in self._tracker.tracked_stracks
                                         if int(track.track_id) not in expired_set]
        self._tracker.lost_stracks = [track for track in self._tracker.lost_stracks
                                      if int(track.track_id) not in expired_set]

    def _bounded_motion(self, previous, predicted, result: DetectionResult, dt: float):
        width, height = max(1, result.frame_width), max(1, result.frame_height)
        px, py = (previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2
        nx, ny = (predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2
        dx, dy = (nx - px) / width, (ny - py) / height
        displacement = float(np.hypot(dx, dy))
        old_size = ((previous[2] - previous[0]) / width, (previous[3] - previous[1]) / height)
        new_size = ((predicted[2] - predicted[0]) / width, (predicted[3] - predicted[1]) / height)
        size_change = max(abs(old_size[0] - new_size[0]), abs(old_size[1] - new_size[1]))
        if displacement < self.config.prediction_deadzone_norm and size_change < self.config.prediction_deadzone_norm:
            return previous
        maximum = self.config.max_prediction_displacement_norm_per_second * max(0.0, dt)
        if maximum > 0 and displacement > maximum:
            scale = maximum / displacement
            shift_x, shift_y = (nx - px) * scale, (ny - py) * scale
            width_box, height_box = predicted[2] - predicted[0], predicted[3] - predicted[1]
            predicted = (px + shift_x - width_box / 2, py + shift_y - height_box / 2,
                         px + shift_x + width_box / 2, py + shift_y + height_box / 2)
        return self._clamped_xyxy(predicted, result)

    @staticmethod
    def _clamped_xyxy(xyxy, result: DetectionResult):
        if xyxy is None or not np.isfinite(xyxy).all():
            return None
        width, height = max(1, result.frame_width), max(1, result.frame_height)
        x1, y1, x2, y2 = xyxy
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _result(self, original: DetectionResult, completed_at: float, active_ids: set[int],
                is_prediction: bool) -> TrackResult:
        visible = []
        predicted = 0
        for track_id, record in self._records.items():
            age_ms = (completed_at - record.person.last_confirmed_at) * 1000
            if record.person.track_state == "predicted" and age_ms <= self.config.hold_box_ms:
                predicted += 1
                visible.append(record.person)
            elif track_id in active_ids and age_ms <= self.config.hold_box_ms:
                visible.append(record.person)
            elif record.person.track_state == "lost" and age_ms <= self.config.hold_box_ms:
                visible.append(record.person)
        visible.sort(key=lambda person: person.track_id)
        return TrackResult(
            camera_id=self.camera_id,
            tracks=tuple(visible),
            source_captured_at=original.source_captured_at,
            inference_started_at=original.inference_started_at,
            completed_at=completed_at,
            source_sequence=self._update_sequence,
            frame_width=original.frame_width,
            frame_height=original.frame_height,
            active_track_count=len(active_ids),
            lost_track_count=len(self._tracker.lost_stracks),
            removed_track_count=self.removed_count,
            predicted_track_count=predicted,
            is_prediction=is_prediction,
            yolo_completed_at=original.completed_at,
        )
