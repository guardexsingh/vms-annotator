"""Exclusive, user-controlled AI session independent of direct video."""
from __future__ import annotations

import logging
import threading
import time

from .ai_capture import AICaptureWorker
from .bytetrack_tracker import ByteTrackPersonTracker
from .detection_backend import select_backend
from .detection_store import DetectionStore
from .inference_scheduler import InferenceScheduler
from .latest_frame import LatestFrame
from .metadata import MetadataHub
from .metrics import Metrics
from .models import AppConfig, CameraConfig, Frame
from .prediction_scheduler import PredictionScheduler

LOG = logging.getLogger(__name__)


class _DetectionSession:
    """All resources for one camera and one selection generation."""

    def __init__(self, owner: "DetectionRuntime", camera: CameraConfig, generation: int) -> None:
        self.owner, self.camera, self.generation = owner, camera, generation
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"detection-session-{camera.id}", daemon=True
        )
        self.slot: LatestFrame[Frame] = LatestFrame()
        self.backend = None
        self.tracker: ByteTrackPersonTracker | None = None
        self.tracker_lock = threading.Lock()
        self.capture: AICaptureWorker | None = None
        self.scheduler: InferenceScheduler | None = None
        self.prediction_scheduler: PredictionScheduler | None = None
        self.uses_retained_backend = False
        self.last_metadata_at: float | None = None
        self.pending_yolo_metadata = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def reset_tracker(self) -> None:
        with self.tracker_lock:
            if self.tracker is not None:
                self.tracker.reset()

    def track(self, result):
        with self.tracker_lock:
            if self.tracker is None:
                return None
            return self.tracker.update(result)

    def predict(self, now: float):
        with self.tracker_lock:
            if self.tracker is None:
                return None
            return self.tracker.predict(now)

    def _run(self) -> None:
        try:
            if self.owner.config.runtime.mode == "metadata_only":
                self.backend, selection, details = self.owner._retained_backend_for_session()
                self.uses_retained_backend = True
            else:
                self.backend, selection = select_backend(self.owner.config.detection)
                while not self.stop_event.is_set():
                    try:
                        details = self.backend.warmup()
                        break
                    except Exception as error:
                        summary = getattr(self.backend, "error_summary", None) or type(error).__name__
                        self.owner._session_failed(self, summary)
                        retry_after = getattr(self.backend, "retry_after", time.monotonic() + 2.0)
                        wait = max(1.0, min(60.0, retry_after - time.monotonic()))
                        if self.stop_event.wait(wait):
                            return
                        self.owner._session_loading(self)
                else:
                    return
            if self.stop_event.is_set() or not self.owner._is_current(self):
                return
            self.tracker = ByteTrackPersonTracker(self.camera.id, self.owner.config.tracking)
            self.owner._backend_ready(self, details, selection)
            self.capture = AICaptureWorker(
                self.camera, self.slot, self.owner.metrics,
                on_status=lambda camera_id, status: self.owner._capture_status(self, camera_id, status),
                max_dimension=self.owner.config.detection.capture_max_dimension,
                sample_fps=self.owner.config.detection.capture_fps,
            )
            self.scheduler = InferenceScheduler(
                {self.camera.id: self.slot},
                self.backend,
                DetectionStore(self.owner.config.tracking.remove_track_ms),
                self.owner.config.detection.target_fps_per_camera,
                on_result=lambda camera_id, result: self.owner._session_result(
                    self, camera_id, result
                ),
                on_error=lambda camera_id, error: self.owner._inference_error(
                    self, camera_id, error
                ),
                batch_mode="serial",
                max_frame_age_ms=self.owner.config.detection.max_frame_age_ms,
                max_batch_wait_ms=self.owner.config.detection.max_batch_wait_ms,
                on_consumed=self.owner.metrics.record_ai_consumed,
                on_stale=self.owner.metrics.record_stale_input,
                on_compute=self.owner.metrics.record_yolo_compute,
            )
            self.capture.start()
            self.scheduler.start()
            self.prediction_scheduler = PredictionScheduler(
                self.owner.config.tracking.prediction_fps,
                on_tick=lambda now: self.owner._prediction_tick(self, now),
                on_skipped=lambda skipped: self.owner._prediction_skipped(self, skipped),
            )
            self.owner.metrics.update_detector(
                inference_scheduler_interval_ms=self.scheduler.interval * 1000,
                prediction_scheduler_interval_ms=self.prediction_scheduler.interval * 1000,
            )
            LOG.info(
                "Detection scheduler camera=%s: YOLO interval %.1f ms; ByteTrack prediction interval %.1f ms",
                self.camera.id,
                self.scheduler.interval * 1000,
                self.prediction_scheduler.interval * 1000,
            )
            self.prediction_scheduler.start()
            self.stop_event.wait()
        except Exception as error:
            LOG.exception("Detection session failed for %s", self.camera.id)
            self.owner._session_failed(self, type(error).__name__)
        finally:
            if self.prediction_scheduler is not None:
                self.prediction_scheduler.stop()
            if self.scheduler is not None:
                self.scheduler.stop()
            if self.capture is not None:
                self.capture.stop()
            self.slot.clear()
            self.reset_tracker()
            if self.backend is not None and not self.uses_retained_backend:
                self.backend.close()


class DetectionRuntime:
    """Owns at most one on-demand camera AI session for all browser clients."""

    def __init__(self, config: AppConfig, metrics: Metrics, hub: MetadataHub) -> None:
        self.config, self.metrics, self.hub = config, metrics, hub
        self._cameras = {
            camera.id: camera for camera in config.cameras
            if camera.enabled and camera.detection_enabled
        }
        self._state_lock = threading.Lock()
        self._transition_lock = threading.Lock()
        self._active_camera_id: str | None = None
        self._status = "disabled"
        self._generation = 0
        self._session: _DetectionSession | None = None
        self._retained_backend = None
        self._retained_selection = None
        self._retained_details: dict | None = None
        self._stopped = False

    @property
    def cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(self._cameras.values())

    def start(self) -> None:
        for camera in self.cameras:
            self.metrics.camera(camera.id)
            self.metrics.reset_detection(camera.id)
        self.metrics.set_detector(
            "disabled", active_camera_id=None,
            tracker=self.config.tracking.tracker,
        )
        self.hub.active_camera(None, "disabled")
        if self.config.runtime.mode == "metadata_only":
            self._prepare_retained_backend()

    def stop(self) -> None:
        with self._transition_lock:
            self._stopped = True
            self._disable_locked()
            backend, self._retained_backend = self._retained_backend, None
            self._retained_selection = None
            self._retained_details = None
        if backend is not None:
            backend.close()

    def _prepare_retained_backend(self) -> None:
        """Fail startup early, then keep the loaded model/engine while idle.

        Metadata-only deployment has one active camera at most, so one retained
        backend is safe and avoids the model/engine warm-up on every alert.
        Capture, inference scheduling, and tracking remain session-scoped.
        """
        backend = None
        try:
            backend, selection = select_backend(self.config.detection)
            details = backend.warmup()
        except Exception:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            self.metrics.set_detector(
                "failed", "Detector startup preflight failed", active_camera_id=None,
                requested_inference_fps=self.config.detection.target_fps_per_camera,
                tracker=self.config.tracking.tracker,
            )
            raise
        self._retained_backend = backend
        self._retained_selection = selection
        self._retained_details = details
        if selection.selected == "tensorrt":
            LOG.info(
                "TensorRT startup: architecture=%s l4t=%s cuda_runtime=%s "
                "python=%s native=%s libnvinfer_package=%s python_package=%s "
                "backend=%s engine=%s sha256_prefix=%s",
                details.get("architecture"), details.get("l4t_release"),
                details.get("cuda_runtime_version"), details.get("tensorrt_python_version"),
                details.get("tensorrt_native_version"), details.get("libnvinfer_package_version"),
                details.get("python_libnvinfer_package_version"), selection.selected,
                details.get("engine_filename"), details.get("engine_sha256_prefix"),
            )
        self._set_retained_backend_metrics()

    def _set_retained_backend_metrics(self) -> None:
        """Expose a retained metadata-only TensorRT backend as ready while idle."""
        selection, details = self._retained_selection, self._retained_details
        if selection is None or details is None:
            return
        self.metrics.set_detector(
            "ready", active_camera_id=None, requested_backend=selection.requested,
            selected_backend=selection.selected, active_backend=selection.selected,
            execution_provider=getattr(selection, "execution_provider", None),
            device=selection.device, precision=selection.precision,
            model_load_ms=details.get("model_load_ms"), warmup_ms=details.get("warmup_ms"),
            requested_inference_fps=self.config.detection.target_fps_per_camera,
            tracker=self.config.tracking.tracker,
        )

    def _retained_backend_for_session(self):
        if self._retained_backend is None or self._retained_selection is None or self._retained_details is None:
            raise RuntimeError("Metadata-only detector backend is not ready")
        return self._retained_backend, self._retained_selection, self._retained_details

    def state(self) -> dict[str, object]:
        with self._state_lock:
            return {"camera_id": self._active_camera_id, "status": self._status}

    def set_active_camera(self, camera_id: str | None) -> dict[str, object]:
        if camera_id is not None and camera_id not in self._cameras:
            raise ValueError("Camera is not enabled for detection")
        with self._transition_lock:
            if self._stopped:
                raise RuntimeError("Detection runtime is stopped")
            with self._state_lock:
                if camera_id == self._active_camera_id:
                    return {"camera_id": self._active_camera_id, "status": self._status}
            self._disable_locked()
            if camera_id is None:
                return self.state()
            camera = self._cameras[camera_id]
            self.metrics.reset_detection(camera_id)
            self.metrics.camera(camera_id).requested_inference_fps = (
                self.config.detection.target_fps_per_camera
            )
            self.metrics.camera(camera_id).requested_tracker_fps = (
                self.config.tracking.prediction_fps
            )
            self.metrics.camera(camera_id).configured_bytetrack_prediction_fps = (
                self.config.tracking.prediction_fps
            )
            with self._state_lock:
                self._generation += 1
                generation = self._generation
                self._active_camera_id, self._status = camera_id, "starting"
                session = self._session = _DetectionSession(self, camera, generation)
            self.metrics.set_detector(
                "loading", active_camera_id=camera_id,
                requested_inference_fps=self.config.detection.target_fps_per_camera,
                tracker=self.config.tracking.tracker,
            )
            self.hub.active_camera(camera_id, "starting")
            self.hub.detector_status(camera_id, "starting")
            session.start()
            return self.state()

    def _disable_locked(self) -> None:
        with self._state_lock:
            self._generation += 1
            old_camera = self._active_camera_id
            session = self._session
            self._active_camera_id, self._session, self._status = None, None, "disabled"
        if old_camera is not None:
            self.hub.active_camera(old_camera, "stopping")
            self.hub.detector_status(old_camera, "stopping")
            self.hub.clear_tracks(old_camera)
        if session is not None:
            session.stop()
        if old_camera is not None:
            self.metrics.reset_detection(old_camera)
        if self.config.runtime.mode == "metadata_only" and self._retained_backend is not None:
            self._set_retained_backend_metrics()
        else:
            self.metrics.set_detector(
                "disabled", active_camera_id=None,
                requested_inference_fps=self.config.detection.target_fps_per_camera,
                tracker=self.config.tracking.tracker,
            )
        self.hub.active_camera(None, "disabled")

    def _is_current(self, session: _DetectionSession) -> bool:
        with self._state_lock:
            return (
                self._session is session
                and self._generation == session.generation
                and self._active_camera_id == session.camera.id
            )

    def _set_current_status(self, session: _DetectionSession, status: str) -> bool:
        with self._state_lock:
            if not self._is_current_unlocked(session):
                return False
            self._status = status
            return True

    def _is_current_unlocked(self, session: _DetectionSession) -> bool:
        return (
            self._session is session
            and self._generation == session.generation
            and self._active_camera_id == session.camera.id
        )

    def _session_loading(self, session: _DetectionSession) -> None:
        if not self._set_current_status(session, "starting"):
            return
        self.metrics.set_detector(
            "loading", active_camera_id=session.camera.id,
            requested_inference_fps=self.config.detection.target_fps_per_camera,
            tracker=self.config.tracking.tracker,
        )
        self.hub.active_camera(session.camera.id, "starting")
        self.hub.detector_status(session.camera.id, "starting")

    def _backend_ready(self, session: _DetectionSession, details: dict, selection) -> None:
        if not self._is_current(session):
            return
        LOG.info(
            "On-demand detector ready camera=%s backend=%s device=%s precision=%s tracker=bytetrack",
            session.camera.id, selection.selected, selection.device, selection.precision,
        )
        detector_details = {
            **details,
            "active_camera_id": session.camera.id,
            "requested_backend": selection.requested,
            "selected_backend": selection.selected,
            "active_backend": selection.selected,
            "execution_provider": getattr(selection, "execution_provider", None),
            "requested_yolo_fps": self.config.detection.target_fps_per_camera,
            "yaml_yolo_fps": self.config.detection.yaml_target_fps_per_camera,
            "yolo_fps_environment_override": self.config.detection.inference_fps_environment_override,
            "yolo_fps_source": self.config.detection.inference_fps_source,
            "requested_ai_capture_fps": self.config.detection.capture_fps,
            "ai_capture_fps_source": self.config.detection.capture_fps_source,
            "requested_precision": self.config.detection.precision,
            "precision_source": self.config.detection.precision_source,
            "device": selection.device,
            "precision": selection.precision,
            "batch_mode": "serial",
            "inference_workers": 1,
            "model_copies": 1,
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "tracker": ByteTrackPersonTracker.implementation,
            "track_buffer_cycles": self.config.tracking.track_buffer,
            "hold_box_ms": self.config.tracking.hold_box_ms,
            "remove_track_ms": self.config.tracking.remove_track_ms,
            "prediction_fps": self.config.tracking.prediction_fps,
            "yaml_prediction_fps": self.config.tracking.yaml_prediction_fps,
            "prediction_fps_environment_override": self.config.tracking.prediction_fps_environment_override,
            "prediction_fps_source": self.config.tracking.prediction_fps_source,
            "prediction_adapter": "KalmanFilterXYAH elapsed-time transition",
        }
        self.metrics.set_detector("ready", **detector_details)

    def _capture_status(self, session: _DetectionSession, camera_id: str, status: str) -> None:
        if not self._is_current(session):
            return
        if status == "ready":
            if self._set_current_status(session, "active"):
                self.hub.active_camera(camera_id, "active")
                self.hub.detector_status(camera_id, "active")
            return
        session.slot.clear()
        session.reset_tracker()
        self.metrics.clear_tracks(camera_id)
        if self._set_current_status(session, "starting"):
            self.hub.active_camera(camera_id, "starting")
            self.hub.detector_status(camera_id, "offline")
            self.hub.clear_tracks(camera_id)

    def _session_result(self, session: _DetectionSession, camera_id: str, result) -> None:
        if not self._is_current(session):
            return
        self.metrics.record_inference(camera_id, result)
        tracked = session.track(result)
        if tracked is None or not self._is_current(session):
            return
        self.metrics.record_tracks(camera_id, tracked)
        self._publish_track_result(session, camera_id, tracked)
        if self._set_current_status(session, "active"):
            self.hub.active_camera(camera_id, "active")

    def _prediction_tick(self, session: _DetectionSession, now: float) -> None:
        if not self._is_current(session):
            return
        started = time.monotonic()
        cpu_started = time.thread_time()
        tracked = session.predict(now)
        if tracked is None or not self._is_current(session):
            return
        self.metrics.record_tracks(session.camera.id, tracked)
        self.metrics.record_prediction(
            session.camera.id, tracked, (time.monotonic() - started) * 1000,
            (time.thread_time() - cpu_started) * 1000,
        )
        self._publish_track_result(session, session.camera.id, tracked)

    def _prediction_skipped(self, session: _DetectionSession, skipped: int) -> None:
        if self._is_current(session):
            self.metrics.record_prediction_skipped(session.camera.id, skipped)

    def _publish_track_result(self, session: _DetectionSession, camera_id: str, tracked) -> None:
        # Do not add a correction on top of a prediction tick (or vice versa).
        # A rate-limited correction is published at the next slot with its
        # original source=yolo metadata, rather than silently relabeling it as
        # a prediction. This keeps the browser rate bounded without concealing
        # a real observation.
        interval = 1.0 / self.config.tracking.prediction_fps
        now = time.monotonic()
        if (session.last_metadata_at is not None
                and now - session.last_metadata_at < interval * .9):
            if not tracked.is_prediction:
                session.pending_yolo_metadata = tracked
            return
        if session.pending_yolo_metadata is not None:
            tracked = session.pending_yolo_metadata
            session.pending_yolo_metadata = None
        self.hub.publish_tracks(
            tracked,
            self.metrics.camera_inference_fps(camera_id),
            self.metrics.camera_prediction_fps(camera_id),
            self.config.tracking.prediction_fps,
            self.config.detection.target_fps_per_camera,
            getattr(session.backend, "name", self.config.detection.backend),
        )
        session.last_metadata_at = now
        self.metrics.record_track_metadata(camera_id)

    def _inference_error(self, session: _DetectionSession, camera_id: str,
                         error: BaseException) -> None:
        summary = getattr(session.backend, "error_summary", None) or type(error).__name__
        self._session_failed(session, summary)

    def _session_failed(self, session: _DetectionSession, summary: str) -> None:
        if not self._set_current_status(session, "error"):
            return
        self.metrics.set_detector(
            "failed", summary, active_camera_id=session.camera.id,
            requested_inference_fps=self.config.detection.target_fps_per_camera,
            tracker=self.config.tracking.tracker,
        )
        self.hub.active_camera(session.camera.id, "error")
        self.hub.detector_status(
            session.camera.id, "error", message="Detector could not start or process frames"
        )
