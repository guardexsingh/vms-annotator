from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import psutil

from .models import AppConfig, CameraMetrics, DetectionConfig, TrackingConfig


def _host_telemetry() -> dict[str, object]:
    temperatures: dict[str, float] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text().strip()
            raw = float((zone / "temp").read_text().strip())
            temperatures[name] = round(raw / 1000 if raw > 1000 else raw, 3)
        except (OSError, TypeError, ValueError):
            continue
    cpu_clocks: list[float] = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            cpu_clocks.append(round(float(path.read_text().strip()) / 1000, 3))
        except (OSError, TypeError, ValueError):
            continue
    gpu_clock = None
    for path in (
        Path("/sys/class/devfreq/17000000.gpu/cur_freq"),
        Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"),
    ):
        try:
            gpu_clock = round(float(path.read_text().strip()) / 1_000_000, 3)
            break
        except (OSError, TypeError, ValueError):
            continue
    return {
        "temperatures_c": temperatures,
        "cpu_clock_mhz": cpu_clocks,
        "gpu_clock_mhz": gpu_clock,
    }


class Metrics:
    """Live metrics bound to the frozen effective application configuration."""

    def __init__(self, config: AppConfig | None = None) -> None:
        # AppConfig and its nested configs are frozen dataclasses.  Retaining
        # this reference makes metrics, health, and runtime report the same
        # resolved environment > YAML > default values.
        self.config = config
        self._detection_config = config.detection if config is not None else DetectionConfig()
        self._tracking_config = config.tracking if config is not None else TrackingConfig()
        self.cameras: dict[str, CameraMetrics] = {}
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=4096))
        self._lock = threading.Lock()
        self._started = time.monotonic()
        # Keep this object for delta-based CPU accounting; constructing a new
        # Process for every snapshot would make cpu_percent() report zero.
        self._process = psutil.Process()
        self._events: dict[str, dict[str, deque[float]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=512)))
        self._detector_state = "loading"
        self._detector_error: str | None = None
        self._detector_detail: dict[str, object] = {}
        self._metadata_clients = 0
        self._last_result_at: dict[str, float] = {}
        self._last_track_confirmed: dict[str, float] = {}
        self._ai_decoder_processes: dict[str, psutil.Process] = {}

    def _configured_detector_detail(self) -> dict[str, object]:
        detection, tracking = self._detection_config, self._tracking_config
        return {
            "requested_backend": detection.backend,
            "backend_source": detection.backend_source,
            "requested_inference_fps": detection.target_fps_per_camera,
            "requested_yolo_fps": detection.target_fps_per_camera,
            "yolo_fps_source": detection.inference_fps_source,
            "requested_ai_capture_fps": detection.capture_fps,
            "ai_capture_fps_source": detection.capture_fps_source,
            "requested_precision": detection.precision,
            "precision_source": detection.precision_source,
            "configured_bytetrack_prediction_fps": tracking.prediction_fps,
            "prediction_fps": tracking.prediction_fps,
            "prediction_fps_source": tracking.prediction_fps_source,
        }

    def _apply_configured_targets(self, metric: CameraMetrics) -> None:
        metric.requested_inference_fps = self._detection_config.target_fps_per_camera
        metric.requested_tracker_fps = self._tracking_config.prediction_fps
        metric.configured_bytetrack_prediction_fps = self._tracking_config.prediction_fps

    def set_detector(self, state: str, error: str | None = None, **detail: object) -> None:
        if state not in {"loading", "ready", "failed", "disabled"}:
            raise ValueError(f"Invalid detector state: {state}")
        with self._lock:
            self._detector_state, self._detector_error = state, error
            self._detector_detail = {**self._configured_detector_detail(), **detail}

    def update_detector(self, **detail: object) -> None:
        """Attach runtime evidence without replacing effective config fields."""
        with self._lock:
            self._detector_detail = {**self._detector_detail, **detail}

    def camera(self, camera_id: str) -> CameraMetrics:
        with self._lock:
            metric = self.cameras.setdefault(camera_id, CameraMetrics(camera_id))
            self._apply_configured_targets(metric)
            return metric

    def record_inference(self, camera_id: str, result) -> None:
        total = (result.completed_at - result.inference_started_at) * 1000
        with self._lock:
            metric = self.cameras[camera_id]
            metric.latest_inference_latency_ms = total
            metric.capture_to_inference_start_ms = (
                result.inference_started_at - result.source_captured_at
            ) * 1000
            metric.capture_to_result_latency_ms = (result.completed_at - result.source_captured_at) * 1000
            previous = self._last_result_at.get(camera_id)
            metric.result_interval_ms = (
                (result.completed_at - previous) * 1000 if previous is not None else None
            )
            self._last_result_at[camera_id] = result.completed_at
            metric.detection_age_ms = 0.0
            metric.person_count = len(result.detections)
            metric.completed_inferences += 1
            self._samples["inference_ms"].append(total)
            self._samples[f"inference_ms:{camera_id}"].append(total)
            self._events[camera_id]["inference"].append(time.monotonic())

    def record_yolo_compute(self, camera_id: str, cpu_ms: float) -> None:
        with self._lock:
            self._samples[f"yolo_cpu_ms:{camera_id}"].append(cpu_ms)

    def set_ai_decoder_pid(self, camera_id: str, pid: int | None) -> None:
        with self._lock:
            if pid is None:
                self._ai_decoder_processes.pop(camera_id, None)
                return
            try:
                self._ai_decoder_processes[camera_id] = psutil.Process(pid)
            except psutil.Error:
                self._ai_decoder_processes.pop(camera_id, None)

    def record_ai_consumed(self, camera_id: str, frame) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            metric.ai_frame_age_ms = max(0.0, (time.monotonic() - frame.captured_at) * 1000)
            self._events[camera_id]["ai_consumed"].append(time.monotonic())

    def record_stale_input(self, camera_id: str) -> None:
        with self._lock:
            self.cameras[camera_id].stale_input_frames += 1

    def record_tracks(self, camera_id: str, result) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            metric.active_track_count = result.active_track_count
            metric.lost_track_count = result.lost_track_count
            metric.removed_track_count = result.removed_track_count
            metric.predicted_track_count = result.predicted_track_count
            metric.person_count = len(result.tracks)
            if result.tracks:
                confirmed = max(track.last_confirmed_at for track in result.tracks)
                self._last_track_confirmed[camera_id] = confirmed
                metric.last_confirmed_age_ms = max(0.0, (time.monotonic() - confirmed) * 1000)
            elif camera_id not in self._last_track_confirmed:
                metric.last_confirmed_age_ms = None

    def clear_tracks(self, camera_id: str) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            metric.person_count = 0
            metric.active_track_count = 0
            metric.lost_track_count = 0
            metric.last_confirmed_age_ms = None
            self._last_track_confirmed.pop(camera_id, None)

    def reset_detection(self, camera_id: str) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            for field, value in {
                "requested_inference_fps": self._detection_config.target_fps_per_camera,
                "completed_inference_fps": 0.0,
                "ai_capture_fps": 0.0,
                "ai_frames_consumed_fps": 0.0,
                "ai_frame_age_ms": None,
                "ai_output_width": 0,
                "ai_output_height": 0,
                "ai_pixel_rate_mib_s": 0.0,
                "ai_capture_status": "disabled",
                "ai_capture_backend": "none",
                "decode_queue_depth": 0,
                "decoder_command": "",
                "latest_inference_latency_ms": None,
                "capture_to_inference_start_ms": None,
                "capture_to_result_latency_ms": None,
                "result_interval_ms": None,
                "detection_age_ms": None,
                "person_count": 0,
                "active_track_count": 0,
                "lost_track_count": 0,
                "removed_track_count": 0,
                "last_confirmed_age_ms": None,
                "requested_tracker_fps": self._tracking_config.prediction_fps,
                "actual_tracker_fps": 0.0,
                "configured_bytetrack_prediction_fps": self._tracking_config.prediction_fps,
                "actual_bytetrack_prediction_fps": 0.0,
                "yolo_updates_total": 0,
                "prediction_updates_total": 0,
                "prediction_ticks_skipped": 0,
                "predicted_track_count": 0,
                "prediction_compute_p50_ms": None,
                "prediction_compute_p95_ms": None,
                "prediction_cpu_p50_ms": None,
                "prediction_cpu_p95_ms": None,
                "yolo_compute_cpu_p50_ms": None,
                "yolo_compute_cpu_p95_ms": None,
                "ai_decoder_cpu_percent": None,
                "metadata_publish_fps": 0.0,
            }.items():
                setattr(metric, field, value)
            for kind in ("inference", "ai_capture", "ai_consumed", "prediction", "track_metadata"):
                self._events[camera_id][kind].clear()
            self._samples[f"inference_ms:{camera_id}"].clear()
            self._samples[f"prediction_ms:{camera_id}"].clear()
            self._samples[f"prediction_cpu_ms:{camera_id}"].clear()
            self._samples[f"yolo_cpu_ms:{camera_id}"].clear()
            self._ai_decoder_processes.pop(camera_id, None)
            self._last_result_at.pop(camera_id, None)
            self._last_track_confirmed.pop(camera_id, None)

    def metadata_clients(self, count: int, subscriptions: dict[str, int]) -> None:
        with self._lock:
            self._metadata_clients = count
            for camera_id, metric in self.cameras.items():
                metric.websocket_subscribers = subscriptions.get(camera_id, 0)

    def record_metadata(self, camera_id: str, replaced: bool) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            metric.metadata_messages_sent += 1
            if replaced:
                metric.metadata_messages_replaced += 1

    def record_prediction(self, camera_id: str, result, compute_ms: float,
                          cpu_ms: float = 0.0, skipped: int = 0) -> None:
        with self._lock:
            metric = self.cameras[camera_id]
            metric.prediction_updates_total += 1
            metric.prediction_ticks_skipped += skipped
            metric.predicted_track_count = result.predicted_track_count
            self._samples[f"prediction_ms:{camera_id}"].append(compute_ms)
            self._samples[f"prediction_cpu_ms:{camera_id}"].append(cpu_ms)
            self._events[camera_id]["prediction"].append(time.monotonic())

    def record_prediction_skipped(self, camera_id: str, skipped: int) -> None:
        with self._lock:
            self.cameras[camera_id].prediction_ticks_skipped += skipped

    def record_track_metadata(self, camera_id: str) -> None:
        with self._lock:
            self._events[camera_id]["track_metadata"].append(time.monotonic())

    def record_browser_stats(self, camera_id: str, payload: dict[str, object]) -> None:
        with self._lock:
            metric = self.cameras.get(camera_id)
            if metric is None:
                raise KeyError(camera_id)
            codec = str(payload.get("webrtc_codec", "unknown")).lower()
            metric.webrtc_codec = codec if codec in {"hevc", "h265"} else "unsupported"
            for name in ("browser_received_fps", "browser_decoded_fps",
                         "browser_jitter_buffer_delay_ms"):
                value = payload.get(name)
                setattr(metric, name, None if value is None else max(0.0, float(value)))
            dropped = payload.get("browser_frames_dropped")
            metric.browser_frames_dropped = None if dropped is None else max(0, int(dropped))

    def record_frame(self, camera_id: str, kind: str) -> None:
        with self._lock:
            self._events[camera_id][kind].append(time.monotonic())

    def camera_inference_fps(self, camera_id: str) -> float:
        with self._lock:
            return self._rate(self._events[camera_id]["inference"], time.monotonic(), 10.0)

    def camera_prediction_fps(self, camera_id: str) -> float:
        with self._lock:
            return self._rate(self._events[camera_id]["prediction"], time.monotonic(), 10.0)

    @staticmethod
    def _rate(events: deque[float], now: float, window: float = 1.0) -> float:
        return sum(1 for event in events if now - event <= window) / window

    def snapshot(self) -> dict:
        with self._lock:
            active_camera_id = self._detector_detail.get("active_camera_id")
            values = (
                list(self._samples[f"inference_ms:{active_camera_id}"])
                if active_camera_id else []
            )
            ordered = sorted(values)
            percentile_for = lambda series, p: series[min(len(series) - 1, int((len(series) - 1) * p))] if series else None
            percentile = lambda p: percentile_for(ordered, p)
            process = self._process
            now = time.monotonic()
            cameras = {key: vars(value).copy() for key, value in self.cameras.items()}
            for key, camera in cameras.items():
                camera["decoded_fps"] = self._rate(self._events[key]["decoded"], now)
                camera["compositor_input_fps"] = self._rate(self._events[key]["compositor_input"], now)
                camera["composited_fps"] = self._rate(self._events[key]["composited"], now)
                camera["encoded_fps"] = self._rate(self._events[key]["encoded"], now)
                camera["published_fps"] = self._rate(self._events[key]["published"], now)
                camera["output_fps"] = camera["published_fps"]
                camera["completed_inference_fps"] = self._rate(self._events[key]["inference"], now, 10.0)
                camera["actual_tracker_fps"] = self._rate(self._events[key]["prediction"], now, 10.0)
                camera["actual_bytetrack_prediction_fps"] = camera["actual_tracker_fps"]
                camera["yolo_updates_total"] = camera["completed_inferences"]
                camera["metadata_publish_fps"] = self._rate(self._events[key]["track_metadata"], now, 10.0)
                camera["ai_capture_fps"] = self._rate(self._events[key]["ai_capture"], now)
                camera["ai_frames_consumed_fps"] = self._rate(self._events[key]["ai_consumed"], now, 10.0)
                camera["ai_pixel_rate_mib_s"] = (
                    camera["ai_output_width"] * camera["ai_output_height"] * 3
                    * camera["ai_capture_fps"] / 1048576
                )
                completed = self._events[key]["inference"][-1] if self._events[key]["inference"] else None
                camera["detection_age_ms"] = (now - completed) * 1000 if completed is not None else None
                confirmed = self._last_track_confirmed.get(key)
                camera["last_confirmed_age_ms"] = (
                    (now - confirmed) * 1000 if confirmed is not None else None
                )
                per_camera = sorted(self._samples[f"inference_ms:{key}"])
                camera["inference_p50_ms"] = percentile_for(per_camera, .5)
                camera["inference_p95_ms"] = percentile_for(per_camera, .95)
                prediction = sorted(self._samples[f"prediction_ms:{key}"])
                camera["prediction_compute_p50_ms"] = percentile_for(prediction, .5)
                camera["prediction_compute_p95_ms"] = percentile_for(prediction, .95)
                prediction_cpu = sorted(self._samples[f"prediction_cpu_ms:{key}"])
                camera["prediction_cpu_p50_ms"] = percentile_for(prediction_cpu, .5)
                camera["prediction_cpu_p95_ms"] = percentile_for(prediction_cpu, .95)
                yolo_cpu = sorted(self._samples[f"yolo_cpu_ms:{key}"])
                camera["yolo_compute_cpu_p50_ms"] = percentile_for(yolo_cpu, .5)
                camera["yolo_compute_cpu_p95_ms"] = percentile_for(yolo_cpu, .95)
                decoder = self._ai_decoder_processes.get(key)
                try:
                    camera["ai_decoder_cpu_percent"] = decoder.cpu_percent(None) if decoder else None
                except psutil.Error:
                    self._ai_decoder_processes.pop(key, None)
                    camera["ai_decoder_cpu_percent"] = None
                if not camera["transcoding"]:
                    for field in ("encoded_fps", "published_fps", "output_fps",
                                  "encoder_failures", "publish_failures",
                                  "encoder_queue_depth", "encoder_command",
                                  "encoder_write_ms", "encoder"):
                        camera.pop(field, None)
            return {"uptime_seconds": round(time.monotonic() - self._started, 1),
                    "cameras": cameras,
                    "detector": {"state": self._detector_state, "error": self._detector_error,
                                 **self._detector_detail},
                    "inference": {"samples": len(values), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
                                  "max_ms": max(values) if values else None},
                    "metadata": {"websocket_clients": self._metadata_clients},
                    "system": {"cpu_percent": psutil.cpu_percent(), "application_cpu_percent": process.cpu_percent(None), "ram_percent": psutil.virtual_memory().percent,
                               "ram_used_bytes": psutil.virtual_memory().used,
                               "process_rss_bytes": process.memory_info().rss,
                               "gpu_utilization_percent": None, "vram_used_bytes": None,
                               "hardware_decoder_utilization_percent": None,
                               "hardware_encoder_utilization_percent": None,
                               **_host_telemetry()}}
