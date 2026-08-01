from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StreamInfo:
    codec: str
    width: int
    height: int
    fps: float
    pixel_format: str | None = None
    decoder: str = "software"
    software_fallback: bool = False


@dataclass
class Frame:
    image: np.ndarray
    captured_at: float
    sequence: int
    source_width: int = 0
    source_height: int = 0


@dataclass(frozen=True)
class Detection:
    camera_id: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class DetectionResult:
    camera_id: str
    detections: tuple[Detection, ...]
    source_captured_at: float
    inference_started_at: float
    completed_at: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    source_sequence: int = 0
    frame_width: int = 0
    frame_height: int = 0


@dataclass(frozen=True)
class TrackedPerson:
    camera_id: str
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    track_state: str
    last_confirmed_at: float
    class_id: int = 0
    source: str = "yolo"
    predicted: bool = False


@dataclass(frozen=True)
class TrackResult:
    camera_id: str
    tracks: tuple[TrackedPerson, ...]
    source_captured_at: float
    inference_started_at: float
    completed_at: float
    source_sequence: int
    frame_width: int
    frame_height: int
    active_track_count: int
    lost_track_count: int
    removed_track_count: int
    predicted_track_count: int = 0
    is_prediction: bool = False
    yolo_completed_at: float | None = None


@dataclass
class CameraMetrics:
    camera_id: str
    codec: str = "unknown"
    source_codec: str = "unknown"
    mediamtx_codec: str = "unknown"
    webrtc_codec: str = "unknown"
    video_path: str = "direct"
    transcoding: bool = False
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    mediamtx_fps: float = 0.0
    browser_received_fps: float | None = None
    browser_decoded_fps: float | None = None
    browser_frames_dropped: int | None = None
    browser_jitter_buffer_delay_ms: float | None = None
    glass_to_glass_latency_ms: float | None = None
    decoded_fps: float = 0.0
    compositor_input_fps: float = 0.0
    composited_fps: float = 0.0
    encoded_fps: float = 0.0
    published_fps: float = 0.0
    output_fps: float = 0.0
    requested_inference_fps: float = 1.0
    completed_inference_fps: float = 0.0
    completed_inferences: int = 0
    ai_capture_fps: float = 0.0
    ai_decoder_input_fps: float = 0.0
    ai_frames_consumed_fps: float = 0.0
    ai_frame_age_ms: float | None = None
    ai_output_width: int = 0
    ai_output_height: int = 0
    ai_pixel_rate_mib_s: float = 0.0
    ai_capture_status: str = "disabled"
    ai_capture_backend: str = "none"
    ai_frames_replaced: int = 0
    video_status: str = "offline"
    websocket_subscribers: int = 0
    metadata_messages_sent: int = 0
    metadata_messages_replaced: int = 0
    latest_inference_latency_ms: float | None = None
    capture_to_inference_start_ms: float | None = None
    capture_to_result_latency_ms: float | None = None
    result_interval_ms: float | None = None
    detection_age_ms: float | None = None
    person_count: int = 0
    inference_frames_replaced: int = 0
    stale_results: int = 0
    stale_input_frames: int = 0
    active_track_count: int = 0
    lost_track_count: int = 0
    removed_track_count: int = 0
    last_confirmed_age_ms: float | None = None
    requested_tracker_fps: float = 5.0
    actual_tracker_fps: float = 0.0
    configured_bytetrack_prediction_fps: float = 5.0
    actual_bytetrack_prediction_fps: float = 0.0
    yolo_updates_total: int = 0
    prediction_updates_total: int = 0
    prediction_ticks_skipped: int = 0
    predicted_track_count: int = 0
    prediction_compute_p50_ms: float | None = None
    prediction_compute_p95_ms: float | None = None
    prediction_cpu_p50_ms: float | None = None
    prediction_cpu_p95_ms: float | None = None
    yolo_compute_cpu_p50_ms: float | None = None
    yolo_compute_cpu_p95_ms: float | None = None
    ai_decoder_cpu_percent: float | None = None
    metadata_publish_fps: float = 0.0
    track_id_switches: int | None = None
    decode_failures: int = 0
    reconnections: int = 0
    encoder_failures: int = 0
    publish_failures: int = 0
    decode_queue_depth: int = 0
    encoder_queue_depth: int = 0
    dropped_frames: int = 0
    decoder_command: str = ""
    encoder_command: str = ""
    timestamp_source: str = "local_monotonic_decode_time"
    compositor_ms: float | None = None
    encoder_write_ms: float | None = None
    raw_frames_through_python: bool = False
    online: bool = False
    decoder: str = "unknown"
    encoder: str = "unknown"


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    url: str
    enabled: bool = True
    rtsp_transport: str = "tcp"
    detection_enabled: bool = True
    detection_source: str = "local_mediamtx"


@dataclass(frozen=True)
class DetectionConfig:
    enabled: bool = True
    model: str = "yolo11n.pt"
    image_size: int = 640
    confidence: float = 0.4
    classes: tuple[int, ...] = (0,)
    target_fps_per_camera: float = 1.0
    yaml_target_fps_per_camera: float = 1.0
    inference_fps_environment_override: float | None = None
    inference_fps_source: str = "built-in default"
    result_ttl_ms: int = 2000
    latest_frame_only: bool = True
    backend: str = "pytorch"
    yaml_backend: str = "pytorch"
    backend_environment_override: str | None = None
    backend_source: str = "built-in default"
    onnx_model: str | None = None
    trt_engine_model: str | None = None
    precision_source: str = "built-in default"
    allow_backend_fallback: bool = True
    device: str = "auto"
    precision: str = "auto"
    batch_mode: str = "auto"
    inference_workers: int = 1
    torch_threads: int = 0
    torch_interop_threads: int = 1
    capture_fps: float | None = 1.0
    yaml_capture_fps: float | None = 1.0
    capture_fps_environment_override: float | None = None
    capture_fps_source: str = "built-in default"
    capture_max_dimension: int = 640
    max_frame_age_ms: int = 1500
    max_batch_wait_ms: int = 20

    @property
    def fps_per_camera(self) -> float:
        """Compatibility name used by the existing benchmark and scheduler."""
        return self.target_fps_per_camera


@dataclass(frozen=True)
class MetadataConfig:
    transport: str = "websocket"
    path: str = "/ws/detections"


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    tracker: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 2
    fuse_score: bool = True
    hold_box_ms: int = 1500
    remove_track_ms: int = 2000
    prediction_fps: float = 5.0
    yaml_prediction_fps: float = 5.0
    prediction_fps_environment_override: float | None = None
    prediction_fps_source: str = "built-in default"
    prediction_deadzone_norm: float = 0.002
    max_prediction_displacement_norm_per_second: float = 0.25


@dataclass(frozen=True)
class OutputConfig:
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    low_latency: bool = True
    fps: float | None = None


@dataclass(frozen=True)
class VideoConfig:
    """Selects the video transport independently of the metadata path."""
    mode: str = "direct_hevc"
    allow_transcode_fallback: bool = False
    h264_mode: str = "copy"


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "native"


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 18080
    public_host: str = "127.0.0.1"
    whep_port: int = 18889


@dataclass(frozen=True)
class AppConfig:
    cameras: tuple[CameraConfig, ...]
    detection: DetectionConfig
    tracking: TrackingConfig
    output: OutputConfig
    video: VideoConfig
    pipeline: PipelineConfig
    metadata: MetadataConfig
    service: ServiceConfig
