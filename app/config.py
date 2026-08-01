from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import Any

import yaml

from .models import (
    AppConfig,
    CameraConfig,
    DetectionConfig,
    MetadataConfig,
    OutputConfig,
    PipelineConfig,
    ServiceConfig,
    TrackingConfig,
    VideoConfig,
)

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _finite_environment_fps(name: str, minimum: float, maximum: float,
                            environ: dict[str, str] | None = None) -> float | None:
    env = os.environ if environ is None else environ
    if name not in env:
        return None
    raw = env[name]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}. Expected a finite number from {minimum:g} to {maximum:g}.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}={raw!r}. Expected a finite number from {minimum:g} to {maximum:g}.")
    return value


def detection_backend_environment_override(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    if "DETECTION_BACKEND" not in env:
        return None
    value = str(env["DETECTION_BACKEND"]).strip().lower()
    if value not in {"pytorch", "onnx", "tensorrt", "auto"}:
        raise ValueError("Invalid DETECTION_BACKEND. Expected one of: pytorch, onnx, tensorrt, auto.")
    return value


def yolo_inference_fps_environment_override(environ: dict[str, str] | None = None) -> float | None:
    return _finite_environment_fps("YOLO_INFERENCE_FPS", .2, 25, environ)


def ai_capture_fps_environment_override(environ: dict[str, str] | None = None) -> float | None:
    return _finite_environment_fps("AI_CAPTURE_FPS", .2, 25, environ)


def detection_precision_environment_override(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    if "DETECTION_PRECISION" not in env:
        return None
    value = str(env["DETECTION_PRECISION"]).strip().lower()
    if value not in {"auto", "fp32", "fp16"}:
        raise ValueError("Invalid DETECTION_PRECISION. Expected one of: auto, fp32, fp16.")
    return value


def trt_engine_environment_override(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    value = str(env.get("TRT_ENGINE_MODEL", "")).strip()
    if not value:
        return None
    if "int8" in Path(value).name.lower():
        raise ValueError("TRT_ENGINE_MODEL must not select an INT8 engine")
    return value


def prediction_fps_environment_override(environ: dict[str, str] | None = None) -> float | None:
    """Parse the optional safe startup override once, without leaking env data."""
    env = os.environ if environ is None else environ
    if "BYTETRACK_PREDICTION_FPS" not in env:
        return None
    raw = env["BYTETRACK_PREDICTION_FPS"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid BYTETRACK_PREDICTION_FPS={raw!r}. Expected a number from 1 to 25."
        ) from None
    if not math.isfinite(value) or not 1 <= value <= 25:
        raise ValueError(
            f"Invalid BYTETRACK_PREDICTION_FPS={raw!r}. Expected a number from 1 to 25."
        )
    return value


def expand_environment(value: Any, environ: dict[str, str] | None = None) -> Any:
    """Recursively expand ${NAME} and ${NAME:-default}; unset required values fail fast."""
    env = os.environ if environ is None else environ
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            result = env.get(name, default)
            if result is None or result == "":
                raise ValueError(f"Environment variable {name} is required")
            return result
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [expand_environment(item, env) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item, env) for key, item in value.items()}
    return value


def required_camera_environment_variables(path: str | Path) -> tuple[str, ...]:
    """Return required URL variables for enabled cameras in one YAML config.

    This intentionally examines only each enabled camera's ``url`` field.  It
    supports arbitrary IDs and one or more ``${NAME}`` placeholders per URL;
    placeholders with a non-empty ``:-default`` are not required.
    """
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    variables: list[str] = []
    for camera in raw.get("cameras", []):
        if camera.get("enabled", True) is False:
            continue
        url = camera.get("url", "")
        if not isinstance(url, str):
            continue
        for match in _ENV.finditer(url):
            name, default = match.group(1), match.group(2)
            if default not in (None, ""):
                continue
            if name not in variables:
                variables.append(name)
    return tuple(variables)


def load_config(path: str | Path, environ: dict[str, str] | None = None) -> AppConfig:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    # A disabled input is never opened.  Replacing its URL before interpolation
    # lets an operator disable a camera without retaining a placeholder secret
    # in the environment.  Build a private copy instead of mutating the
    # structure returned by YAML parsing; all following precedence work uses
    # this one input to construct the frozen AppConfig.
    raw = {
        **loaded,
        "cameras": [
            {**camera, "url": ""} if camera.get("enabled", True) is False else dict(camera)
            for camera in loaded.get("cameras", [])
        ],
    }
    raw = expand_environment(raw, environ)
    cameras = tuple(CameraConfig(**camera) for camera in raw.get("cameras", []))
    if not cameras:
        raise ValueError("At least one camera must be configured")
    ids = [camera.id for camera in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("Camera IDs must be unique")
    for camera in cameras:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", camera.id):
            raise ValueError(f"Camera ID is not safe for a stream path: {camera.id}")
        if camera.rtsp_transport not in {"tcp", "udp", "auto"}:
            raise ValueError(f"Unsupported RTSP transport for {camera.id}: {camera.rtsp_transport}")
        if camera.detection_source not in {"local_mediamtx", "camera"}:
            raise ValueError(f"Unsupported detection source for {camera.id}: {camera.detection_source}")
    detection_raw = dict(raw.get("detection", {}))
    detection_raw["classes"] = tuple(detection_raw.get("classes", (0,)))
    if "fps_per_camera" in detection_raw and "target_fps_per_camera" not in detection_raw:
        detection_raw["target_fps_per_camera"] = detection_raw.pop("fps_per_camera")
    yaml_backend_configured = "backend" in detection_raw
    yaml_backend = str(detection_raw.get("backend", DetectionConfig.backend)).lower()
    backend_override = detection_backend_environment_override(environ)
    detection_raw.update({
        "backend": backend_override or yaml_backend,
        "yaml_backend": yaml_backend,
        "backend_environment_override": backend_override,
        "backend_source": "DETECTION_BACKEND" if backend_override is not None else
                          "config/cameras.yaml" if yaml_backend_configured else "built-in default",
    })
    yaml_fps_configured = "target_fps_per_camera" in detection_raw
    try:
        yaml_fps = float(detection_raw.get("target_fps_per_camera", DetectionConfig.target_fps_per_camera))
    except (TypeError, ValueError):
        raise ValueError("detection.target_fps_per_camera must be a number from 0.2 to 25") from None
    inference_override = yolo_inference_fps_environment_override(environ)
    detection_raw.update({
        "target_fps_per_camera": inference_override if inference_override is not None else yaml_fps,
        "yaml_target_fps_per_camera": yaml_fps,
        "inference_fps_environment_override": inference_override,
        "inference_fps_source": "YOLO_INFERENCE_FPS" if inference_override is not None else
                                "config/cameras.yaml" if yaml_fps_configured else "built-in default",
    })
    yaml_precision_configured = "precision" in detection_raw
    yaml_precision = str(detection_raw.get("precision", DetectionConfig.precision)).lower()
    precision_override = detection_precision_environment_override(environ)
    detection_raw.update({
        "precision": precision_override or yaml_precision,
        "precision_source": "DETECTION_PRECISION" if precision_override is not None else
                            "config/cameras.yaml" if yaml_precision_configured else "built-in default",
    })
    engine_override = trt_engine_environment_override(environ)
    if engine_override is not None:
        detection_raw["trt_engine_model"] = engine_override
    yaml_capture_configured = "capture_fps" in detection_raw
    yaml_capture = detection_raw.get("capture_fps", DetectionConfig.capture_fps)
    if yaml_capture is not None:
        try:
            yaml_capture = float(yaml_capture)
        except (TypeError, ValueError):
            raise ValueError("detection.capture_fps must be a number from 0.2 to 25 or null") from None
    capture_override = ai_capture_fps_environment_override(environ)
    configured_capture = capture_override if capture_override is not None else yaml_capture
    # A capture rate below the inference target would silently cap fresh
    # inferences.  Promote it to the requested target instead.
    effective_capture = max(float(configured_capture or 0), float(detection_raw["target_fps_per_camera"]))
    detection_raw.update({
        "capture_fps": effective_capture,
        "yaml_capture_fps": yaml_capture,
        "capture_fps_environment_override": capture_override,
        "capture_fps_source": "AI_CAPTURE_FPS" if capture_override is not None else
                              "config/cameras.yaml" if yaml_capture_configured else "built-in default",
    })
    if tuple(detection_raw["classes"]) != (0,):
        raise ValueError("Only COCO person class 0 is supported")
    if not detection_raw.get("latest_frame_only", True):
        raise ValueError("Detection must use latest_frame_only")
    if detection_raw.get("backend", "pytorch") not in {"auto", "pytorch", "onnx", "tensorrt", "onnxruntime", "openvino"}:
        raise ValueError(f"Unsupported detection backend: {detection_raw.get('backend')}")
    if detection_raw.get("device", "auto") not in {"auto", "cpu"}:
        raise ValueError(f"Unsupported detection device: {detection_raw.get('device')}")
    if detection_raw.get("precision", "auto") not in {"auto", "fp32", "fp16"}:
        raise ValueError(f"Unsupported detection precision: {detection_raw.get('precision')}")
    if detection_raw.get("precision") == "fp16" and detection_raw.get("backend") not in {"tensorrt", "auto"}:
        raise ValueError("detection.precision=fp16 requires detection.backend=tensorrt or auto")
    if detection_raw.get("batch_mode", "auto") not in {"auto", "serial", "batch", "opportunistic"}:
        raise ValueError(f"Unsupported detection batch mode: {detection_raw.get('batch_mode')}")
    if not 1 <= int(detection_raw.get("inference_workers", 1)) <= 16:
        raise ValueError("detection.inference_workers must be between 1 and 16")
    capture_fps = detection_raw.get("capture_fps")
    if capture_fps is not None and (not math.isfinite(float(capture_fps)) or not .2 <= float(capture_fps) <= 25):
        raise ValueError("detection.capture_fps must be a finite number from 0.2 to 25")
    if int(detection_raw.get("capture_max_dimension", 640)) < int(detection_raw.get("image_size", 640)):
        raise ValueError("detection.capture_max_dimension cannot be below image_size")
    if int(detection_raw.get("max_frame_age_ms", 750)) <= 0:
        raise ValueError("detection.max_frame_age_ms must be positive")
    if not math.isfinite(float(detection_raw["target_fps_per_camera"])) or not .2 <= float(detection_raw["target_fps_per_camera"]) <= 25:
        raise ValueError("detection.target_fps_per_camera must be between 0.2 and 25")
    tracking_raw = dict(raw.get("tracking", {}))
    yaml_prediction_fps_is_configured = "prediction_fps" in tracking_raw
    yaml_prediction_fps = tracking_raw.get("prediction_fps", TrackingConfig.prediction_fps)
    try:
        yaml_prediction_fps = float(yaml_prediction_fps)
    except (TypeError, ValueError):
        raise ValueError("tracking.prediction_fps must be a number from 1 to 25") from None
    override = prediction_fps_environment_override(environ)
    if override is not None:
        tracking_raw["prediction_fps"] = override
        prediction_source = "BYTETRACK_PREDICTION_FPS"
    else:
        tracking_raw["prediction_fps"] = yaml_prediction_fps
        prediction_source = "config/cameras.yaml" if yaml_prediction_fps_is_configured else "built-in default"
    tracking = TrackingConfig(
        **tracking_raw,
        yaml_prediction_fps=yaml_prediction_fps,
        prediction_fps_environment_override=override,
        prediction_fps_source=prediction_source,
    )
    if not tracking.enabled or tracking.tracker != "bytetrack":
        raise ValueError("Tracking must use enabled ByteTrack")
    thresholds = (
        tracking.track_low_thresh, tracking.track_high_thresh,
        tracking.new_track_thresh, tracking.match_thresh,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("ByteTrack thresholds must be between zero and one")
    if tracking.track_low_thresh >= tracking.track_high_thresh:
        raise ValueError("track_low_thresh must be below track_high_thresh")
    if not 1 <= tracking.track_buffer <= 5:
        raise ValueError("track_buffer must be between one and five update cycles")
    if tracking.hold_box_ms <= 0 or tracking.remove_track_ms < tracking.hold_box_ms:
        raise ValueError("remove_track_ms must be at least hold_box_ms")
    if not math.isfinite(tracking.prediction_fps) or not 1 <= tracking.prediction_fps <= 25:
        raise ValueError("prediction_fps must be between one and twenty-five")
    if not 0 <= tracking.prediction_deadzone_norm <= .05:
        raise ValueError("prediction_deadzone_norm must be between zero and .05")
    if not 0 < tracking.max_prediction_displacement_norm_per_second <= 1:
        raise ValueError("max_prediction_displacement_norm_per_second must be in (0, 1]")
    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    if pipeline.mode not in {"native", "legacy_annotated"}:
        raise ValueError(f"Unsupported pipeline mode: {pipeline.mode}")
    video = VideoConfig(**raw.get("video", {}))
    if video.mode not in {"direct_hevc", "diagnostic_transcode"}:
        raise ValueError(f"Unsupported video mode: {video.mode}")
    if video.mode == "direct_hevc" and video.allow_transcode_fallback:
        raise ValueError("direct_hevc does not permit an automatic transcode fallback")
    if video.h264_mode not in {"copy", "transcode", "direct"}:
        raise ValueError(f"Unsupported H.264 mode: {video.h264_mode}")
    metadata = MetadataConfig(**raw.get("metadata", {}))
    if metadata.transport != "websocket" or not metadata.path.startswith("/"):
        raise ValueError("Metadata must use an absolute WebSocket path")
    return AppConfig(
        cameras=cameras,
        detection=DetectionConfig(**detection_raw),
        tracking=tracking,
        output=OutputConfig(**raw.get("output", {})),
        video=video,
        pipeline=pipeline,
        metadata=metadata,
        service=ServiceConfig(**raw.get("service", {})),
    )
