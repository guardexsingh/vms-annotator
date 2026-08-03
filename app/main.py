from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

from .camera_worker import CameraWorker
from .compositor import Compositor
from .config import load_config, required_camera_environment_variables
from .direct_video import load_validation
from .detection_runtime import DetectionRuntime
from .detection_store import DetectionStore
from .health import HealthServer
from .metadata import MetadataHub
from .metrics import Metrics
from .native_relay import NativeRelayPipeline
from .stream_probe import probe_stream
from .detection_backend import select_backend
from .yolo_detector import YoloPersonDetector  # legacy test/extension compatibility

LOG = logging.getLogger(__name__)


def build_video_components(config, metrics: Metrics) -> tuple[list[NativeRelayPipeline], list[CameraWorker]]:
    """Select video components without consulting detector state."""
    if config.runtime.mode == "metadata_only":
        return ([], [])
    if config.video.mode == "direct_hevc":
        return ([], [])
    if config.video.h264_mode == "direct":
        return ([], [])
    if config.pipeline.mode == "native":
        return ([NativeRelayPipeline(camera, config.output, metrics, config.video)
                 for camera in config.cameras if camera.enabled], [])
    compositor = Compositor(DetectionStore(config.detection.result_ttl_ms))
    return ([], [CameraWorker(camera, config.output, None, compositor, metrics)
                 for camera in config.cameras if camera.enabled])


def probe_only(config) -> int:
    print(f"{'Camera':<10} {'Codec':<8} {'Resolution':<13} {'FPS':<8} Selected decoder")
    from .decoder import select_decoder
    for camera in config.cameras:
        if not camera.enabled:
            continue
        try:
            info = probe_stream(camera.url, camera.rtsp_transport)
            decoder, fallback = select_decoder(info.codec)
            suffix = " (software fallback)" if fallback else ""
            print(f"{camera.id:<10} {info.codec:<8} {info.width}x{info.height:<8} {info.fps:<8.2f} {decoder}{suffix}")
        except Exception as error:
            print(f"{camera.id:<10} ERROR: {type(error).__name__}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--video-validation", default="run/video-validation.json")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--detector-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.detector_only:
        # The detector benchmark has no camera I/O and must be runnable before
        # an operator enters every credential.  These placeholders are never
        # opened or printed.
        environment = {**os.environ, **{name: "rtsp://redacted"
                                        for name in required_camera_environment_variables(args.config)}}
        config = load_config(args.config, environment)
    else:
        config = load_config(args.config)
    LOG.info("Detection backend: %s; source: %s",
             config.detection.backend, config.detection.backend_source)
    LOG.info("Runtime mode: %s; configured cameras: %s", config.runtime.mode,
             ", ".join(camera.id for camera in config.cameras if camera.enabled))
    LOG.info("Detector assets: model=%s engine=%s precision=%s",
             Path(config.detection.model).name,
             Path(config.detection.trt_engine_model or "").name or "default",
             config.detection.precision)
    LOG.info("YOLO inference FPS: %.1f; source: %s; interval: %.1f ms",
             config.detection.target_fps_per_camera, config.detection.inference_fps_source,
             1000.0 / config.detection.target_fps_per_camera)
    LOG.info("AI capture FPS: %.1f; source: %s; interval: %.1f ms",
             config.detection.capture_fps, config.detection.capture_fps_source,
             1000.0 / config.detection.capture_fps)
    LOG.info("Detection precision: %s; source: %s",
             config.detection.precision, config.detection.precision_source)
    LOG.info(
        "ByteTrack prediction FPS: %.1f; source: %s; interval: %.1f ms",
        config.tracking.prediction_fps,
        config.tracking.prediction_fps_source,
        1000.0 / config.tracking.prediction_fps,
    )
    if args.probe_only:
        return probe_only(config)
    if args.detector_only:
        if not config.detection.enabled:
            print({"detector": "disabled"})
            return 0
        detector, selection = select_backend(config.detection)
        try:
            details = detector.warmup()
        except Exception as error:
            LOG.error("Detector preflight failed: %s", error)
            return 1
        samples: list[float] = []
        from .models import Frame
        import numpy as np
        for _ in range(20):
            started = time.monotonic()
            detector.detect("benchmark", Frame(np.zeros((config.detection.image_size, config.detection.image_size, 3),
                                               dtype=np.uint8), started, 0))
            samples.append((time.monotonic() - started) * 1000)
        samples.sort()
        print({"model_load_ms": detector.model_initialization_ms, "warmup_ms": detector.warmup_ms,
               "inference_p50_ms": samples[len(samples) // 2], "inference_p95_ms": samples[int(len(samples) * .95) - 1],
               "device": selection.device, **details})
        return 0

    metrics = Metrics(config)
    enabled = tuple(camera for camera in config.cameras if camera.enabled)
    if config.video.mode == "direct_hevc" and config.runtime.mode != "metadata_only":
        try:
            validation = load_validation(Path(args.video_validation))
        except (OSError, ValueError) as error:
            LOG.error("Direct HEVC validation is unavailable: %s", type(error).__name__)
            return 1
        missing = [camera.id for camera in enabled if camera.id not in validation]
        if missing:
            LOG.error("Direct HEVC validation is missing camera IDs: %s", ", ".join(missing))
            return 1
    for camera in config.cameras:
        metric = metrics.camera(camera.id)
        metric.ai_capture_status = "disabled"
        if config.runtime.mode == "metadata_only" and camera.enabled:
            metric.video_status, metric.video_path = "metadata_only", "metadata_only"
            metric.decoder, metric.encoder = "none", "none"
            metric.raw_frames_through_python = False
        elif config.video.mode == "direct_hevc" and camera.enabled:
            validated = validation[camera.id]
            metric.codec = metric.source_codec = str(validated["source_codec"])
            metric.mediamtx_codec = str(validated["mediamtx_codec"])
            metric.width, metric.height = int(validated["width"]), int(validated["height"])
            metric.source_fps = float(validated["source_fps"])
            metric.mediamtx_fps = float(validated["mediamtx_fps"])
            metric.video_status, metric.online = "ready", True
            metric.video_path, metric.transcoding = "direct", False
            metric.decoder, metric.encoder = "none", "none"
            metric.raw_frames_through_python = False
    hub = MetadataHub((camera.id for camera in enabled), metrics, config.tracking.remove_track_ms)
    # Detector state is intentionally absent from video selection.
    relays, workers = build_video_components(config, metrics)
    detection_runtime = DetectionRuntime(config, metrics, hub) if config.detection.enabled else None
    if not detection_runtime:
        metrics.set_detector("disabled")
    server = HealthServer(
        config.service.host, config.service.port, metrics, Path(__file__).parents[1] / "web",
        metadata_hub=hub, metadata_path=config.metadata.path, cameras=config.cameras,
        ttl_ms=config.tracking.remove_track_ms, whep_port=config.service.whep_port,
        video_mode=config.video.mode, runtime_mode=config.runtime.mode,
        detection_runtime=detection_runtime,
        tracking_config=config.tracking, effective_config=config,
    )
    stop = threading.Event()
    def request_stop(*_): stop.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for worker in workers:
        metrics.camera(worker.config.id).requested_inference_fps = config.detection.target_fps_per_camera
        worker.start()
    for relay in relays:
        relay.start()
    hub.start()
    if detection_runtime:
        detection_runtime.start()
    LOG.info("Metadata service listening on %s:%s", config.service.host, config.service.port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stop.wait()
    if detection_runtime:
        detection_runtime.stop()
    hub.stop()
    server.stop()
    for worker in workers: worker.stop()
    for relay in relays: relay.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
