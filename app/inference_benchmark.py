"""Reproducible in-memory YOLO benchmark using representative live frames."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil

from .config import load_config, required_camera_environment_variables
from .decoder import CameraDecoder


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))]


def _thermal() -> dict[str, float]:
    values: dict[str, float] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text().strip()
            value = float((zone / "temp").read_text().strip())
            values[name] = value / 1000 if value > 1000 else value
        except (OSError, TypeError, ValueError):
            continue
    return values


def _capture_frames(config, rtsp_port: int = 18554) -> dict[str, object]:
    frames: dict[str, object] = {}
    for camera in config.cameras:
        if not camera.enabled:
            continue
        decoder = CameraDecoder(
            f"rtsp://127.0.0.1:{rtsp_port}/live/{camera.id}",
            camera.id, prefer_hardware=False, rtsp_transport="tcp",
            max_dimension=config.detection.image_size,
        )
        decoder.probe()
        decoder.start()
        deadline = time.monotonic() + 8
        try:
            frame = None
            while time.monotonic() < deadline and frame is None:
                frame = decoder.get_latest_frame()
                if frame is None:
                    time.sleep(.02)
            if frame is None:
                raise RuntimeError(f"No representative frame for {camera.id}")
            frames[camera.id] = frame.image.copy()
        finally:
            decoder.stop()
    return frames


class ResourceMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.cpu: list[float] = []
        self.host_cpu: list[float] = []
        self.rss: list[int] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.process.cpu_percent(None)
        while not self.stop_event.wait(.1):
            self.cpu.append(self.process.cpu_percent(None))
            self.host_cpu.append(psutil.cpu_percent(None))
            self.rss.append(self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=1)


def benchmark(config_path: Path, iterations: int, warm_iterations: int,
              threads: int) -> dict[str, object]:
    environment = {name: "rtsp://redacted"
                   for name in required_camera_environment_variables(config_path)}
    config = load_config(config_path, environment)
    frames = _capture_frames(config)
    camera_ids = list(frames)

    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    load_started = time.monotonic()
    model = YOLO(config.detection.model)
    model_load_ms = (time.monotonic() - load_started) * 1000
    predictor_started = time.monotonic()
    model.predict(
        frames[camera_ids[0]], imgsz=config.detection.image_size,
        conf=config.detection.confidence, classes=list(config.detection.classes),
        device="cpu", verbose=False,
    )
    predictor_initialization_ms = (time.monotonic() - predictor_started) * 1000
    # Ultralytics configures a host-wide default when its predictor is first
    # created. Re-apply limits after that one-time setup and before all warm-up
    # and measured inference.
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    rows = []
    baseline_detections: dict[str, object] = {}

    for batch_size in (1, len(camera_ids)):
        selected = camera_ids[:batch_size]
        images = [frames[camera_id] for camera_id in selected]
        source = images[0] if batch_size == 1 else images
        warm_times = []
        for _ in range(warm_iterations):
            started = time.monotonic()
            model.predict(source, imgsz=config.detection.image_size,
                          conf=config.detection.confidence,
                          classes=list(config.detection.classes), device="cpu", verbose=False)
            warm_times.append((time.monotonic() - started) * 1000)
        latencies: list[float] = []
        preprocess: list[float] = []
        inference: list[float] = []
        postprocess: list[float] = []
        context_before = psutil.Process().num_ctx_switches()
        thermal_before = _thermal()
        with ResourceMonitor() as resources:
            started_all = time.monotonic()
            last_results = []
            for _ in range(iterations):
                started = time.monotonic()
                last_results = model.predict(
                    source, imgsz=config.detection.image_size,
                    conf=config.detection.confidence,
                    classes=list(config.detection.classes), device="cpu", verbose=False,
                )
                latencies.append((time.monotonic() - started) * 1000)
                for result in last_results:
                    speed = result.speed or {}
                    preprocess.append(float(speed.get("preprocess", 0)))
                    inference.append(float(speed.get("inference", 0)))
                    postprocess.append(float(speed.get("postprocess", 0)))
            elapsed = time.monotonic() - started_all
        context_after = psutil.Process().num_ctx_switches()
        if not baseline_detections and batch_size == len(camera_ids):
            for camera_id, result in zip(selected, last_results):
                boxes = result.boxes
                baseline_detections[camera_id] = {
                    "person_count": 0 if boxes is None else len(boxes),
                    "confidences": [] if boxes is None else
                        [round(float(value), 4) for value in boxes.conf.cpu().tolist()],
                }
        rows.append({
            "backend": "pytorch",
            "device": "cpu",
            "precision": "fp32",
            "threads": threads,
            "interop_threads": 1,
            "batch_size": batch_size,
            "input_size": config.detection.image_size,
            "warmup_p50_ms": _percentile(warm_times, .5),
            "batch_p50_ms": _percentile(latencies, .5),
            "batch_p95_ms": _percentile(latencies, .95),
            "batch_max_ms": max(latencies),
            "per_image_p50_ms": _percentile(latencies, .5) / batch_size,
            "images_per_second": iterations * batch_size / elapsed,
            "preprocess_p50_ms": _percentile(preprocess, .5),
            "inference_p50_ms": _percentile(inference, .5),
            "postprocess_p50_ms": _percentile(postprocess, .5),
            "process_cpu_percent": statistics.mean(resources.cpu) if resources.cpu else None,
            "host_cpu_percent": statistics.mean(resources.host_cpu) if resources.host_cpu else None,
            "rss_peak_bytes": max(resources.rss) if resources.rss else psutil.Process().memory_info().rss,
            "context_switches": {
                "voluntary": context_after.voluntary - context_before.voluntary,
                "involuntary": context_after.involuntary - context_before.involuntary,
            },
            "thermal_before_c": thermal_before,
            "thermal_after_c": _thermal(),
        })
    return {
        "model": config.detection.model,
        "model_load_ms": model_load_ms,
        "predictor_initialization_ms": predictor_initialization_ms,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "camera_ids": camera_ids,
        "representative_frames_persisted": False,
        "baseline_detections": baseline_detections,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warm-iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 6), required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config, max(5, args.iterations),
                               max(1, args.warm_iterations), args.threads), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
