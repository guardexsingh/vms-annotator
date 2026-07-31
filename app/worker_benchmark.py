"""Compare bounded CPU inference-worker layouts against live local HEVC paths."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
from pathlib import Path

import psutil

from .config import load_config, required_camera_environment_variables


LAYOUTS = {
    "one_serial_4t": {"workers": 1, "threads": 4, "batch": False},
    "one_batch_4t": {"workers": 1, "threads": 4, "batch": True},
    "two_serial_2t": {"workers": 2, "threads": 2, "batch": False},
    "three_serial_1t": {"workers": 3, "threads": 1, "batch": False},
    "three_serial_2t": {"workers": 3, "threads": 2, "batch": False},
}


def _worker(camera_ids: list[str], model: str, image_size: int, confidence: float,
            threads: int, batched: bool, duration: float, start: mp.Event,
            ready: mp.Queue, output: mp.Queue) -> None:
    os.environ.update({
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "PYTHONNOUSERSITE": "1",
    })
    try:
        from .decoder import CameraDecoder
        from .models import Frame
        from .yolo_detector import YoloPersonDetector

        detector = YoloPersonDetector(model, image_size, confidence,
                                      torch_threads=threads, torch_interop_threads=1)
        detail = detector.warmup()
        decoders = {
            camera_id: CameraDecoder(
                f"rtsp://127.0.0.1:18554/live/{camera_id}", camera_id,
                prefer_hardware=False, rtsp_transport="tcp",
                max_dimension=image_size, sample_fps=5.0,
            )
            for camera_id in camera_ids
        }
        for decoder in decoders.values():
            decoder.probe()
            decoder.start()
        deadline = time.monotonic() + 10
        initial = {}
        while time.monotonic() < deadline and len(initial) < len(camera_ids):
            for camera_id, decoder in decoders.items():
                frame = decoder.get_latest_frame()
                if frame is not None:
                    initial[camera_id] = frame
            time.sleep(.005)
        if len(initial) != len(camera_ids):
            raise RuntimeError("AI capture did not become ready")
        ready.put({"pid": os.getpid(), "detail": detail})
        start.wait(60)
        run_started = time.monotonic()
        counts = {camera_id: 0 for camera_id in camera_ids}
        useful = counts.copy()
        stale = counts.copy()
        ages: dict[str, list[float]] = {camera_id: [] for camera_id in camera_ids}
        inference_ms: list[float] = []
        previous_sequence = {camera_id: -1 for camera_id in camera_ids}
        cursor = 0
        while time.monotonic() - run_started < duration:
            available: list[tuple[str, Frame]] = []
            order = camera_ids if batched else [camera_ids[cursor % len(camera_ids)]]
            cursor += 1
            for camera_id in order:
                frame = decoders[camera_id].get_latest_frame()
                if frame is not None and frame.sequence != previous_sequence[camera_id]:
                    previous_sequence[camera_id] = frame.sequence
                    available.append((camera_id, frame))
            if not available:
                time.sleep(.002)
                continue
            infer_started = time.monotonic()
            results = detector.detect_batch(available) if batched else [
                detector.detect(camera_id, frame) for camera_id, frame in available
            ]
            inference_ms.append((time.monotonic() - infer_started) * 1000)
            for result in results:
                age = (result.completed_at - result.source_captured_at) * 1000
                counts[result.camera_id] += 1
                ages[result.camera_id].append(age)
                if age <= 750:
                    useful[result.camera_id] += 1
                else:
                    stale[result.camera_id] += 1
        elapsed = time.monotonic() - run_started
        for decoder in decoders.values():
            decoder.stop()
        detector.close()
        output.put({
            "pid": os.getpid(),
            "elapsed_seconds": elapsed,
            "counts": counts,
            "useful": useful,
            "stale": stale,
            "age_ms": ages,
            "inference_ms": inference_ms,
        })
    except BaseException as error:
        ready.put({"pid": os.getpid(), "error": f"{type(error).__name__}: {error}"})


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]


def benchmark_layout(config, name: str, duration: float) -> dict[str, object]:
    spec = LAYOUTS[name]
    camera_ids = [camera.id for camera in config.cameras if camera.enabled and camera.detection_enabled]
    assignments = [camera_ids[index::spec["workers"]] for index in range(spec["workers"])]
    context = mp.get_context("spawn")
    start, ready, output = context.Event(), context.Queue(), context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(assignment, config.detection.model, config.detection.image_size,
                  config.detection.confidence, spec["threads"], spec["batch"],
                  duration, start, ready, output),
        )
        for assignment in assignments if assignment
    ]
    for process in processes:
        process.start()
    ready_rows = []
    for _ in processes:
        try:
            ready_rows.append(ready.get(timeout=90))
        except queue.Empty:
            ready_rows.append({"error": "worker readiness timeout"})
    errors = [row["error"] for row in ready_rows if "error" in row]
    if errors:
        for process in processes:
            process.terminate()
            process.join(5)
        raise RuntimeError("; ".join(errors))
    monitors = [psutil.Process(process.pid) for process in processes]
    for monitor in monitors:
        monitor.cpu_percent(None)
    start.set()
    host_cpu: list[float] = []
    aggregate_cpu: list[float] = []
    aggregate_rss: list[int] = []
    monitor_deadline = time.monotonic() + duration + 5
    while any(process.is_alive() for process in processes) and time.monotonic() < monitor_deadline:
        time.sleep(.25)
        host_cpu.append(psutil.cpu_percent(None))
        cpu, rss = 0.0, 0
        for monitor in monitors:
            try:
                family = [monitor, *monitor.children(recursive=True)]
                cpu += sum(process.cpu_percent(None) for process in family)
                rss += sum(process.memory_info().rss for process in family)
            except psutil.Error:
                continue
        aggregate_cpu.append(cpu)
        aggregate_rss.append(rss)
    rows = []
    for _ in processes:
        try:
            rows.append(output.get(timeout=10))
        except queue.Empty:
            rows.append({"error": "worker result timeout"})
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    if any("error" in row for row in rows):
        raise RuntimeError("; ".join(row["error"] for row in rows if "error" in row))
    counts = {camera_id: 0 for camera_id in camera_ids}
    useful = counts.copy()
    stale = counts.copy()
    ages = {camera_id: [] for camera_id in camera_ids}
    inference = []
    elapsed = max(float(row["elapsed_seconds"]) for row in rows)
    for row in rows:
        inference.extend(row["inference_ms"])
        for camera_id in row["counts"]:
            counts[camera_id] += row["counts"][camera_id]
            useful[camera_id] += row["useful"][camera_id]
            stale[camera_id] += row["stale"][camera_id]
            ages[camera_id].extend(row["age_ms"][camera_id])
    return {
        "layout": name,
        "workers": len(processes),
        "threads_per_worker": spec["threads"],
        "batch": spec["batch"],
        "model_copies": len(processes),
        "elapsed_seconds": elapsed,
        "raw_fps": {camera_id: counts[camera_id] / elapsed for camera_id in camera_ids},
        "useful_fps": {camera_id: useful[camera_id] / elapsed for camera_id in camera_ids},
        "stale_results": stale,
        "aggregate_raw_fps": sum(counts.values()) / elapsed,
        "aggregate_useful_fps": sum(useful.values()) / elapsed,
        "result_age_p50_ms": {camera_id: _percentile(ages[camera_id], .5) for camera_id in camera_ids},
        "result_age_p95_ms": {camera_id: _percentile(ages[camera_id], .95) for camera_id in camera_ids},
        "call_p50_ms": _percentile(inference, .5),
        "call_p95_ms": _percentile(inference, .95),
        "host_cpu_percent": statistics.mean(host_cpu) if host_cpu else None,
        "worker_family_cpu_percent": statistics.mean(aggregate_cpu) if aggregate_cpu else None,
        "worker_family_rss_peak_bytes": max(aggregate_rss) if aggregate_rss else None,
        "worker_details": ready_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--layout", choices=tuple(LAYOUTS) + ("all",), default="all")
    args = parser.parse_args()
    environment = {name: "rtsp://redacted"
                   for name in required_camera_environment_variables(args.config)}
    config = load_config(args.config, environment)
    layouts = list(LAYOUTS) if args.layout == "all" else [args.layout]
    for layout in layouts:
        print(json.dumps(benchmark_layout(config, layout, max(10.0, args.duration)),
                         sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
