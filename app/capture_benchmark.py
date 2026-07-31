"""Measure the AI-only decode/output branch without loading a detector."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from .config import load_config, required_camera_environment_variables
from .decoder import CameraDecoder
from .latest_frame import LatestFrame
from .models import Frame


STRATEGIES = {
    "continuous-scaled": {"sample_fps": None, "max_dimension": 640, "hardware": False},
    "sampled-scaled": {"sample_fps": 5.0, "max_dimension": 640, "hardware": False},
    "continuous-full": {"sample_fps": None, "max_dimension": None, "hardware": False},
    "sampled-full": {"sample_fps": 5.0, "max_dimension": None, "hardware": False},
    "hardware-sampled": {"sample_fps": 5.0, "max_dimension": 640, "hardware": True},
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))]


def _packet_rate(camera_id: str, seconds: int, rtsp_port: int) -> float | None:
    command = [
        "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
        "-read_intervals", f"%+{seconds}", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "json", f"rtsp://127.0.0.1:{rtsp_port}/live/{camera_id}",
    ]
    try:
        payload = json.loads(subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=seconds + 10,
        ).stdout)
        packets = int(payload["streams"][0]["nb_read_packets"])
        return packets / seconds
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def benchmark(config_path: Path, strategy: str, duration: float,
              consume_fps: float = 5.0, rtsp_port: int = 18554) -> dict[str, object]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    environment = {name: "rtsp://redacted"
                   for name in required_camera_environment_variables(config_path)}
    config = load_config(config_path, environment)
    cameras = [camera for camera in config.cameras if camera.enabled]
    options = STRATEGIES[strategy]
    slots = {camera.id: LatestFrame[Frame]() for camera in cameras}
    decoders: dict[str, CameraDecoder] = {}
    state = {
        camera.id: {"written": 0, "consumed": 0, "ages_ms": [], "sequences": set()}
        for camera in cameras
    }
    lock = threading.Lock()

    for camera in cameras:
        def on_frame(frame: Frame, _replaced: bool, camera_id: str = camera.id) -> None:
            with lock:
                state[camera_id]["written"] += 1

        decoder = CameraDecoder(
            f"rtsp://127.0.0.1:{rtsp_port}/live/{camera.id}", camera.id,
            prefer_hardware=bool(options["hardware"]), rtsp_transport="tcp",
            frame_slot=slots[camera.id], on_frame=on_frame,
            max_dimension=options["max_dimension"], sample_fps=options["sample_fps"],
        )
        decoder.probe()
        decoders[camera.id] = decoder

    if options["hardware"]:
        unavailable = [camera_id for camera_id, decoder in decoders.items()
                       if decoder.info is None or decoder.info.software_fallback]
        if unavailable:
            return {
                "strategy": strategy, "available": False,
                "reason": "NVDEC device is not accessible",
                "cameras": unavailable,
            }

    with ThreadPoolExecutor(max_workers=max(1, len(cameras))) as executor:
        packet_rates = dict(zip(
            [camera.id for camera in cameras],
            executor.map(lambda item: _packet_rate(item.id, 3, rtsp_port), cameras),
        ))

    for decoder in decoders.values():
        decoder.start()
    deadline = time.monotonic() + duration
    interval = 1.0 / consume_fps
    next_consume = time.monotonic()
    process_samples: dict[str, list[float]] = {camera.id: [] for camera in cameras}
    host_samples: list[float] = []
    process_handles: dict[str, psutil.Process] = {}
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            for camera_id, decoder in decoders.items():
                if decoder.process_pid and camera_id not in process_handles:
                    process_handles[camera_id] = psutil.Process(decoder.process_pid)
                    process_handles[camera_id].cpu_percent(None)
                process = process_handles.get(camera_id)
                if process:
                    try:
                        process_samples[camera_id].append(process.cpu_percent(None))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            host_samples.append(psutil.cpu_percent(None))
            if now >= next_consume:
                while next_consume <= now:
                    next_consume += interval
                for camera_id, slot in slots.items():
                    frame = slot.take()
                    if frame is None:
                        continue
                    with lock:
                        if frame.sequence not in state[camera_id]["sequences"]:
                            state[camera_id]["sequences"].add(frame.sequence)
                            state[camera_id]["consumed"] += 1
                            state[camera_id]["ages_ms"].append(
                                (time.monotonic() - frame.captured_at) * 1000
                            )
            time.sleep(0.1)
    finally:
        for decoder in decoders.values():
            decoder.stop()

    rows = {}
    for camera in cameras:
        decoder = decoders[camera.id]
        assert decoder.info is not None
        width, height = decoder._output_size()
        values = state[camera.id]
        output_fps = values["written"] / duration
        rows[camera.id] = {
            "input_packet_fps": packet_rates[camera.id],
            "decoder_input_fps": decoder.info.fps,
            "decoder_output_fps": output_fps,
            "frames_written_fps": output_fps,
            "unique_frames_consumed_fps": values["consumed"] / duration,
            "frames_overwritten": slots[camera.id].replaced,
            "decoder_cpu_percent": round(statistics.mean(process_samples[camera.id]), 2)
                                   if process_samples[camera.id] else None,
            "pixel_conversion_cpu": "included in decoder process CPU",
            "output_resolution": f"{width}x{height}",
            "bgr_pipe_mib_s": round(width * height * 3 * output_fps / 1048576, 2),
            "frame_age_p50_ms": _percentile(values["ages_ms"], .5),
            "frame_age_p95_ms": _percentile(values["ages_ms"], .95),
            "decoder": decoder.info.decoder,
            "software_fallback": decoder.info.software_fallback,
            "failure": type(decoder.failure).__name__ if decoder.failure else None,
        }
    return {
        "strategy": strategy,
        "available": True,
        "duration_seconds": duration,
        "consume_fps": consume_fps,
        "host_cpu_percent": round(statistics.mean(host_samples), 2) if host_samples else None,
        "cameras": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/cameras.relay.yaml"))
    parser.add_argument("--strategy", choices=tuple(STRATEGIES), required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config, args.strategy, max(2.0, args.duration)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
