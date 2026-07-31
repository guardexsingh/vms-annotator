"""Private MediaMTX direct-HEVC configuration and validation.

Camera URLs are written only to a mode-0600 runtime file. Public diagnostics
contain camera IDs and stream properties, never source URLs or credentials.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .models import AppConfig
from .stream_probe import probe_stream


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def generate_direct_config(config_path: Path, base_path: Path, output_path: Path) -> AppConfig:
    config = load_config(config_path)
    if config.video.mode != "direct_hevc":
        raise ValueError("Direct MediaMTX generation requires video.mode: direct_hevc")
    with base_path.open(encoding="utf-8") as handle:
        media: dict[str, Any] = yaml.safe_load(handle) or {}
    paths: dict[str, dict[str, object]] = {}
    for camera in config.cameras:
        if not camera.enabled:
            continue
        source: dict[str, object] = {
            "source": camera.url,
            "sourceOnDemand": True,
        }
        if camera.rtsp_transport != "auto":
            source["rtspTransport"] = camera.rtsp_transport
        paths[f"live/{camera.id}"] = source
    media["paths"] = paths
    _write_private(output_path, yaml.safe_dump(media, sort_keys=False))
    return config


def _verify_camera(camera, rtsp_port: int, timeout_seconds: float) -> dict[str, object]:
    try:
        source = probe_stream(camera.url, camera.rtsp_transport, timeout_seconds)
    except Exception as error:
        raise RuntimeError(f"{camera.id}: source probe failed ({type(error).__name__})") from None
    if source.codec != "hevc":
        raise ValueError(f"{camera.id}: source codec is {source.codec}, expected hevc")
    local_url = f"rtsp://127.0.0.1:{rtsp_port}/live/{camera.id}"
    try:
        media = probe_stream(local_url, "tcp", timeout_seconds)
    except Exception as error:
        raise RuntimeError(f"{camera.id}: MediaMTX path probe failed ({type(error).__name__})") from None
    if media.codec != "hevc":
        raise ValueError(f"{camera.id}: MediaMTX codec is {media.codec}, expected hevc")
    return {
        "camera_id": camera.id,
        "stream_path": f"live/{camera.id}",
        "source_codec": source.codec,
        "mediamtx_codec": media.codec,
        "width": media.width,
        "height": media.height,
        "source_fps": source.fps,
        "mediamtx_fps": media.fps,
        "video_path": "direct",
        "transcoding": False,
    }


def verify_direct_paths(config_path: Path, output_path: Path, rtsp_port: int = 18554,
                        timeout_seconds: float = 12.0) -> list[dict[str, object]]:
    config = load_config(config_path)
    enabled = [camera for camera in config.cameras if camera.enabled]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(enabled)))) as executor:
        futures = {executor.submit(_verify_camera, camera, rtsp_port, timeout_seconds): camera
                   for camera in enabled}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["camera_id"]))
    _write_private(output_path, json.dumps({"cameras": rows}, sort_keys=True) + "\n")
    return rows


def load_validation(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(row["camera_id"]): row for row in payload.get("cameras", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    mode = subparsers.add_parser("mode")
    mode.add_argument("--config", type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--base", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--rtsp-port", type=int, default=18554)
    args = parser.parse_args()
    if args.command == "mode":
        print(load_config(args.config).video.mode)
    elif args.command == "generate":
        config = generate_direct_config(args.config, args.base, args.output)
        print(f"Prepared private direct HEVC paths for {sum(camera.enabled for camera in config.cameras)} camera(s).")
    else:
        for row in verify_direct_paths(args.config, args.output, args.rtsp_port):
            print(f"{row['camera_id']}: source=HEVC MediaMTX=HEVC "
                  f"{row['width']}x{row['height']} {float(row['mediamtx_fps']):.2f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
