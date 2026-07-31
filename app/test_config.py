"""Create private, single-camera configurations for relay comparisons.

Generated MediaMTX direct-pull files can contain RTSP credentials, so callers
must store them under ``run/`` with mode 0600 and remove them after the test.
This module never prints a URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import expand_environment


def _write_private_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)


def build(input_path: Path, mediamtx_base: Path, camera_id: str, h264_mode: str,
          app_output: Path, mediamtx_output: Path) -> None:
    with input_path.open(encoding="utf-8") as handle:
        app = yaml.safe_load(handle) or {}
    selected = next((item for item in app.get("cameras", []) if item.get("id") == camera_id), None)
    if selected is None:
        raise ValueError(f"Unknown camera ID: {camera_id}")
    # Expand only the selected source.  A disabled/retired camera in the same
    # YAML must not impose a URL requirement on this single-camera test.
    camera_url = expand_environment(selected.get("url", ""))
    for item in app.get("cameras", []):
        item["enabled"] = item.get("id") == camera_id
    app.setdefault("pipeline", {})["mode"] = "native"
    app.setdefault("detection", {})["enabled"] = False
    app.setdefault("video", {})["mode"] = "diagnostic_transcode"
    app["video"]["allow_transcode_fallback"] = False
    app.setdefault("video", {})["h264_mode"] = h264_mode
    _write_private_yaml(app_output, app)

    with mediamtx_base.open(encoding="utf-8") as handle:
        mediamtx = yaml.safe_load(handle) or {}
    path: dict[str, object] = {}
    if h264_mode == "direct":
        path["source"] = camera_url
        transport = selected.get("rtsp_transport", "tcp")
        if transport != "auto":
            path["rtspTransport"] = transport
    mediamtx["paths"] = {f"live/{camera_id}": path}
    _write_private_yaml(mediamtx_output, mediamtx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mediamtx-base", type=Path, default=Path("config/mediamtx.yml"))
    parser.add_argument("--camera", required=True)
    parser.add_argument("--h264-mode", choices=("copy", "transcode", "direct"), required=True)
    parser.add_argument("--app-output", type=Path, required=True)
    parser.add_argument("--mediamtx-output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.mediamtx_base, args.camera, args.h264_mode, args.app_output, args.mediamtx_output)
    print(f"Prepared private single-camera {args.h264_mode} test for {args.camera}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
