"""Prepare a private config with all video enabled and one AI branch enabled."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build(input_path: Path, camera_id: str, output_path: Path,
          only_video_camera: bool = False) -> None:
    with input_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cameras = raw.get("cameras", [])
    if camera_id not in {camera.get("id") for camera in cameras}:
        raise ValueError(f"Unknown camera ID: {camera_id}")
    for camera in cameras:
        if only_video_camera:
            camera["enabled"] = camera.get("id") == camera_id and camera.get("enabled", True)
        camera["detection_enabled"] = camera.get("id") == camera_id and camera.get("enabled", True)
    raw.setdefault("detection", {})["enabled"] = not only_video_camera
    raw.setdefault("pipeline", {})["mode"] = "native"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    output_path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-video-camera", action="store_true")
    args = parser.parse_args()
    build(args.input, args.camera, args.output, args.only_video_camera)
    purpose = "direct video" if args.only_video_camera else "detection"
    print(f"Prepared private {purpose} validation for {args.camera}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
