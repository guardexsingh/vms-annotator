"""Generate a private selected-backend stability configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def build(input_path: Path, output_path: Path, scope: str) -> None:
    with input_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    ids = [str(camera.get("id")) for camera in raw.get("cameras", [])]
    if scope != "all" and scope not in ids:
        raise ValueError(f"Unknown camera ID: {scope}")
    if scope != "all":
        for camera in raw.get("cameras", []):
            camera["enabled"] = str(camera.get("id")) == scope
            camera["detection_enabled"] = str(camera.get("id")) == scope
    detection = raw.setdefault("detection", {})
    detection.update({
        "backend": "pytorch",
        "allow_backend_fallback": False,
        "device": "cpu",
        "precision": "fp32",
        "batch_mode": "serial",
        "inference_workers": 3 if scope == "all" else 1,
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "capture_fps": 5,
        "capture_max_dimension": 640,
        "max_frame_age_ms": 750,
        "max_batch_wait_ms": 20,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    output_path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    build(args.input, args.output, args.scope)
    print(f"Prepared private {args.scope} stability configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
