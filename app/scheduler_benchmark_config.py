"""Generate a private scheduler comparison config without touching video."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def build(input_path: Path, output_path: Path, batch_mode: str,
          threads: int, capture_fps: float, inference_workers: int = 1) -> None:
    with input_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    detection = raw.setdefault("detection", {})
    detection.update({
        "backend": "pytorch",
        "allow_backend_fallback": True,
        "device": "cpu",
        "precision": "fp32",
        "batch_mode": batch_mode,
        "inference_workers": inference_workers,
        "torch_threads": threads,
        "torch_interop_threads": 1,
        "capture_fps": capture_fps,
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
    parser.add_argument("--batch-mode", choices=("serial", "batch", "opportunistic"), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 6), default=4)
    parser.add_argument("--inference-workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--capture-fps", type=float, default=5.0)
    args = parser.parse_args()
    build(args.input, args.output, args.batch_mode, args.threads, args.capture_fps,
          args.inference_workers)
    print(f"Prepared private {args.batch_mode} scheduler benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
