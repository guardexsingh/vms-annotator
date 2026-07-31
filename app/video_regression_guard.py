"""Fail a benchmark if the frozen direct-HEVC video architecture changes."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import psutil


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status}")
        return json.load(response)


def _tracks(path: dict) -> tuple[str, ...]:
    tracks = path.get("tracks") or path.get("readyTracks") or []
    values: list[str] = []
    for track in tracks:
        if isinstance(track, str):
            values.append(track.lower())
        elif isinstance(track, dict):
            values.append(str(track.get("codec") or track.get("type") or "").lower())
    return tuple(values)


def _process_guard() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    forbidden = ("libx264", "h264_nvenc", "h264_v4l2m2m", "h264_omx",
                 "annotated/", "-c:v h264", "-codec:v h264")
    for name in ("app", "mediamtx"):
        pid_file = Path("run") / f"{name}.pid"
        if not pid_file.exists():
            failures.append(f"{name} PID file missing")
            continue
        try:
            root = psutil.Process(int(pid_file.read_text().strip()))
            processes = [root, *root.children(recursive=True)]
        except (ValueError, psutil.Error):
            failures.append(f"{name} process unavailable")
            continue
        for process in processes:
            try:
                command = " ".join(process.cmdline())
                row = {"pid": process.pid, "ppid": process.ppid(), "name": process.name()}
            except psutil.Error:
                continue
            rows.append(row)
            lowered = command.lower()
            if any(token in lowered for token in forbidden):
                failures.append(f"forbidden video encoder or annotated path in PID {process.pid}")
    return rows, failures


def evaluate() -> dict[str, object]:
    metrics = _json("http://127.0.0.1:18080/metrics")
    public = _json("http://127.0.0.1:18080/api/cameras")
    media = _json("http://127.0.0.1:19997/v3/paths/list")
    expected = [str(camera["id"]) for camera in public.get("cameras", [])]
    path_items = {str(item.get("name")): item for item in media.get("items", [])}
    failures: list[str] = []
    cameras: dict[str, dict[str, object]] = {}
    for camera_id in expected:
        metric = metrics.get("cameras", {}).get(camera_id)
        if not isinstance(metric, dict):
            failures.append(f"{camera_id}: metrics missing")
            continue
        path = path_items.get(f"live/{camera_id}")
        tracks = _tracks(path or {})
        browser_fps = metric.get("browser_decoded_fps")
        row = {
            "source_codec": metric.get("source_codec"),
            "mediamtx_codec": metric.get("mediamtx_codec"),
            "mediamtx_tracks": tracks,
            "source_fps": metric.get("source_fps"),
            "mediamtx_fps": metric.get("mediamtx_fps"),
            "video_path": metric.get("video_path"),
            "transcoding": metric.get("transcoding"),
            "browser_codec": metric.get("webrtc_codec"),
            "browser_fps": browser_fps,
            "browser_telemetry": "measured" if browser_fps is not None else "not_connected",
        }
        cameras[camera_id] = row
        if str(row["source_codec"]).lower() not in {"hevc", "h265"}:
            failures.append(f"{camera_id}: source codec is not HEVC")
        if str(row["mediamtx_codec"]).lower() not in {"hevc", "h265"}:
            failures.append(f"{camera_id}: MediaMTX codec is not HEVC")
        if row["video_path"] != "direct" or row["transcoding"] is not False:
            failures.append(f"{camera_id}: video is not direct/non-transcoded")
        if path is None or not any(codec in {"h265", "hevc"} for codec in tracks):
            failures.append(f"{camera_id}: MediaMTX live path has no HEVC track")
        if browser_fps is not None:
            if str(row["browser_codec"]).lower() not in {"hevc", "h265"}:
                failures.append(f"{camera_id}: browser codec is not HEVC")
            if float(browser_fps) < 20.0:
                failures.append(f"{camera_id}: browser decoded FPS below 20")
    processes, process_failures = _process_guard()
    failures.extend(process_failures)
    return {
        "checked_at_unix": time.time(),
        "status": "pass" if not failures else "fail",
        "video_architecture": "camera HEVC -> MediaMTX direct pull -> WHEP",
        "cameras": cameras,
        "processes": processes,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate()
    text = json.dumps(payload, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(f"Video regression guard: {payload['status']}")
    for failure in payload["failures"]:
        print(failure)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
