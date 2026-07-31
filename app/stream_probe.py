from __future__ import annotations

import json
import subprocess
import time
from fractions import Fraction
from statistics import median
from typing import Any

from .models import StreamInfo


def normalize_codec(codec: str) -> str:
    value = codec.lower().strip()
    if value in {"h264", "avc", "avc1"}:
        return "h264"
    if value in {"hevc", "h265", "hev1", "hvc1"}:
        return "hevc"
    raise ValueError(f"Unsupported video codec: {codec}")


def parse_ffprobe(payload: dict) -> StreamInfo:
    streams = payload.get("streams", [])
    stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError("ffprobe found no video stream")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return StreamInfo(
        codec=normalize_codec(str(stream.get("codec_name", ""))),
        width=int(stream["width"]), height=int(stream["height"]), fps=fps,
        pixel_format=stream.get("pix_fmt"),
    )


def _transport_arguments(transport: str) -> list[str]:
    if transport not in {"tcp", "udp", "auto"}:
        raise ValueError(f"Unsupported RTSP transport: {transport}")
    return [] if transport == "auto" else ["-rtsp_transport", transport]


def probe_stream(url: str, transport: str = "tcp", timeout_seconds: float = 8.0) -> StreamInfo:
    command = ["ffprobe", "-v", "error", *_transport_arguments(transport), "-select_streams", "v:0",
               "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,codec_type",
               "-of", "json", url]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=True)
    return parse_ffprobe(json.loads(completed.stdout))


def _fraction(value: Any) -> float | None:
    try:
        result = float(Fraction(str(value)))
        return result if result >= 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _packet_diagnostics(packets: list[dict[str, Any]], frames: list[dict[str, Any]], sample_seconds: float | None = None) -> dict[str, Any]:
    pts = [_float(packet.get("pts_time")) for packet in packets]
    dts = [_float(packet.get("dts_time")) for packet in packets]
    valid_pts = [value for value in pts if value is not None]
    valid_dts = [value for value in dts if value is not None]
    positive_gaps = [right - left for left, right in zip(valid_pts, valid_pts[1:]) if right is not None and left is not None and right > left]
    typical_gap = median(positive_gaps) if positive_gaps else None
    # A gap over four frame periods is a source-timeline discontinuity.  This
    # measures packet PTS burstiness, not physical network arrival times.
    discontinuities = sum(1 for left, right in zip(valid_pts, valid_pts[1:])
                          if right is not None and left is not None and right < left)
    if typical_gap:
        discontinuities += sum(1 for gap in positive_gaps if gap > typical_gap * 4)
    key_times = [_float(frame.get("best_effort_timestamp_time")) for frame in frames if int(frame.get("key_frame", 0)) == 1]
    key_times = [value for value in key_times if value is not None]
    key_intervals = [right - left for left, right in zip(key_times, key_times[1:]) if right > left]
    side_data = [item for packet in packets for item in packet.get("side_data_list", [])]
    parameter_sets = sum(1 for item in side_data if "parameter" in str(item.get("side_data_type", "")).lower()
                         or "extradata" in str(item.get("side_data_type", "")).lower())
    return {
        "sample_packets": len(packets),
        "negative_timestamps": sum(1 for value in [*valid_pts, *valid_dts] if value < 0),
        "non_monotonic_dts": sum(1 for left, right in zip(valid_dts, valid_dts[1:]) if right < left),
        "pts_dts_discontinuities": discontinuities,
        "keyframe_interval_seconds": median(key_intervals) if key_intervals else None,
        "sps_pps_observations": parameter_sets,
        "sps_pps_frequency_per_second": round(parameter_sets / sample_seconds, 3) if sample_seconds else None,
        "packet_pts_gap_median_seconds": typical_gap,
        "packet_pts_gap_max_seconds": max(positive_gaps) if positive_gaps else None,
        "packet_arrival_burstiness": "not inferred from PTS; measure with a packet capture when live network timing is required",
    }


def probe_diagnostics(url: str, transport: str = "tcp", sample_seconds: int = 12,
                      timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Collect redaction-safe source timing evidence from a bounded live sample."""
    sample_seconds = max(1, min(sample_seconds, 60))
    command = ["ffprobe", "-v", "error", *_transport_arguments(transport), "-select_streams", "v:0",
               "-read_intervals", f"%+{sample_seconds}", "-show_entries",
               "stream=codec_name,profile,width,height,bit_rate,has_b_frames,time_base,start_time,avg_frame_rate,r_frame_rate:"
               "packet=pts_time,dts_time,flags,size,side_data_list:frame=key_frame,best_effort_timestamp_time,pict_type",
               "-show_packets", "-show_frames", "-of", "json", url]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=True)
    payload = json.loads(completed.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_name")), {})
    result = {
        "codec": normalize_codec(str(stream.get("codec_name", ""))),
        "codec_profile": stream.get("profile"),
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "fps": _fraction(stream.get("avg_frame_rate")),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "real_frame_rate": stream.get("r_frame_rate"),
        "bitrate": _float(stream.get("bit_rate")),
        "has_b_frames": stream.get("has_b_frames"),
        "time_base": stream.get("time_base"),
        "start_time": stream.get("start_time"),
        "rtsp_transport": transport,
        "sample_elapsed_seconds": round(time.monotonic() - started, 3),
    }
    result.update(_packet_diagnostics(payload.get("packets", []), payload.get("frames", []), time.monotonic() - started))
    return result
