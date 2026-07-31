from __future__ import annotations

import argparse
import json

from .config import load_config
from .metrics import Metrics
from .native_relay import NativeRelayPipeline, _redact
from .stream_probe import probe_diagnostics, probe_stream


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cameras.relay.yaml")
    parser.add_argument("--sample-seconds", type=int, default=12)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    for camera in config.cameras:
        if not camera.enabled:
            continue
        try:
            info = probe_stream(camera.url, camera.rtsp_transport)
            row = {"camera": camera.id, "codec": info.codec, "resolution": f"{info.width}x{info.height}",
                   "nominal_fps": info.fps, "video_mode": config.video.mode,
                   "rtsp_transport": camera.rtsp_transport}
            if config.video.mode == "direct_hevc":
                if info.codec != "hevc":
                    raise ValueError(f"{camera.id}: source codec is {info.codec}, expected hevc")
                row.update({"backend": "mediamtx-direct-pull", "video_path": "direct",
                            "transcoding": False})
            else:
                relay = NativeRelayPipeline(camera, config.output, Metrics(), config.video)
                command = relay.command(info)
                row.update({"backend": relay.backend, "encoder": relay.encoder,
                            "command": _redact(command), "video_path": "diagnostic",
                            "transcoding": config.video.h264_mode != "direct",
                            "h264_mode": config.video.h264_mode})
            if info.codec == "h264":
                row["source_diagnostics"] = probe_diagnostics(camera.url, camera.rtsp_transport, args.sample_seconds)
            rows.append(row)
        except Exception as error:
            # CalledProcessError can include the complete ffprobe argv, which
            # contains the RTSP credential.  Keep diagnostics useful without
            # ever printing the source URL.
            detail = _redact([str(getattr(error, "stderr", "") or "")]).strip().replace("\n", " ")[:160]
            rows.append({"camera": camera.id, "error": type(error).__name__,
                         **({"detail": detail} if detail else {})})
    print(json.dumps({"cameras": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
