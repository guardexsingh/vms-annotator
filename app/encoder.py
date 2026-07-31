from __future__ import annotations

import subprocess
import re
from pathlib import Path


def ffmpeg_has_encoder(name: str) -> bool:
    try:
        text = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.DEVNULL)
        return name in text
    except (OSError, subprocess.SubprocessError):
        return False


def select_encoder() -> tuple[str, bool]:
    # A compiled FFmpeg wrapper is insufficient: require a locally visible
    # accelerator device, then let the first real frame validate it.
    has_device = any(Path(device).exists() for device in ("/dev/nvhost-msenc", "/dev/video0"))
    encoders = ("h264_nvenc", "h264_v4l2m2m", "h264_omx")
    for encoder in encoders:
        if has_device and ffmpeg_has_encoder(encoder):
            return encoder, False
    return "libx264", True


class H264Encoder:
    def __init__(self, width: int, height: int, fps: float, publish_url: str, encoder: str | None = None) -> None:
        self.width, self.height, self.fps, self.publish_url = width, height, max(1.0, fps), publish_url
        self.encoder, self.software_fallback = (encoder, encoder == "libx264") if encoder else select_encoder()
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def command(self) -> list[str]:
        codec_options = ["-preset", "ultrafast", "-tune", "zerolatency"] if self.encoder == "libx264" else []
        return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
                "-i", "-", "-an", "-c:v", self.encoder, *codec_options, "-pix_fmt", "yuv420p", "-bf", "0",
                "-g", str(max(1, round(self.fps))), "-f", "rtsp", "-rtsp_transport", "tcp", self.publish_url]

    def redacted_command(self) -> str:
        return re.sub(r"rtsp://[^\s]+", "rtsp://REDACTED", " ".join(self.command()))

    def write(self, image) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("H.264 encoder is not running")
        self.process.stdin.write(image.tobytes())

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
