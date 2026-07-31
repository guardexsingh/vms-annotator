from __future__ import annotations

from .encoder import H264Encoder


class MediaMtxPublisher:
    """Narrow publishing boundary, kept separate so protocol changes do not touch detection."""
    def __init__(self, width: int, height: int, fps: float, rtsp_url: str) -> None:
        self.encoder = H264Encoder(width, height, fps, rtsp_url)

    def start(self) -> None:
        self.encoder.start()

    def publish(self, image) -> None:
        self.encoder.write(image)

    def stop(self) -> None:
        self.encoder.stop()
