"""Independent, latest-frame-only capture workers for detection."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .decoder import CameraDecoder
from .latest_frame import LatestFrame
from .metrics import Metrics
from .models import CameraConfig, Frame

LOG = logging.getLogger(__name__)


class AICaptureWorker:
    """Continuously drains one source into exactly one replaceable frame slot."""

    def __init__(self, camera: CameraConfig, slot: LatestFrame[Frame], metrics: Metrics,
                 local_rtsp_port: int = 18554, local_path_prefix: str = "live",
                 on_status: Callable[[str, str], None] | None = None, max_dimension: int = 640,
                 sample_fps: float | None = None) -> None:
        self.camera, self.slot, self.metrics = camera, slot, metrics
        self.local_rtsp_port, self.local_path_prefix = local_rtsp_port, local_path_prefix
        self.on_status = on_status
        self.max_dimension = max_dimension
        self.sample_fps = sample_fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._decoder: CameraDecoder | None = None

    @property
    def source_url(self) -> str:
        if self.camera.detection_source == "local_mediamtx":
            return f"rtsp://127.0.0.1:{self.local_rtsp_port}/{self.local_path_prefix}/{self.camera.id}"
        return self.camera.url

    def _on_frame(self, frame: Frame, replaced: bool) -> None:
        metric = self.metrics.camera(self.camera.id)
        self.metrics.record_frame(self.camera.id, "ai_capture")
        if metric.ai_capture_status != "ready" and self.on_status:
            self.on_status(self.camera.id, "ready")
        metric.ai_capture_status = "ready"
        metric.ai_capture_backend = "ffmpeg"
        metric.decode_queue_depth = self.slot.depth
        if replaced:
            metric.ai_frames_replaced += 1

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"ai-capture-{self.camera.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._decoder:
            self._decoder.stop()
        if self._thread:
            self._thread.join(timeout=4)

    def _run(self) -> None:
        backoff, prefer_hardware = 1.0, True
        metric = self.metrics.camera(self.camera.id)
        metric.ai_capture_status = "loading"
        while not self._stop.is_set():
            decoder = CameraDecoder(self.source_url, self.camera.id, prefer_hardware=prefer_hardware,
                                    rtsp_transport="tcp" if self.camera.detection_source == "local_mediamtx" else self.camera.rtsp_transport,
                                    frame_slot=self.slot,
                                    on_frame=self._on_frame, max_dimension=self.max_dimension,
                                    sample_fps=self.sample_fps)
            self._decoder = decoder
            try:
                info = decoder.probe()
                metric.width, metric.height, metric.source_fps = info.width, info.height, info.fps
                output_width, output_height = decoder._output_size()
                metric.ai_decoder_input_fps = info.fps
                metric.ai_output_width, metric.ai_output_height = output_width, output_height
                metric.ai_capture_backend = f"ffmpeg/{info.decoder}"
                metric.decoder_command = decoder.redacted_command()
                decoder.start()
                backoff = 1.0
                while not self._stop.wait(0.1):
                    self.metrics.set_ai_decoder_pid(self.camera.id, decoder.process_pid)
                    if decoder.failure is not None:
                        raise decoder.failure
                return
            except Exception as error:
                metric.ai_capture_status = "offline"
                if self.on_status:
                    self.on_status(self.camera.id, "offline")
                metric.decode_failures += 1
                metric.reconnections += 1
                if decoder.info and not decoder.info.software_fallback:
                    prefer_hardware = False
                LOG.warning("%s AI capture offline; retry in %.1fs: %s", self.camera.id, backoff,
                            type(error).__name__)
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                decoder.stop()
                self.metrics.set_ai_decoder_pid(self.camera.id, None)
