from __future__ import annotations

import logging
import threading
import time

from .compositor import Compositor
from .decoder import CameraDecoder
from .encoder import H264Encoder
from .latest_frame import LatestFrame
from .metrics import Metrics
from .models import CameraConfig, Frame, OutputConfig

LOG = logging.getLogger(__name__)


class CameraWorker:
    """Owns one camera, preserving isolation across probe/decode/encode failures."""
    def __init__(self, config: CameraConfig, output: OutputConfig, inference_slot: LatestFrame[Frame] | None,
                 compositor: Compositor, metrics: Metrics, rtsp_port: int = 18554) -> None:
        self.config, self.output, self.inference_slot = config, output, inference_slot
        self.compositor, self.metrics, self.rtsp_port = compositor, metrics, rtsp_port
        self._render_slot: LatestFrame[Frame] = LatestFrame()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.config.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)

    def _run(self) -> None:
        backoff, prefer_hardware, force_software_encoder = 1.0, True, False
        while not self._stop.is_set():
            decoder = CameraDecoder(self.config.url, self.config.id, prefer_hardware=prefer_hardware)
            encoder = None
            try:
                info = decoder.probe()
                metric = self.metrics.camera(self.config.id)
                metric.codec, metric.width, metric.height, metric.source_fps = info.codec, info.width, info.height, info.fps
                metric.decoder, metric.online = info.decoder, True
                LOG.info("%s: input codec %s, decoder %s%s", self.config.id, info.codec, info.decoder,
                         " (software fallback)" if info.software_fallback else "")
                decoder.start()
                encoder = H264Encoder(info.width, info.height, self.output.fps or info.fps,
                                      f"rtsp://127.0.0.1:{self.rtsp_port}/diagnostic-overlay/{self.config.id}",
                                      encoder="libx264" if force_software_encoder else None)
                encoder.start()
                metric.encoder = encoder.encoder
                metric.decoder_command = decoder.redacted_command()
                metric.encoder_command = encoder.redacted_command()
                LOG.info("%s: output encoder %s%s", self.config.id, encoder.encoder,
                         " (software fallback)" if encoder.software_fallback else "")
                backoff = 1.0
                while not self._stop.is_set() and decoder.failure is None:
                    frame = decoder.get_latest_frame()
                    if frame is None:
                        self._stop.wait(.002)
                        continue
                    metric.decode_queue_depth = decoder.queue_depth
                    metric.dropped_frames = decoder.replaced_frames
                    self.metrics.record_frame(self.config.id, "compositor_input")
                    if self.inference_slot is not None and self.inference_slot.put(frame):
                        metric.inference_frames_replaced += 1
                    self.metrics.record_frame(self.config.id, "decoded")
                    composed_at = time.monotonic()
                    image, count, age, newly_stale = self.compositor.compose(self.config.id, frame)
                    metric.compositor_ms = (time.monotonic() - composed_at) * 1000
                    self.metrics.record_frame(self.config.id, "composited")
                    metric.person_count, metric.detection_age_ms = count, age
                    if newly_stale:
                        metric.stale_results += 1
                    try:
                        encode_started = time.monotonic()
                        encoder.write(image)
                        metric.encoder_write_ms = (time.monotonic() - encode_started) * 1000
                        metric.encoder_queue_depth = 0
                        self.metrics.record_frame(self.config.id, "encoded")
                        self.metrics.record_frame(self.config.id, "published")
                    except Exception as error:
                        metric.encoder_failures += 1
                        if not encoder.software_fallback:
                            force_software_encoder = True
                            LOG.warning("%s: hardware H.264 encoder failed; switching to libx264", self.config.id)
                        raise RuntimeError(f"encoder/publisher failed: {error}") from error
                if decoder.failure:
                    metric.decode_failures += 1
                    if decoder.info and not decoder.info.software_fallback:
                        prefer_hardware = False
                        LOG.warning("%s: hardware decoder failed; switching to software fallback", self.config.id)
                    raise decoder.failure
            except Exception as error:
                metric = self.metrics.camera(self.config.id)
                if encoder and not encoder.software_fallback:
                    # Startup errors can surface only once FFmpeg has attempted initialization.
                    force_software_encoder = True
                metric.online = False
                metric.reconnections += 1
                LOG.warning("%s offline; retry in %.1fs: %s", self.config.id, backoff, error)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                decoder.stop()
                if encoder:
                    encoder.stop()
