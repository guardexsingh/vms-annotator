"""Independent low-latency relay that never sends video frames through Python."""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time

from .encoder import select_encoder
from .metrics import Metrics
from .models import CameraConfig, OutputConfig, StreamInfo, VideoConfig
from .stream_probe import probe_stream

LOG = logging.getLogger(__name__)


def _redact(command: list[str]) -> str:
    return re.sub(r"rtsp://[^\s]+", "rtsp://REDACTED", " ".join(command))


class NativeRelayPipeline:
    """One FFmpeg process per camera; no stdout/stdin raw-video pipes exist."""
    queue_limit = 1

    def __init__(self, camera: CameraConfig, output: OutputConfig, metrics: Metrics,
                 video: VideoConfig | None = None, rtsp_port: int = 18554) -> None:
        self.camera, self.output, self.metrics, self.video, self.rtsp_port = camera, output, metrics, video or VideoConfig(), rtsp_port
        self.info: StreamInfo | None = None
        self.backend = "pending"
        self.encoder = "pending"
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"relay-{self.camera.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_direct(self) -> bool:
        return self.info is not None and self.info.codec == "h264" and self.video.h264_mode == "direct"

    def command(self, info: StreamInfo) -> list[str]:
        transport = [] if self.camera.rtsp_transport == "auto" else ["-rtsp_transport", self.camera.rtsp_transport]
        if info.codec == "h264" and self.video.h264_mode == "direct":
            self.backend, self.encoder = "mediamtx-direct-pull", "passthrough"
            return []
        # Keep the copy path close to the source so its timing behaviour is a
        # useful control.  The transcode path below intentionally regenerates
        # its timeline rather than trusting camera PTS/DTS.
        transcode = info.codec != "h264" or self.video.h264_mode == "transcode"
        input_fflags = "+genpts+nobuffer+discardcorrupt" if transcode else "+nobuffer+discardcorrupt"
        common = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-fflags", input_fflags,
                  "-flags", "low_delay", "-avioflags", "direct", "-max_delay", "0", "-analyzeduration", "0",
                  "-probesize", "32768", "-thread_queue_size", str(self.queue_limit), *transport,
                  "-i", self.camera.url, "-map", "0:v:0", "-an"]
        publish = ["-f", "rtsp", "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{self.rtsp_port}/live/{self.camera.id}"]
        if info.codec == "h264" and self.video.h264_mode == "copy":
            self.backend, self.encoder = "ffmpeg-stream-copy", "copy"
            return [*common, "-c:v", "copy", *publish]
        selected, software = select_encoder()
        source = "h264" if info.codec == "h264" else "hevc"
        self.backend, self.encoder = f"ffmpeg-{source}-h264-transcode", selected
        # This test path deliberately creates a fresh CFR timeline.  It does
        # not use use_wallclock_as_timestamps: wall-clock arrival jitter is
        # not a stable video clock.  make_zero removes source negative PTS;
        # setpts then makes frames monotonic in decoded display order.
        options = ["-preset", "ultrafast", "-tune", "zerolatency", "-x264-params",
                   "bframes=0:rc-lookahead=0:sync-lookahead=0:scenecut=0:repeat-headers=1"] if software else []
        fps = self.output.fps or info.fps
        clean_timeline = ["-avoid_negative_ts", "make_zero", "-vf", f"setpts=N/({max(1.0, fps):.6f}*TB)",
                          "-r", f"{max(1.0, fps):.6f}"]
        return [*common, *clean_timeline, "-c:v", selected, *options, "-bf", "0", "-g", str(max(1, round(fps))),
                "-pix_fmt", "yuv420p", "-bsf:v", "dump_extra=freq=keyframe", "-muxdelay", "0", "-muxpreload", "0", *publish]

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            metric = self.metrics.camera(self.camera.id)
            try:
                self.info = probe_stream(self.camera.url, self.camera.rtsp_transport)
                # Direct mode deliberately has no FFmpeg relay.  The companion
                # test script generates a MediaMTX path whose source is the
                # camera itself; ffprobe is only a short diagnostic probe.
                if self.is_direct:
                    self.backend, self.encoder = "mediamtx-direct-pull", "passthrough"
                    metric.codec, metric.width, metric.height, metric.source_fps = self.info.codec, self.info.width, self.info.height, self.info.fps
                    metric.decoder, metric.encoder, metric.online, metric.video_status = "mediamtx", "passthrough", True, "ready"
                    metric.decoder_command = metric.encoder_command = "MediaMTX direct RTSP pull (no FFmpeg relay)"
                    return
                command = self.command(self.info)
                metric.codec, metric.width, metric.height, metric.source_fps = self.info.codec, self.info.width, self.info.height, self.info.fps
                metric.decoder, metric.encoder = ("packet-copy" if self.backend.endswith("copy") else self.info.codec), self.encoder
                metric.source_codec, metric.mediamtx_codec = self.info.codec, "h264"
                metric.video_path, metric.transcoding = "diagnostic_transcode", True
                metric.decoder_command = _redact(command)
                metric.encoder_command = _redact(command)
                metric.decode_queue_depth = metric.encoder_queue_depth = self.queue_limit
                metric.online = True
                metric.video_status = "ready"
                metric.raw_frames_through_python = False
                LOG.info("%s native relay: %s / %s", self.camera.id, self.backend, self.encoder)
                self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                while not self._stop.wait(.25):
                    if self._process.poll() is not None:
                        raise RuntimeError(f"relay exited ({self._process.returncode})")
                    # Do not invent frame-rate telemetry from a timer.  MediaMTX
                    # and the browser test provide the actual delivered rate.
                return
            except Exception as error:
                metric.online = False
                metric.video_status = "offline"
                metric.reconnections += 1
                LOG.warning("%s relay offline; retry in %.1fs: %s", self.camera.id, backoff, _redact([str(error)]))
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                if self._process and self._process.poll() is None:
                    self._process.terminate()
