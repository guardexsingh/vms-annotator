from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .latest_frame import LatestFrame
from .models import Frame, StreamInfo
from .stream_probe import probe_stream

LOG = logging.getLogger(__name__)


def ffmpeg_has_decoder(name: str) -> bool:
    try:
        text = subprocess.check_output(["ffmpeg", "-hide_banner", "-decoders"], text=True, stderr=subprocess.DEVNULL)
        return name in text
    except (OSError, subprocess.SubprocessError):
        return False


def hardware_decode_available(codec: str) -> bool:
    # A compiled wrapper alone is not enough; the device must be accessible.
    return any(Path(device).exists() for device in ("/dev/nvhost-nvdec", "/dev/video0")) and ffmpeg_has_decoder(f"{codec}_nvv4l2dec")


def select_decoder(codec: str, prefer_hardware: bool = True) -> tuple[str, bool]:
    if codec not in {"h264", "hevc"}:
        raise ValueError(f"Unsupported input codec {codec}")
    if prefer_hardware and hardware_decode_available(codec):
        return f"{codec}_nvv4l2dec", False
    return codec, prefer_hardware  # second field means software fallback was selected


class CameraDecoder:
    """Codec-specific FFmpeg decoding behind a common BGR-frame interface."""
    def __init__(self, url: str, camera_id: str, prefer_hardware: bool = True,
                 rtsp_transport: str = "tcp", frame_slot: LatestFrame[Frame] | None = None,
                 on_frame: Callable[[Frame, bool], None] | None = None,
                 max_dimension: int | None = None, sample_fps: float | None = None) -> None:
        self.url, self.camera_id = url, camera_id
        self.prefer_hardware, self.rtsp_transport, self.on_frame = prefer_hardware, rtsp_transport, on_frame
        self.max_dimension = max_dimension
        self.sample_fps = sample_fps
        self.info: StreamInfo | None = None
        self._frames: LatestFrame[Frame] = frame_slot or LatestFrame()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.failure: Exception | None = None

    def probe(self) -> StreamInfo:
        base = probe_stream(self.url, self.rtsp_transport)
        decoder, fallback = select_decoder(base.codec, self.prefer_hardware)
        self.info = StreamInfo(**{**base.__dict__, "decoder": decoder, "software_fallback": fallback})
        return self.info

    def start(self) -> None:
        if self.info is None:
            self.probe()
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name=f"decode-{self.camera_id}", daemon=True)
        self._thread.start()

    def get_latest_frame(self) -> Frame | None:
        return self._frames.take()

    @property
    def replaced_frames(self) -> int:
        return self._frames.replaced

    @property
    def queue_depth(self) -> int:
        return self._frames.depth

    def redacted_command(self) -> str:
        decoder = self.info.decoder if self.info else "unknown"
        return re.sub(r"rtsp://[^\s]+", "rtsp://REDACTED", " ".join(self._command(decoder)))

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process and process.poll() is None:
            process.terminate()
        if self._thread:
            self._thread.join(timeout=2)
        if process is not None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            if self._process is process:
                self._process = None

    def _command(self, decoder: str) -> list[str]:
        transport = [] if self.rtsp_transport == "auto" else ["-rtsp_transport", self.rtsp_transport]
        filters: list[str] = []
        if self.sample_fps:
            source_fps = self.info.fps if self.info and self.info.fps > 0 else self.sample_fps
            frame_interval = max(1, round(source_fps / self.sample_fps))
            # Select by decoded frame number. FFmpeg still drains/decodes the
            # inter-frame HEVC stream, but scaling, BGR conversion, pipe I/O,
            # and Python allocation happen only for the selected fresh frame.
            filters.append(f"select='not(mod(n\\,{frame_interval}))'")
        if self.max_dimension:
            filters.append(f"scale='if(gt(iw,ih),{self.max_dimension},-2)':'if(gt(iw,ih),-2,{self.max_dimension})'")
        video_filter = ["-vf", ",".join(filters)] if filters else []
        # JetPack's FFmpeg predates -fps_mode; -vsync vfr prevents duplication
        # without adding a timestamp-paced output queue.
        output_timing = ["-vsync", "vfr"] if self.sample_fps else []
        return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer",
                "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "32768", "-thread_queue_size", "1", *transport,
                "-c:v", decoder, "-i", self.url, "-an", *video_filter, *output_timing,
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]

    @property
    def process_pid(self) -> int | None:
        return self._process.pid if self._process and self._process.poll() is None else None

    def _output_size(self) -> tuple[int, int]:
        assert self.info is not None
        if not self.max_dimension or max(self.info.width, self.info.height) <= self.max_dimension:
            return self.info.width, self.info.height
        ratio = self.max_dimension / max(self.info.width, self.info.height)
        width = max(2, round(self.info.width * ratio / 2) * 2)
        height = max(2, round(self.info.height * ratio / 2) * 2)
        return width, height

    def _read_loop(self) -> None:
        assert self.info is not None
        output_width, output_height = self._output_size()
        byte_count, sequence = output_width * output_height * 3, 0
        try:
            self._process = subprocess.Popen(self._command(self.info.decoder), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            assert self._process.stdout is not None
            while not self._stop.is_set():
                data = self._process.stdout.read(byte_count)
                if len(data) != byte_count:
                    raise RuntimeError(f"FFmpeg decoder ended ({len(data)}/{byte_count} bytes)")
                image = np.frombuffer(data, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
                sequence += 1
                frame = Frame(image=image, captured_at=time.monotonic(), sequence=sequence,
                              source_width=self.info.width, source_height=self.info.height)
                replaced = self._frames.put(frame)
                if self.on_frame:
                    self.on_frame(frame, replaced)
        except Exception as error:
            if not self._stop.is_set():
                self.failure = error
                LOG.warning("%s decoder failed: %s", self.camera_id, error)
