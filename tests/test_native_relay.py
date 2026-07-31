from app.metrics import Metrics
from app.models import CameraConfig, OutputConfig, StreamInfo, VideoConfig
from app.native_relay import NativeRelayPipeline, _redact


def relay(codec: str, mode: str = "copy") -> NativeRelayPipeline:
    return NativeRelayPipeline(CameraConfig("cam03", "Camera", "rtsp://user:secret@example/live", rtsp_transport="udp"),
                               OutputConfig(), Metrics(),
                               VideoConfig(mode="diagnostic_transcode", h264_mode=mode))


def test_h264_native_relay_uses_packet_copy_without_raw_video_pipe():
    pipeline = relay("h264")
    command = pipeline.command(StreamInfo("h264", 1920, 1080, 25))
    assert pipeline.backend == "ffmpeg-stream-copy"
    assert "copy" in command
    assert "rawvideo" not in command
    assert "pipe:" not in " ".join(command)
    assert pipeline.queue_limit == 1


def test_hevc_relay_uses_h264_output_with_bounded_input_queue():
    pipeline = relay("hevc")
    command = pipeline.command(StreamInfo("hevc", 1920, 1080, 25))
    assert pipeline.backend == "ffmpeg-hevc-h264-transcode"
    assert "-thread_queue_size" in command
    assert command[command.index("-thread_queue_size") + 1] == "1"
    assert "-bf" in command and command[command.index("-bf") + 1] == "0"
    assert "rawvideo" not in command


def test_h264_transcode_regenerates_a_low_latency_timeline():
    pipeline = relay("h264", "transcode")
    command = pipeline.command(StreamInfo("h264", 1920, 1080, 25))
    assert pipeline.backend == "ffmpeg-h264-h264-transcode"
    assert "copy" not in command
    assert command[command.index("-bf") + 1] == "0"
    assert "repeat-headers=1" in " ".join(command)
    assert "setpts=N/(25.000000*TB)" in command
    assert command[command.index("-fflags") + 1] == "+genpts+nobuffer+discardcorrupt"
    assert "-avoid_negative_ts" in command
    assert "dump_extra=freq=keyframe" in command
    assert "-muxdelay" in command
    assert "rawvideo" not in command


def test_h264_direct_mode_has_no_ffmpeg_command():
    pipeline = relay("h264", "direct")
    assert pipeline.command(StreamInfo("h264", 1920, 1080, 25)) == []
    assert pipeline.backend == "mediamtx-direct-pull"


def test_relay_command_redacts_credentials():
    pipeline = relay("h264")
    command = pipeline.command(StreamInfo("h264", 1, 1, 1))
    assert "secret" not in _redact(command)


def test_relay_error_redaction_handles_ffprobe_style_command_text():
    assert "secret" not in _redact(["Command '['ffprobe', 'rtsp://user:secret@example/live']' failed"])
