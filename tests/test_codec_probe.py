import pytest

from app.stream_probe import _packet_diagnostics, parse_ffprobe


@pytest.mark.parametrize(("codec", "expected"), [("h264", "h264"), ("hevc", "hevc")])
def test_codec_probe_parses_h264_and_hevc(codec, expected):
    info = parse_ffprobe({"streams":[{"codec_type":"video", "codec_name":codec, "width":1920, "height":1080, "avg_frame_rate":"25/1", "pix_fmt":"yuv420p"}]})
    assert (info.codec, info.width, info.height, info.fps) == (expected, 1920, 1080, 25)


def test_packet_diagnostics_identify_problematic_source_timestamps():
    result = _packet_diagnostics(
        [{"pts_time": "-0.040", "dts_time": "-0.080"}, {"pts_time": "0.000", "dts_time": "0.000"},
         {"pts_time": "2.000", "dts_time": "-0.040"}],
        [{"key_frame": 1, "best_effort_timestamp_time": "0.000"},
         {"key_frame": 1, "best_effort_timestamp_time": "2.000"}],
    )
    assert result["negative_timestamps"] == 3
    assert result["non_monotonic_dts"] == 1
    assert result["keyframe_interval_seconds"] == 2
    assert "not inferred" in result["packet_arrival_burstiness"]
