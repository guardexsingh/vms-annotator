from unittest.mock import Mock, patch

from app.decoder import CameraDecoder, select_decoder
from app.models import StreamInfo


def test_decoder_falls_back_to_software_when_hardware_is_unavailable():
    with patch("app.decoder.hardware_decode_available", return_value=False):
        assert select_decoder("hevc") == ("hevc", True)


def test_ai_decoder_scales_only_its_sampling_copy():
    decoder = CameraDecoder("rtsp://example/live", "cam03", max_dimension=640)
    decoder.info = StreamInfo("hevc", 2560, 1440, 25, decoder="hevc")
    assert decoder._output_size() == (640, 360)
    assert "scale='if(gt(iw,ih),640,-2)':'if(gt(iw,ih),-2,640)'" in decoder._command("hevc")


def test_stop_terminates_and_reaps_ffmpeg_process():
    decoder = CameraDecoder("rtsp://example/live", "office")
    process = Mock()
    process.poll.return_value = None
    decoder._process = process
    decoder.stop()
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=1)
    assert decoder._process is None
