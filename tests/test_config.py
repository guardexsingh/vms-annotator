from pathlib import Path

import pytest
import yaml

from app.config import load_config, required_camera_environment_variables
from app.test_config import build


def test_camera_config_and_environment_substitution(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text("cameras:\n  - id: cam01\n    name: Cam\n    url: ${URL}\ndetection: {fps_per_camera: 5}\n")
    result = load_config(config, {"URL": "rtsp://example/cam"})
    assert result.cameras[0].url == "rtsp://example/cam"
    assert result.detection.fps_per_camera == 5


def test_prediction_fps_environment_override_has_precedence_and_is_safe(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text(
        "cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live}\n"
        "detection: {target_fps_per_camera: 1, capture_fps: 1}\n"
        "tracking: {prediction_fps: 5}\n"
    )
    loaded = load_config(config, {"BYTETRACK_PREDICTION_FPS": "12.5"})
    assert loaded.detection.target_fps_per_camera == loaded.detection.capture_fps == 1
    assert loaded.tracking.yaml_prediction_fps == 5
    assert loaded.tracking.prediction_fps == 12.5
    assert loaded.tracking.prediction_fps_environment_override == 12.5
    assert loaded.tracking.prediction_fps_source == "BYTETRACK_PREDICTION_FPS"


def test_prediction_fps_uses_yaml_then_builtin_default(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live}\n"
                      "tracking: {prediction_fps: 2.5}\n")
    yaml_value = load_config(config, {})
    assert yaml_value.tracking.prediction_fps == 2.5
    assert yaml_value.tracking.prediction_fps_source == "config/cameras.yaml"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live}\n")
    default_value = load_config(config, {})
    assert default_value.tracking.prediction_fps == 5
    assert default_value.tracking.prediction_fps_source == "built-in default"


@pytest.mark.parametrize("value", ("", "0", "-1", "abc", "NaN", "inf", "25.1"))
def test_prediction_fps_rejects_invalid_environment_values(tmp_path: Path, value: str):
    config = tmp_path / "cameras.yml"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live}\n")
    with pytest.raises(ValueError, match="Invalid BYTETRACK_PREDICTION_FPS"):
        load_config(config, {"BYTETRACK_PREDICTION_FPS": value})


def test_multiple_camera_definitions_are_parsed(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text("cameras:\n  - {id: one, name: One, url: rtsp://one}\n  - {id: two, name: Two, url: rtsp://two}\n")
    result = load_config(config)
    assert [camera.id for camera in result.cameras] == ["one", "two"]


def test_disabled_camera_does_not_require_its_url(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text(
        "cameras:\n"
        "  - id: cam01\n"
        "    name: Disabled\n"
        "    url: ${CAM01_URL}\n"
        "    enabled: false\n"
        "  - id: cam02\n"
        "    name: Enabled\n"
        "    url: ${CAM02_URL}\n"
    )
    result = load_config(config, {"CAM02_URL": "rtsp://example/cam02"})
    assert result.cameras[0].enabled is False
    assert result.cameras[0].url == ""
    assert result.cameras[1].url == "rtsp://example/cam02"


def test_relay_only_configuration_disables_detection(tmp_path: Path):
    config = tmp_path / "relay.yml"
    config.write_text("""cameras:\n  - {id: cam01, name: Cam, url: rtsp://example/cam}\ndetection: {enabled: false}\n""")
    result = load_config(config)
    assert result.detection.enabled is False


def test_camera_transport_and_native_pipeline_are_loaded(tmp_path: Path):
    config = tmp_path / "relay.yml"
    config.write_text("""cameras:\n  - {id: cam03, name: Cam, url: rtsp://example/cam, rtsp_transport: udp}\npipeline: {mode: native}\n""")
    result = load_config(config)
    assert result.cameras[0].rtsp_transport == "udp"
    assert result.pipeline.mode == "native"


def test_h264_mode_is_loaded_and_restricted(tmp_path: Path):
    config = tmp_path / "relay.yml"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/cam}\n"
                      "video: {mode: diagnostic_transcode, h264_mode: transcode}\n")
    assert load_config(config).video.h264_mode == "transcode"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/cam}\nvideo: {h264_mode: invalid}\n")
    with pytest.raises(ValueError, match="Unsupported H.264 mode"):
        load_config(config)


def test_direct_hevc_is_default_and_never_allows_automatic_fallback(tmp_path: Path):
    config = tmp_path / "direct.yml"
    config.write_text("cameras:\n  - {id: entry, name: Entry, url: rtsp://example/cam}\n")
    assert load_config(config).video.mode == "direct_hevc"
    assert load_config(config).video.allow_transcode_fallback is False
    config.write_text(
        "cameras:\n  - {id: entry, name: Entry, url: rtsp://example/cam}\n"
        "video: {mode: direct_hevc, allow_transcode_fallback: true}\n"
    )
    with pytest.raises(ValueError, match="does not permit"):
        load_config(config)


def test_required_variables_are_derived_from_enabled_arbitrary_camera_entries(tmp_path: Path):
    config = tmp_path / "renamed.yml"
    config.write_text(
        "cameras:\n"
        "  - {id: office-entry, name: Entry, url: '${OFFICE_ENTRY_URL}'}\n"
        "  - {id: warehouse-west, name: Warehouse, url: '${WAREHOUSE_WEST_URL}'}\n"
        "  - {id: removed-legacy, name: Retired, url: '${RETIRED_CAMERA_URL}', enabled: false}\n"
    )
    assert required_camera_environment_variables(config) == ("OFFICE_ENTRY_URL", "WAREHOUSE_WEST_URL")


def test_normal_and_relay_configurations_have_independent_url_requirements(tmp_path: Path):
    normal = tmp_path / "normal.yml"
    relay = tmp_path / "relay.yml"
    normal.write_text("cameras:\n  - {id: office-entry, name: Entry, url: '${OFFICE_ENTRY_URL}'}\n")
    relay.write_text("cameras:\n  - {id: loading-bay, name: Bay, url: '${LOADING_BAY_URL}'}\n")
    assert required_camera_environment_variables(normal) == ("OFFICE_ENTRY_URL",)
    assert required_camera_environment_variables(relay) == ("LOADING_BAY_URL",)


def test_single_camera_direct_config_uses_only_selected_arbitrary_camera(monkeypatch, tmp_path: Path):
    source = tmp_path / "relay.yml"
    base = tmp_path / "mediamtx.yml"
    app_output, media_output = tmp_path / "single.yml", tmp_path / "direct.yml"
    source.write_text(
        "cameras:\n"
        "  - {id: office-entry-v2, name: Entry, url: '${OFFICE_ENTRY_URL}'}\n"
        "  - {id: removed-camera, name: Retired, url: '${RETIRED_URL}', enabled: false}\n"
        "detection: {enabled: false}\n"
    )
    base.write_text("paths: {}\nrtsp: true\n")
    monkeypatch.setenv("OFFICE_ENTRY_URL", "rtsp://user:secret@example/entry")
    build(source, base, "office-entry-v2", "direct", app_output, media_output)
    app, media = yaml.safe_load(app_output.read_text()), yaml.safe_load(media_output.read_text())
    assert [item["enabled"] for item in app["cameras"]] == [True, False]
    assert app["video"]["h264_mode"] == "direct"
    assert app["video"]["mode"] == "diagnostic_transcode"
    assert list(media["paths"]) == ["live/office-entry-v2"]
    assert media["paths"]["live/office-entry-v2"]["source"].endswith("/entry")
    assert app_output.stat().st_mode & 0o077 == 0
    assert media_output.stat().st_mode & 0o077 == 0


def test_single_camera_copy_config_never_puts_source_in_mediamtx_file(monkeypatch, tmp_path: Path):
    source, base = tmp_path / "relay.yml", tmp_path / "mediamtx.yml"
    app_output, media_output = tmp_path / "single.yml", tmp_path / "copy.yml"
    source.write_text("cameras:\n  - {id: renamed, name: Renamed, url: '${RENAMED_URL}'}\n")
    base.write_text("paths: {}\n")
    monkeypatch.setenv("RENAMED_URL", "rtsp://user:secret@example/live")
    build(source, base, "renamed", "copy", app_output, media_output)
    assert yaml.safe_load(media_output.read_text())["paths"] == {"live/renamed": {}}
