from pathlib import Path

import yaml

from app.config import load_config
from app.detection_validation_config import build as build_validation_config
from app.ai_capture import AICaptureWorker
from app.latest_frame import LatestFrame
from app.metrics import Metrics
from app.models import CameraConfig
from app import main


PROJECT = Path(__file__).resolve().parents[1]


def test_default_video_paths_are_live_and_never_annotated():
    media = yaml.safe_load((PROJECT / "config" / "mediamtx.yml").read_text())
    assert media["paths"] == {}
    assert media["pathDefaults"]["sourceOnDemand"] is True
    assert "annotated/" not in (PROJECT / "app" / "native_relay.py").read_text()
    assert "annotated/" not in (PROJECT / "web" / "app.js").read_text()


def test_ai_capture_prefers_credential_free_local_mediamtx_path():
    camera = CameraConfig("renamed-entry", "Entry", "rtsp://user:secret@nvr/live",
                          detection_source="local_mediamtx")
    worker = AICaptureWorker(camera, LatestFrame(), Metrics())
    assert worker.source_url == "rtsp://127.0.0.1:18554/live/renamed-entry"
    assert "secret" not in worker.source_url


def test_enabling_detection_does_not_select_compositor_video(monkeypatch, tmp_path):
    config_file = tmp_path / "cameras.yml"
    config_file.write_text(
        "cameras:\n  - {id: entry, name: Entry, url: rtsp://example/live, detection_enabled: true}\n"
        "detection: {enabled: true}\npipeline: {mode: native}\n"
    )
    config = load_config(config_file)
    monkeypatch.setattr(main, "Compositor", lambda *args: (_ for _ in ()).throw(AssertionError("must not start")))
    relays, workers = main.build_video_components(config, Metrics())
    assert relays == []
    assert workers == []


def test_current_configuration_has_only_three_expected_cameras():
    config = yaml.safe_load((PROJECT / "config" / "cameras.yaml").read_text())
    assert [camera["id"] for camera in config["cameras"]] == ["cam01", "cam02", "cam03"]
    removed_id = "cam" + "04"
    source_files = [*PROJECT.glob("app/*.py"), *PROJECT.glob("web/*"), *PROJECT.glob("config/*")]
    assert all(removed_id not in path.read_text(errors="ignore") for path in source_files if path.is_file())


def test_locked_person_detection_configuration():
    config = load_config(PROJECT / "config" / "cameras.yaml",
                         {"CAM01_URL": "rtsp://one", "CAM02_URL": "rtsp://two", "CAM03_URL": "rtsp://three"})
    assert config.detection.enabled is True
    assert config.detection.model == "yolo11n.pt"
    assert config.detection.image_size == 640
    assert config.detection.classes == (0,)
    assert config.detection.confidence == 0.40
    assert config.detection.target_fps_per_camera == 1
    assert config.detection.result_ttl_ms == 2000
    assert config.detection.capture_fps == 1
    assert config.detection.inference_workers == 1
    assert config.detection.max_frame_age_ms == 1500
    assert config.tracking.tracker == "bytetrack"
    assert config.tracking.track_buffer == 2
    assert config.tracking.hold_box_ms == 1500
    assert config.tracking.remove_track_ms == 2000
    assert config.tracking.prediction_fps == 5
    assert config.detection.latest_frame_only is True
    assert all(camera.detection_source == "local_mediamtx" for camera in config.cameras)


def test_single_camera_validation_keeps_all_video_enabled(tmp_path):
    source, output = tmp_path / "cameras.yml", tmp_path / "validation.yml"
    source.write_text(
        "cameras:\n"
        "  - {id: entry, name: Entry, url: '${ENTRY_URL}', enabled: true}\n"
        "  - {id: bay, name: Bay, url: '${BAY_URL}', enabled: true}\n"
        "detection: {enabled: false}\n"
    )
    build_validation_config(source, "entry", output)
    raw = yaml.safe_load(output.read_text())
    assert [camera["enabled"] for camera in raw["cameras"]] == [True, True]
    assert [camera["detection_enabled"] for camera in raw["cameras"]] == [True, False]
    assert output.stat().st_mode & 0o077 == 0


def test_single_camera_video_validation_disables_other_video_and_detection(tmp_path):
    source, output = tmp_path / "cameras.yml", tmp_path / "validation.yml"
    source.write_text(
        "cameras:\n"
        "  - {id: entry, name: Entry, url: '${ENTRY_URL}', enabled: true}\n"
        "  - {id: bay, name: Bay, url: '${BAY_URL}', enabled: true}\n"
        "detection: {enabled: true}\n"
    )
    build_validation_config(source, "entry", output, only_video_camera=True)
    raw = yaml.safe_load(output.read_text())
    assert [camera["enabled"] for camera in raw["cameras"]] == [True, False]
    assert raw["detection"]["enabled"] is False
