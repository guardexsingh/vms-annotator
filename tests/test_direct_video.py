from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from app import main
from app.config import load_config
from app.direct_video import generate_direct_config, load_validation, verify_direct_paths
from app.metrics import Metrics
from app.models import StreamInfo


PROJECT = Path(__file__).resolve().parents[1]


def _config(path: Path) -> None:
    path.write_text(
        "cameras:\n"
        "  - {id: office-entry-v2, name: Entry, url: '${OFFICE_ENTRY_URL}', enabled: true}\n"
        "  - {id: retired-west, name: Retired, url: '${REMOVED_URL}', enabled: false}\n"
        "  - {id: loading-bay, name: Bay, url: '${BAY_FEED}', enabled: true, rtsp_transport: udp}\n"
        "detection: {enabled: false}\n"
        "video: {mode: direct_hevc, allow_transcode_fallback: false}\n"
    )


def test_private_direct_paths_are_dynamic_exact_and_mode_0600(monkeypatch, tmp_path: Path):
    config, base, output = tmp_path / "cameras.yml", tmp_path / "base.yml", tmp_path / "direct.yml"
    _config(config)
    base.write_text("rtsp: true\npathDefaults: {sourceOnDemand: true}\npaths: {}\n")
    monkeypatch.setenv("OFFICE_ENTRY_URL", "rtsp://user:office-secret@example/entry")
    monkeypatch.setenv("BAY_FEED", "rtsp://user:bay-secret@example/bay")
    generated = generate_direct_config(config, base, output)
    payload = yaml.safe_load(output.read_text())
    assert [camera.id for camera in generated.cameras if camera.enabled] == ["office-entry-v2", "loading-bay"]
    assert list(payload["paths"]) == ["live/office-entry-v2", "live/loading-bay"]
    assert "retired-west" not in output.read_text()
    assert payload["paths"]["live/loading-bay"]["rtspTransport"] == "udp"
    assert all(path["sourceOnDemand"] is True for path in payload["paths"].values())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_direct_validation_checks_source_and_local_mediamtx_as_hevc(monkeypatch, tmp_path: Path):
    config, output = tmp_path / "cameras.yml", tmp_path / "validation.json"
    _config(config)
    monkeypatch.setenv("OFFICE_ENTRY_URL", "rtsp://example/entry")
    monkeypatch.setenv("BAY_FEED", "rtsp://example/bay")
    calls: list[tuple[str, str]] = []

    def probe(url: str, transport: str, timeout: float):
        calls.append((url, transport))
        return StreamInfo("hevc", 1920, 1080, 25.0)

    monkeypatch.setattr("app.direct_video.probe_stream", probe)
    rows = verify_direct_paths(config, output)
    assert len(rows) == 2
    assert any(url == "rtsp://127.0.0.1:18554/live/office-entry-v2" for url, _ in calls)
    assert all(row["source_codec"] == row["mediamtx_codec"] == "hevc" for row in rows)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert load_validation(output)["loading-bay"]["transcoding"] is False


def test_h264_source_is_rejected_without_fallback(monkeypatch, tmp_path: Path):
    config, output = tmp_path / "cameras.yml", tmp_path / "validation.json"
    config.write_text(
        "cameras:\n  - {id: renamed, name: Renamed, url: '${RENAMED_URL}'}\n"
        "detection: {enabled: false}\nvideo: {mode: direct_hevc, allow_transcode_fallback: false}\n"
    )
    monkeypatch.setenv("RENAMED_URL", "rtsp://example/live")
    monkeypatch.setattr("app.direct_video.probe_stream",
                        lambda *_args, **_kwargs: StreamInfo("h264", 1920, 1080, 25.0))
    with pytest.raises(ValueError, match="source codec is h264"):
        verify_direct_paths(config, output)
    assert not output.exists()


def test_default_runtime_builds_no_video_relay_or_python_video_worker(tmp_path: Path):
    config = tmp_path / "cameras.yml"
    config.write_text("cameras:\n  - {id: renamed, name: Renamed, url: rtsp://example/live}\n")
    relays, workers = main.build_video_components(load_config(config), Metrics())
    assert relays == []
    assert workers == []


def test_detection_toggle_does_not_change_direct_video_components(tmp_path: Path):
    disabled, enabled = tmp_path / "off.yml", tmp_path / "on.yml"
    body = "cameras:\n  - {id: renamed, name: Renamed, url: rtsp://example/live}\n"
    disabled.write_text(body + "detection: {enabled: false}\n")
    enabled.write_text(body + "detection: {enabled: true}\n")
    assert main.build_video_components(load_config(disabled), Metrics()) == ([], [])
    assert main.build_video_components(load_config(enabled), Metrics()) == ([], [])


def test_direct_metrics_do_not_expose_encoder_telemetry():
    metrics = Metrics()
    camera = metrics.camera("renamed")
    camera.source_codec = camera.mediamtx_codec = "hevc"
    camera.source_fps = camera.mediamtx_fps = 25
    payload = metrics.snapshot()["cameras"]["renamed"]
    assert payload["video_path"] == "direct"
    assert payload["transcoding"] is False
    assert payload["source_codec"] == payload["mediamtx_codec"] == "hevc"
    assert "encoder" not in payload
    assert "encoder_command" not in payload
    assert "encoded_fps" not in payload
    assert payload["glass_to_glass_latency_ms"] is None


def test_checked_in_template_and_startup_have_no_source_urls_or_default_relay():
    media = yaml.safe_load((PROJECT / "config" / "mediamtx.yml").read_text())
    assert media["paths"] == {}
    assert "source:" not in (PROJECT / "config" / "mediamtx.yml").read_text()
    script = (PROJECT / "scripts" / "start.sh").read_text()
    assert "app.direct_video generate" in script
    assert "app.direct_video verify" in script
    assert "app.main" in script
    assert script.index("app.direct_video verify") < script.index("app.main")
    normal = (PROJECT / "config" / "cameras.yaml").read_text().lower()
    assert "mode: direct_hevc" in normal
    assert "codec: h264" not in normal


def test_direct_validation_artifact_contains_no_camera_urls(tmp_path: Path):
    payload = {"cameras": [{"camera_id": "renamed", "stream_path": "live/renamed",
                            "source_codec": "hevc", "mediamtx_codec": "hevc",
                            "width": 1, "height": 1, "source_fps": 25,
                            "mediamtx_fps": 25, "video_path": "direct",
                            "transcoding": False}]}
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(payload))
    assert "rtsp://" not in path.read_text()
