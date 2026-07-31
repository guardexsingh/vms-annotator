"""Black-box checks for the experiment's process-management shell scripts."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _project_copy(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    shutil.copytree(PROJECT / "scripts", root / "scripts")
    shutil.copytree(PROJECT / "config", root / "config")
    shutil.copytree(PROJECT / "app", root / "app")
    (root / "logs").mkdir(parents=True)
    (root / "run").mkdir()
    return root


def _write_env(root: Path, values: dict[str, str]) -> None:
    (root / ".env").write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    (root / ".env").chmod(0o600)


def _camera_values(*names: str) -> dict[str, str]:
    return {name: f"rtsp://example.invalid/{name.lower()}" for name in names}


def _write_camera_config(root: Path, cameras: list[tuple[str, str, bool]], name: str = "cameras.yaml") -> Path:
    lines = ["cameras:"]
    for camera_id, variable, enabled in cameras:
        lines.extend([f"  - id: {camera_id}", f"    name: {camera_id}", f"    url: '${{{variable}}}'",
                      f"    enabled: {'true' if enabled else 'false'}"])
    path = root / "config" / name
    path.write_text("\n".join(lines) + "\ndetection: {enabled: false}\n")
    return path


def _fake_services(root: Path) -> tuple[Path, Path, Path, Path]:
    marker = root / "fake-mediamtx.pid"
    app_ready = root / "fake-app.ready"
    media = root / "fake-mediamtx"
    app = root / "fake-python"
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        media,
        "#!/bin/sh\n"
        "printf '%s\\n' \"$$\" > \"$FAKE_MEDIAMTX_PID_FILE\"\n"
        "while :; do sleep 1; done\n",
    )
    _write_executable(
        app,
        "#!/bin/sh\n"
        "if [ \"$1\" = -c ]; then\n"
        "  case \"$2\" in\n"
        "    *required_camera_environment_variables*) exec /usr/bin/python3 \"$@\" ;;\n"
        "    *import\\ psutil*) exit 0 ;;\n"
        "    *sys.executable*) echo \"Python executable: $0\"; echo \"Python prefix: fake\"; exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "if [ \"$1\" = -m ] && { [ -n \"${PYTHONHOME:-}\" ] || [ -n \"${PYTHONPATH:-}\" ] || [ -n \"${VIRTUAL_ENV:-}\" ]; }; then exit 88; fi\n"
        "if [ \"$1\" = -m ] && [ \"$2\" = app.direct_video ] && [ \"$3\" = mode ]; then echo diagnostic_transcode; exit 0; fi\n"
        "if [ \"$1\" = -m ] && [ \"$2\" = app.main ] && [ -n \"${FAKE_PREDICTION_FPS_FILE:-}\" ]; then printf '%s' \"${BYTETRACK_PREDICTION_FPS:-}\" > \"$FAKE_PREDICTION_FPS_FILE\"; fi\n"
        "if [ \"${FAKE_APP_MODE:-success}\" = fail ]; then exit 42; fi\n"
        "sleep \"${FAKE_APP_DELAY:-0}\"\n"
        ": > \"$FAKE_APP_READY_FILE\"\n"
        "while :; do sleep 1; done\n",
    )
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _write_executable(venv_python, app.read_text())
    _write_executable(
        fake_bin / "curl",
        "#!/bin/sh\n"
        "for argument in \"$@\"; do\n"
        "  case \"$argument\" in\n"
        "    *healthz) test -f \"$FAKE_APP_READY_FILE\"; exit $? ;;\n"
        "    *19997*) exit 0 ;;\n"
        "  esac\n"
        "done\n"
        "exit 1\n",
    )
    _write_executable(fake_bin / "mediamtx", media.read_text())
    return media, app, marker, app_ready


def _run(root: Path, script: str, *arguments: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    if "FAKE_CURL_DIR" in env:
        env["PATH"] = f"{env['FAKE_CURL_DIR']}:{env['PATH']}"
    return subprocess.run(
        [str(root / "scripts" / script), *arguments],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )


def _stop_if_started(root: Path) -> None:
    _run(root, "stop.sh")


def test_start_rejects_missing_dotenv_before_services(tmp_path: Path):
    root = _project_copy(tmp_path)
    media, app, marker, app_ready = _fake_services(root)
    result = _run(root, "start.sh", MEDIAMTX_BIN=str(media), FAKE_MEDIAMTX_PID_FILE=str(marker))
    assert result.returncode != 0
    assert "Missing .env" in result.stderr
    assert not marker.exists()


def test_start_prints_only_the_missing_camera_name(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True), ("loading-bay", "LOADING_BAY_URL", True)])
    _write_env(root, _camera_values("LOADING_BAY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    result = _run(root, "start.sh", MEDIAMTX_BIN=str(media), FAKE_MEDIAMTX_PID_FILE=str(marker))
    assert result.returncode != 0
    assert result.stderr.splitlines() == ["OFFICE_ENTRY_URL"]
    assert not marker.exists()


def test_start_reports_all_missing_camera_names(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True), ("warehouse-west", "WAREHOUSE_WEST_URL", True),
                                ("front-desk", "FRONT_DESK_URL", True)])
    _write_env(root, _camera_values("OFFICE_ENTRY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    result = _run(root, "start.sh", MEDIAMTX_BIN=str(media), FAKE_MEDIAMTX_PID_FILE=str(marker))
    assert result.returncode != 0
    assert result.stderr.splitlines() == ["WAREHOUSE_WEST_URL", "FRONT_DESK_URL"]
    assert not marker.exists()


def test_disabled_camera_does_not_need_a_url_at_preflight(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True), ("removed-legacy-camera", "RETIRED_CAMERA_URL", False)])
    _write_env(root, _camera_values("OFFICE_ENTRY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    result = _run(
        root,
        "start.sh",
        MEDIAMTX_BIN=str(media),
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_CURL_DIR=str(root / "fake-bin"),
        HEALTH_TIMEOUT_SECONDS="3",
    )
    try:
        assert result.returncode == 0, result.stderr
        assert "Started experiment" in result.stdout
    finally:
        _stop_if_started(root)


def test_failed_app_start_cleans_up_mediamtx_and_never_reports_success(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True)])
    _write_env(root, _camera_values("OFFICE_ENTRY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    result = _run(
        root,
        "start.sh",
        MEDIAMTX_BIN=str(media),
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_CURL_DIR=str(root / "fake-bin"),
        FAKE_APP_MODE="fail",
        HEALTH_TIMEOUT_SECONDS="3",
    )
    assert result.returncode != 0
    assert "Started experiment" not in result.stdout
    assert "logs/app.log" in result.stderr
    assert "logs/mediamtx.log" in result.stderr
    assert not (root / "run" / "app.pid").exists()
    assert not (root / "run" / "mediamtx.pid").exists()
    if marker.exists():
        pid = int(marker.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("MediaMTX fake service survived failed startup")


def test_success_is_reported_only_after_app_health_passes(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True)])
    _write_env(root, _camera_values("OFFICE_ENTRY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    started = time.monotonic()
    result = _run(
        root,
        "start.sh",
        MEDIAMTX_BIN=str(media),
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_CURL_DIR=str(root / "fake-bin"),
        FAKE_APP_DELAY="0.5",
        HEALTH_TIMEOUT_SECONDS="3",
    )
    elapsed = time.monotonic() - started
    try:
        assert result.returncode == 0, result.stderr
        assert "Started experiment" in result.stdout
        assert elapsed >= 0.45
    finally:
        _stop_if_started(root)


def test_dotenv_python_controls_cannot_redirect_or_break_isolated_runtime(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True)])
    _write_env(root, {**_camera_values("OFFICE_ENTRY_URL"), "PYTHON_BIN": "/usr/bin/python3",
                      "PYTHONPATH": "/tmp/example", "PYTHONHOME": "/tmp/example", "VIRTUAL_ENV": "/tmp/example",
                      "ROOT": "/tmp/example", "MEDIAMTX_BIN": "/tmp/example"})
    media, app, marker, app_ready = _fake_services(root)
    result = _run(
        root,
        "start.sh",
        MEDIAMTX_BIN=str(media),
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_CURL_DIR=str(root / "fake-bin"),
    )
    try:
        assert result.returncode == 0, result.stderr
        expected_python = root / ".venv" / "bin" / "python"
        assert f"Python executable: {expected_python}" in result.stdout
        assert "Python prefix: fake" in result.stdout
    finally:
        _stop_if_started(root)


def test_start_preserves_explicit_prediction_rate_for_the_python_process(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True)])
    _write_env(root, {**_camera_values("OFFICE_ENTRY_URL"), "BYTETRACK_PREDICTION_FPS": "5"})
    media, app, marker, app_ready = _fake_services(root)
    observed = root / "prediction-fps"
    result = _run(
        root,
        "start.sh",
        BYTETRACK_PREDICTION_FPS="10",
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_PREDICTION_FPS_FILE=str(observed),
        FAKE_CURL_DIR=str(root / "fake-bin"),
    )
    try:
        assert result.returncode == 0, result.stderr
        assert observed.read_text() == "10"
    finally:
        _stop_if_started(root)


def test_relay_benchmark_validates_its_own_configuration(tmp_path: Path):
    root = _project_copy(tmp_path)
    _write_camera_config(root, [("office-entry", "OFFICE_ENTRY_URL", True)], "cameras.yaml")
    _write_camera_config(root, [("loading-bay", "LOADING_BAY_URL", True)], "cameras.relay.yaml")
    _write_env(root, _camera_values("OFFICE_ENTRY_URL"))
    media, app, marker, app_ready = _fake_services(root)
    result = _run(
        root,
        "relay_benchmark.sh",
        "0",
        MEDIAMTX_BIN=str(media),
        FAKE_MEDIAMTX_PID_FILE=str(marker),
        FAKE_APP_READY_FILE=str(app_ready),
        FAKE_CURL_DIR=str(root / "fake-bin"),
        HEALTH_TIMEOUT_SECONDS="3",
    )
    assert result.returncode != 0
    assert result.stderr.splitlines() == ["LOADING_BAY_URL"]


def test_status_identifies_stale_pid_files_and_unready_live_process(tmp_path: Path):
    root = _project_copy(tmp_path)
    (root / "run" / "app.pid").write_text("999999\n")
    (root / "run" / "mediamtx.pid").write_text("not-a-pid\n")
    stale = _run(root, "status.sh")
    assert "app: PID file is stale" in stale.stdout
    assert "mediamtx: PID file is stale" in stale.stdout

    sleeper = subprocess.Popen(["sleep", "5"])
    try:
        (root / "run" / "app.pid").write_text(f"{sleeper.pid}\n")
        (root / "run" / "mediamtx.pid").unlink()
        live = _run(
            root,
            "status.sh",
            APP_HEALTH_URL="http://127.0.0.1:1/healthz",
        )
        assert "app: process running" in live.stdout
        assert "HTTP service not ready" in live.stdout
        assert "mediamtx: process stopped" in live.stdout
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_healthcheck_requires_application_and_mediamtx_api(tmp_path: Path):
    root = _project_copy(tmp_path)
    result = _run(
        root,
        "healthcheck.sh",
        APP_HEALTH_URL="http://127.0.0.1:1/healthz",
        MEDIAMTX_API_URL="http://127.0.0.1:1/v3/config/global/get",
    )
    assert result.returncode != 0
    assert "application health: unavailable" in result.stderr
    assert "MediaMTX API: unavailable" in result.stderr


def test_unused_mediamtx_protocols_are_disabled():
    config = yaml.safe_load((PROJECT / "config" / "mediamtx.yml").read_text())
    assert config["rtsp"] is True
    assert config["rtspAddress"] == ":18554"
    assert config["webrtc"] is True
    assert config["webrtcAddress"] == ":18889"
    assert config["webrtcLocalUDPAddress"] == ":18189"
    assert config["api"] is True
    assert config["apiAddress"] == ":19997"
    for protocol in ("rtmp", "hls", "srt", "moq"):
        assert config[protocol] is False
