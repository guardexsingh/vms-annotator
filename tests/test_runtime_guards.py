from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from app.health import health_payload
from app.latest_frame import LatestFrame
from app.metrics import Metrics
from app.models import Frame
from app.yolo_detector import DetectorUnavailable, YoloPersonDetector


def test_numpy_major_version_is_compatible():
    assert int(np.__version__.split(".")[0]) == 1


def test_matplotlib_and_ultralytics_import_from_isolated_environment():
    import matplotlib
    from ultralytics import YOLO

    root = Path(sys.prefix).resolve()
    assert root.name == ".venv"
    assert str(root) in matplotlib.__file__
    assert YOLO is not None


def test_failed_detector_state_makes_health_degraded(monkeypatch):
    metrics = Metrics()
    metrics.set_detector("failed", "ImportError: redacted")
    monkeypatch.setattr("app.health.urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    status, payload = health_payload(metrics)
    assert status == 503
    assert payload["detector"]["status"] == "failed"
    assert payload["status"] == "degraded"
    assert "redacted" in payload["detector_error"]


def test_permanent_detector_failure_is_backed_off_without_reimport(monkeypatch):
    detector = YoloPersonDetector("missing.pt", 32, 0.4)
    detector.state = "failed"
    detector.error_summary = "ImportError: unavailable"
    detector._next_retry = time.monotonic() + 60
    frame = Frame(np.zeros((1, 1, 3), dtype=np.uint8), time.monotonic(), 1)
    for _ in range(10):
        with pytest.raises(DetectorUnavailable):
            detector.detect("cam01", frame)
    assert detector._failures == 0


def test_detection_disabled_never_constructs_a_detector(monkeypatch, tmp_path: Path):
    from app import main

    config = tmp_path / "relay.yml"
    config.write_text("""cameras:\n  - {id: cam01, name: Cam, url: '${CAM01_URL}'}\ndetection: {enabled: false}\n""")
    monkeypatch.setattr(sys, "argv", ["app.main", "--config", str(config), "--detector-only"])
    monkeypatch.setattr(main, "YoloPersonDetector", lambda *args: (_ for _ in ()).throw(AssertionError("must not load")))
    assert main.main() == 0


def test_latest_frame_queue_is_bounded():
    slot = LatestFrame[int]()
    for value in range(100):
        slot.put(value)
        assert slot.depth == 1
    assert slot.take() == 99
    assert slot.depth == 0


def test_start_script_defaults_to_project_venv_python():
    script = (Path(__file__).parents[1] / "scripts" / "start.sh").read_text()
    assert 'readonly VENV_PYTHON="$ROOT/.venv/bin/python"' in script
    assert "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV" in script
    assert "import psutil; import yaml; import cv2" in script
    assert "PYTHONNOUSERSITE=1" in script
    assert 'export MPLCONFIGDIR YOLO_CONFIG_DIR XDG_CONFIG_HOME' in script
