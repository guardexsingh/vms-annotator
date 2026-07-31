from http import HTTPStatus

from app.health import health_payload
from app.metrics import Metrics


class ReadyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def ready_metrics(detector_state: str) -> Metrics:
    metrics = Metrics()
    camera = metrics.camera("cam01")
    camera.online = True
    camera.video_status = "ready"
    metrics.set_detector(detector_state, "detector unavailable" if detector_state == "failed" else None)
    return metrics


def test_video_readiness_does_not_wait_for_detector(monkeypatch):
    monkeypatch.setattr("app.health.urllib.request.urlopen", lambda *args, **kwargs: ReadyResponse())
    status, payload = health_payload(ready_metrics("loading"))
    assert status == HTTPStatus.OK
    assert payload["video"] == {"cam01": "ready"}
    assert payload["detector"]["status"] == "loading"


def test_detector_failure_leaves_video_health_ready(monkeypatch):
    monkeypatch.setattr("app.health.urllib.request.urlopen", lambda *args, **kwargs: ReadyResponse())
    status, payload = health_payload(ready_metrics("failed"))
    assert status == HTTPStatus.OK
    assert payload["status"] == "ok"
    assert payload["video"]["cam01"] == "ready"
    assert payload["detector"]["status"] == "failed"

