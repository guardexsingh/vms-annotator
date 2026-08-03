from __future__ import annotations

import json
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .websocket import MetadataWebSocketSession, websocket_accept


def set_active_camera_from_payload(runtime, payload: object) -> dict[str, object]:
    """Validate the public control payload without ever handling camera URLs."""
    if runtime is None:
        raise RuntimeError("Detection is disabled")
    if not isinstance(payload, dict) or set(payload) != {"camera_id"}:
        raise ValueError("Expected only camera_id")
    camera_id = payload["camera_id"]
    if camera_id is not None and (not isinstance(camera_id, str) or not camera_id):
        raise ValueError("camera_id must be a non-empty string or null")
    return runtime.set_active_camera(camera_id)


def health_payload(metrics, mediamtx_url: str | None = "http://127.0.0.1:19997/v3/config/global/get",
                   runtime_mode: str = "standalone") -> tuple[int, dict]:
    payload = metrics.snapshot()
    if runtime_mode == "metadata_only":
        mediamtx = "not_applicable"
    else:
        try:
            with urllib.request.urlopen(mediamtx_url, timeout=0.5) as response:
                mediamtx = "ready" if 200 <= response.status < 300 else "unavailable"
        except (OSError, urllib.error.URLError):
            mediamtx = "unavailable"
    detector = payload["detector"]
    ready = runtime_mode == "metadata_only" or mediamtx == "ready"
    video = {camera_id: value.get("video_status", "ready" if value["online"] else "offline")
             for camera_id, value in payload["cameras"].items()}
    return (HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
            {**payload, "status": "ok" if ready else "degraded", "http": "ready", "mediamtx": mediamtx,
             "video": video, "detector": {"status": detector["state"], **detector},
             "detector_error": detector["error"], "runtime_mode": runtime_mode,
             "active_camera": {"camera_id": detector.get("active_camera_id"),
                               "status": detector["state"]},
             "capture": {camera_id: value.get("ai_capture_status", "disabled")
                         for camera_id, value in payload["cameras"].items()}})


def effective_detection_public(effective_config) -> dict[str, object]:
    """Public, secret-free effective targets shared with the frontend."""
    return {
        "requested_backend": getattr(getattr(effective_config, "detection", None), "backend", None),
        "requested_inference_fps": getattr(getattr(effective_config, "detection", None), "target_fps_per_camera", None),
        "requested_ai_capture_fps": getattr(getattr(effective_config, "detection", None), "capture_fps", None),
        "requested_precision": getattr(getattr(effective_config, "detection", None), "precision", None),
        "configured_bytetrack_prediction_fps": getattr(getattr(effective_config, "tracking", None), "prediction_fps", None),
    }


class HealthServer:
    def __init__(self, host: str, port: int, metrics, web_root: Path, mediamtx_url: str | None = None,
                 metadata_hub=None, metadata_path: str = "/ws/detections", cameras=(), ttl_ms: int = 750,
                 whep_port: int = 18889, video_mode: str = "direct_hevc", runtime_mode: str = "standalone",
                 detection_runtime=None, tracking_config=None, effective_config=None) -> None:
        self.metrics, self.web_root = metrics, web_root
        self.runtime_mode = runtime_mode
        self.mediamtx_url = mediamtx_url or "http://127.0.0.1:19997/v3/config/global/get"
        self.metadata_hub, self.metadata_path = metadata_hub, metadata_path
        self.detection_runtime = detection_runtime
        self.effective_config = effective_config
        self.public_cameras = [
            ({"id": camera.id, "name": camera.name, "detection_enabled": camera.detection_enabled,
              "runtime_mode": runtime_mode}
             if runtime_mode == "metadata_only" else
             {"id": camera.id, "name": camera.name, "stream_path": f"live/{camera.id}",
              "detection_enabled": camera.detection_enabled,
              "video_path": "direct" if video_mode == "direct_hevc" else "diagnostic",
              "expected_codec": "hevc" if video_mode == "direct_hevc" else "h264",
              "transcoding": video_mode != "direct_hevc"})
            for camera in cameras if camera.enabled
        ]
        self.ttl_ms, self.whep_port = ttl_ms, whep_port
        self.tracking_public = {
            "hold_box_ms": getattr(tracking_config, "hold_box_ms", ttl_ms),
            "remove_track_ms": getattr(tracking_config, "remove_track_ms", ttl_ms),
            "prediction_fps": getattr(tracking_config, "prediction_fps", 5.0),
            "yaml_prediction_fps": getattr(tracking_config, "yaml_prediction_fps", 5.0),
            "prediction_fps_environment_override": getattr(tracking_config, "prediction_fps_environment_override", None),
            "prediction_fps_source": getattr(tracking_config, "prediction_fps_source", "built-in default"),
            "debug_labels": False,
        }
        self.detection_public = effective_detection_public(effective_config)
        outer = self
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(outer.web_root), **kwargs)
            def do_GET(self):
                if self.path == outer.metadata_path and self.headers.get("Upgrade", "").lower() == "websocket":
                    key = self.headers.get("Sec-WebSocket-Key", "")
                    if not key or outer.metadata_hub is None:
                        self.send_error(HTTPStatus.BAD_REQUEST)
                        return
                    self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
                    self.send_header("Upgrade", "websocket")
                    self.send_header("Connection", "Upgrade")
                    self.send_header("Sec-WebSocket-Accept", websocket_accept(key))
                    self.end_headers()
                    self.close_connection = True
                    MetadataWebSocketSession(outer.metadata_hub, self.connection, self.rfile).run()
                    return
                if self.path in {"/healthz", "/metrics"}:
                    status, payload = health_payload(
                        outer.metrics,
                        None if outer.runtime_mode == "metadata_only" else outer.mediamtx_url,
                        outer.runtime_mode,
                    )
                    self._json(status, payload)
                    return
                if self.path == "/api/cameras":
                    active = (
                        outer.detection_runtime.state()
                        if outer.detection_runtime is not None
                        else {"camera_id": None, "status": "disabled"}
                    )
                    response = {
                        "cameras": outer.public_cameras,
                        "detection_ttl_ms": outer.ttl_ms,
                        "tracking": outer.tracking_public,
                        "detection": outer.detection_public,
                        "active_camera": active,
                        "metadata_path": outer.metadata_path,
                    }
                    if outer.runtime_mode != "metadata_only":
                        response["whep_port"] = outer.whep_port
                    self._json(HTTPStatus.OK, response)
                    return
                if self.path == "/api/detection/active-camera":
                    state = (
                        outer.detection_runtime.state()
                        if outer.detection_runtime is not None
                        else {"camera_id": None, "status": "disabled"}
                    )
                    self._json(HTTPStatus.OK, state)
                    return
                return super().do_GET()
            def do_POST(self):
                if self.path == "/api/detection/active-camera":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length < 2 or length > 512:
                            raise ValueError("invalid body size")
                        payload = json.loads(self.rfile.read(length))
                        state = set_active_camera_from_payload(
                            outer.detection_runtime, payload
                        )
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                        self._json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Invalid or unavailable camera_id"},
                        )
                        return
                    except RuntimeError:
                        self._json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"error": "Detection runtime is unavailable"},
                        )
                        return
                    self._json(HTTPStatus.OK, state)
                    return
                if self.path != "/api/browser-stats":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 2 or length > 4096:
                        raise ValueError("invalid body size")
                    payload = json.loads(self.rfile.read(length))
                    camera_id = str(payload.pop("camera_id"))
                    outer.metrics.record_browser_stats(camera_id, payload)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            def _json(self, status, payload):
                body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def end_headers(self):
                # Controls and overlays are a single tightly coupled client;
                # never let an old cached app.js silently control a new server.
                self.send_header("Cache-Control", "no-store")
                super().end_headers()
            def log_message(self, format, *args):
                return
        self.server = ThreadingHTTPServer((host, port), Handler)

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def stop(self) -> None:
        self.server.shutdown()
