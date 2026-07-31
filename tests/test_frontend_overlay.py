from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def run_js(expression: str):
    script = f"const app=require('./web/app.js'); console.log(JSON.stringify({expression}));"
    completed = subprocess.run(["node", "-e", script], cwd=PROJECT, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def test_matching_aspect_ratio_mapping():
    rect = run_js("app.calculateVideoRect(1600,900,1920,1080)")
    assert rect == {"x": 0, "y": 0, "width": 1600, "height": 900}


def test_letterbox_mapping_avoids_black_bars():
    rect = run_js("app.calculateVideoRect(1000,1000,1920,1080)")
    assert [rect[key] for key in ("x", "y", "width", "height")] == pytest.approx([0, 218.75, 1000, 562.5])
    box = run_js("app.mapNormalizedBox({x:.25,y:.2,width:.5,height:.4},app.calculateVideoRect(1000,1000,1920,1080))")
    assert [box[key] for key in ("x", "y", "width", "height")] == pytest.approx([250, 331.25, 500, 225])


def test_pillarbox_mapping_avoids_black_bars():
    rect = run_js("app.calculateVideoRect(1600,900,720,1280)")
    assert rect == {"x": 546.875, "y": 0, "width": 506.25, "height": 900}


def test_canvas_scaling_uses_device_pixel_ratio():
    assert run_js("app.canvasPixelSize(640,360,2)") == {"width": 1280, "height": 720, "ratio": 2}


def test_detection_ttl_expires_boxes():
    expression = "app.isExpired({metadata_age_ms:500,receivedAt:1000},750,1300)"
    assert run_js(expression) is True


def test_lost_track_box_obeys_confirmation_hold_time():
    result = "{receivedAt:1000,boxes:[{track_state:'lost',last_confirmed_age_ms:1000},{track_state:'active',last_confirmed_age_ms:5000}]}"
    assert len(run_js(f"app.visibleTrackBoxes({result},1500,1400)")) == 2
    visible = run_js(f"app.visibleTrackBoxes({result},1500,1600)")
    assert len(visible) == 1
    assert visible[0]["track_state"] == "active"


def test_subscription_deduplicates_camera_ids():
    assert run_js("app.buildSubscription(['cam01','cam01','cam02'])") == {
        "type": "subscribe", "camera_ids": ["cam01", "cam02"]
    }


def test_control_states_and_frontend_request_contract_are_explicit():
    source = (PROJECT / "web" / "app.js").read_text()
    styles = (PROJECT / "web" / "styles.css").read_text()
    assert run_js("app.controlState('loading')") == "starting"
    assert run_js("app.controlState('ready')") == "active"
    assert 'headers:{"Content-Type":"application/json"}' in source
    assert "event.preventDefault(); event.stopPropagation();" in source
    assert "[detection-control]" in source
    assert 'data-state="starting"' in styles
    assert 'data-state="active"' in styles
    assert 'data-state="error"' in styles
    assert ".camera-overlay" in styles and "z-index:2" in styles
    assert "z-index:4" in styles


def test_resize_and_reconnect_hooks_are_present():
    source = (PROJECT / "web" / "app.js").read_text()
    assert "new ResizeObserver" in source
    assert 'addEventListener("resize"' in source
    assert 'addEventListener("loadedmetadata"' in source
    assert 'document.addEventListener("fullscreenchange"' in source
    assert "let subscribed=false" in source


def test_browser_hevc_capability_and_sdp_codec_checks():
    assert run_js("app.supportsHevcCodec({codecs:[{mimeType:'video/H265'}]})") is True
    assert run_js("app.supportsHevcCodec({codecs:[{mimeType:'video/H264'}]})") is False
    hevc = "'m=video 9 UDP/TLS/RTP/SAVPF 98\\r\\na=rtpmap:98 H265/90000\\r\\n'"
    h264 = "'m=video 9 UDP/TLS/RTP/SAVPF 96\\r\\na=rtpmap:96 H264/90000\\r\\n'"
    assert run_js(f"app.negotiatedVideoCodec({hevc})") == "hevc"
    assert run_js(f"app.negotiatedVideoCodec({h264})") == "h264"


def test_browser_refuses_codec_fallback_and_reports_rtp_stats():
    source = (PROJECT / "web" / "app.js").read_text()
    assert "no H.264 fallback" in source
    assert "RTCRtpReceiver" in source
    assert 'fetch("/api/browser-stats"' in source


def test_debug_view_exposes_distinct_yolo_and_prediction_status():
    source = (PROJECT / "web" / "app.js").read_text()
    assert "actual_tracker_fps" in source
    assert "configured_bytetrack_prediction_fps" in source
    assert "Source: ${source}" in source
    assert "Confirmed age:" in source
