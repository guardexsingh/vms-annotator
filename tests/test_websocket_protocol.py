import json
import socket
import struct
import threading

from app.metadata import MetadataHub
from app.metrics import Metrics
from app.models import DetectionResult
from app.websocket import MetadataWebSocketSession, encode_frame, websocket_accept


def test_websocket_handshake_matches_rfc_example():
    assert websocket_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_server_text_frame_contains_metadata_only_payload():
    frame = encode_frame(b'{"type":"detections","boxes":[]}', opcode=1)
    assert frame[0] == 0x81
    assert b"detections" in frame
    assert b"JPEG" not in frame and b"rtsp://" not in frame


def masked_client_frame(message: dict, opcode: int = 1) -> bytes:
    payload = json.dumps(message).encode()
    mask = b"test"
    assert len(payload) < 126
    encoded = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bytes([0x80 | opcode, 0x80 | len(payload)]) + mask + encoded


def read_server_message(connection: socket.socket) -> dict:
    header = connection.recv(2)
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", connection.recv(2))[0]
    payload = bytearray()
    while len(payload) < length:
        payload.extend(connection.recv(length - len(payload)))
    return json.loads(payload)


def test_one_websocket_subscribes_and_receives_latest_metadata():
    server, browser = socket.socketpair()
    browser.settimeout(2)
    metrics = Metrics()
    metrics.camera("cam01")
    hub = MetadataHub(["cam01"], metrics)
    session = MetadataWebSocketSession(hub, server, server.makefile("rb"), heartbeat_seconds=60)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    try:
        browser.sendall(masked_client_frame({"type": "subscribe", "camera_ids": ["cam01"]}))
        assert read_server_message(browser) == {"type": "subscribed", "camera_ids": ["cam01"]}
        assert read_server_message(browser) == {
            "type": "active_camera", "camera_id": None, "status": "disabled"
        }
        hub.publish(DetectionResult("cam01", (), 1, 1, 1, 0, 0, 0, 7, 1920, 1080))
        message = read_server_message(browser)
        assert message["type"] == "detections" and message["sequence"] == 7 and message["boxes"] == []
        browser.sendall(masked_client_frame({}, opcode=8))
        thread.join(timeout=2)
        assert not thread.is_alive()
    finally:
        browser.close()
        server.close()
