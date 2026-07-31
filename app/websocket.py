"""Small RFC 6455 server adapter for the metadata-only endpoint."""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from typing import BinaryIO

from .metadata import MetadataHub

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def websocket_accept(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")


def encode_frame(payload: bytes, opcode: int = 1) -> bytes:
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.extend((126, *struct.pack("!H", length)))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def read_frame(stream: BinaryIO) -> tuple[int, bytes]:
    header = stream.read(2)
    if len(header) != 2:
        raise ConnectionError("WebSocket closed")
    first, second = header
    if not first & 0x80:
        raise ValueError("Fragmented WebSocket frames are unsupported")
    opcode, masked, length = first & 0x0F, bool(second & 0x80), second & 0x7F
    if length == 126:
        length = struct.unpack("!H", stream.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", stream.read(8))[0]
    if length > 1_048_576:
        raise ValueError("WebSocket message too large")
    if not masked:
        raise ValueError("Client WebSocket frames must be masked")
    mask = stream.read(4)
    payload = stream.read(length)
    if len(mask) != 4 or len(payload) != length:
        raise ConnectionError("Truncated WebSocket frame")
    return opcode, bytes(value ^ mask[index % 4] for index, value in enumerate(payload))


class MetadataWebSocketSession:
    """One reader and one bounded-mailbox sender for a browser connection."""

    def __init__(self, hub: MetadataHub, connection: socket.socket, stream: BinaryIO,
                 heartbeat_seconds: float = 20.0) -> None:
        self.hub, self.connection, self.stream, self.heartbeat_seconds = hub, connection, stream, heartbeat_seconds
        self.client = hub.add_client()
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self._last_pong = time.monotonic()

    def _send(self, payload: bytes, opcode: int = 1) -> None:
        with self._send_lock:
            self.connection.sendall(encode_frame(payload, opcode))

    def _sender(self) -> None:
        last_ping = time.monotonic()
        try:
            while not self._closed.is_set():
                message = self.client.mailbox.take(timeout=1.0)
                if message is not None:
                    self._send(self.hub.encode(message).encode("utf-8"))
                now = time.monotonic()
                if now - last_ping >= self.heartbeat_seconds:
                    self._send(b"metadata", opcode=9)
                    last_ping = now
                if now - self._last_pong >= self.heartbeat_seconds * 3:
                    raise TimeoutError("WebSocket heartbeat timed out")
        except (OSError, ConnectionError, TimeoutError):
            self._closed.set()
        finally:
            if self._closed.is_set():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def run(self) -> None:
        sender = threading.Thread(target=self._sender, name="metadata-ws-sender", daemon=True)
        sender.start()
        try:
            while not self._closed.is_set():
                opcode, payload = read_frame(self.stream)
                if opcode == 8:
                    break
                if opcode == 9:
                    self._send(payload, opcode=10)
                    continue
                if opcode == 10:
                    self._last_pong = time.monotonic()
                    continue
                if opcode != 1:
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                    if message.get("type") != "subscribe" or not isinstance(message.get("camera_ids"), list):
                        raise ValueError("Expected a subscribe message")
                    self.hub.subscribe(self.client, [str(value) for value in message["camera_ids"]])
                    self._send(self.hub.encode({"type": "subscribed",
                                                "camera_ids": sorted(self.client.subscriptions)}).encode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    self._send(self.hub.encode({"type": "error", "error": str(error)[:160]}).encode("utf-8"))
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            self._closed.set()
            self.hub.remove_client(self.client)
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sender.join(timeout=1.0)
