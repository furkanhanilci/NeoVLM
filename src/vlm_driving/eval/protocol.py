"""Pure-Python socket protocol for CARLA <-> VLM policy bridge."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!I")


def encode_message(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message too large: {len(payload)} bytes")
    return _HEADER.pack(len(payload)) + payload


def decode_message(payload: bytes) -> dict[str, Any]:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("bridge message must decode to a JSON object")
    return data


def encode_request(message: dict[str, Any]) -> bytes:
    return encode_message(message)


def decode_request(payload: bytes) -> dict[str, Any]:
    return decode_message(payload)


def encode_response(message: dict[str, Any]) -> bytes:
    return encode_message(message)


def decode_response(payload: bytes) -> dict[str, Any]:
    return decode_message(payload)


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"incoming bridge message too large: {length} bytes")
    return decode_message(_recv_exact(sock, length))


def make_ready() -> dict[str, Any]:
    return {"type": "ready", "ready": True}


def make_reset() -> dict[str, Any]:
    return {"type": "reset"}


def make_shutdown() -> dict[str, Any]:
    return {"type": "shutdown"}


def make_act(record: dict[str, Any], frame_path: str) -> dict[str, Any]:
    return {"type": "act", "record": record, "frame_path": frame_path}


def make_action_response(steer: float, acceleration: float) -> dict[str, Any]:
    return {"type": "action", "steer": float(steer), "acceleration": float(acceleration)}


def make_ok() -> dict[str, Any]:
    return {"type": "ok"}


def make_error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading bridge message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "MAX_MESSAGE_BYTES",
    "decode_message",
    "decode_request",
    "decode_response",
    "encode_message",
    "encode_request",
    "encode_response",
    "make_act",
    "make_action_response",
    "make_error",
    "make_ok",
    "make_ready",
    "make_reset",
    "make_shutdown",
    "recv_message",
    "send_message",
]
