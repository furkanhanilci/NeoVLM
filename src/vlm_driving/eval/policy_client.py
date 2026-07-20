"""Torch-free client for the remote BC policy server."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from vlm_driving.carla.observations import NormalizedAction
from vlm_driving.eval.protocol import make_act, make_reset, make_shutdown, recv_message, send_message


class RemoteBCPolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 30.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def connect(self) -> "RemoteBCPolicy":
        if self._sock is not None:
            return self
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        ready = recv_message(sock)
        if ready.get("type") != "ready":
            sock.close()
            raise RuntimeError(f"policy server did not send ready message: {ready}")
        self._sock = sock
        return self

    def reset(self) -> None:
        sock = self._connected_socket()
        send_message(sock, make_reset())
        response = recv_message(sock)
        self._require_ok(response)

    def act(self, record: dict[str, Any], frame_path: str | Path) -> NormalizedAction:
        sock = self._connected_socket()
        send_message(sock, make_act(record=record, frame_path=str(frame_path)))
        response = recv_message(sock)
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("message", "policy server error")))
        if response.get("type") != "action":
            raise RuntimeError(f"unexpected policy server response: {response}")
        return NormalizedAction(
            steer=float(response["steer"]),
            acceleration=float(response["acceleration"]),
        ).clipped()

    def shutdown(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            send_message(sock, make_shutdown())
            recv_message(sock)
        finally:
            self.close()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "RemoteBCPolicy":
        return self.connect()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _connected_socket(self) -> socket.socket:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        return self._sock

    @staticmethod
    def _require_ok(response: dict[str, Any]) -> None:
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("message", "policy server error")))
        if response.get("type") != "ok":
            raise RuntimeError(f"unexpected policy server response: {response}")


__all__ = ["RemoteBCPolicy"]
