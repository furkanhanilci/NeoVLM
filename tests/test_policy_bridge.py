from __future__ import annotations

import socket
import threading
import time
from typing import Any

import pytest

from vlm_driving.carla.observations import NormalizedAction
from vlm_driving.eval.policy_client import RemoteBCPolicy
from vlm_driving.eval.policy_server import PolicyServer
from vlm_driving.eval.protocol import decode_message, decode_request, decode_response, encode_message, encode_request, encode_response, make_act, recv_message, send_message


class DummyAgent:
    def __init__(self) -> None:
        self.previous_action = (0.5, -0.5)
        self.calls: list[tuple[dict[str, Any], str | None]] = []

    def act(self, record: dict[str, Any], image: str | None = None) -> NormalizedAction:
        self.calls.append((record, image))
        self.previous_action = (0.2, -0.3)
        return NormalizedAction(steer=2.0, acceleration=-2.0)


class FailingAgent(DummyAgent):
    def act(self, record: dict[str, Any], image: str | None = None) -> NormalizedAction:
        raise RuntimeError("intentional failure")


def test_protocol_length_prefixed_json_roundtrip():
    message = make_act(record={"ego": {"speed_mps": 1.0}}, frame_path="frame.png")
    encoded = encode_message(message)

    assert decode_message(encoded[4:]) == message
    assert decode_request(encode_request(message)[4:]) == message
    assert decode_response(encode_response({"type": "ok"})[4:]) == {"type": "ok"}


def test_protocol_socketpair_send_recv_roundtrip():
    left, right = socket.socketpair()
    try:
        send_message(left, {"type": "reset"})
        assert recv_message(right) == {"type": "reset"}
    finally:
        left.close()
        right.close()


def test_policy_bridge_loopback_act_reset_and_shutdown():
    agent = DummyAgent()
    server = PolicyServer(agent=agent, host="127.0.0.1", port=0, timeout_s=5.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = _wait_for_port(server)

    client = RemoteBCPolicy(host="127.0.0.1", port=port, timeout_s=5.0).connect()
    client.reset()
    assert agent.previous_action == (0.0, 0.0)
    action = client.act({"ego": {"speed_mps": 3.0}, "camera": {"path": "frames/frame.png"}}, "frames/frame.png")
    client.shutdown()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert action.steer == 1.0
    assert action.acceleration == -1.0
    assert agent.calls == [({"ego": {"speed_mps": 3.0}, "camera": {"path": "frames/frame.png"}}, "frames/frame.png")]


def test_policy_bridge_surfaces_server_errors():
    server = PolicyServer(agent=FailingAgent(), host="127.0.0.1", port=0, timeout_s=5.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = _wait_for_port(server)
    client = RemoteBCPolicy(host="127.0.0.1", port=port, timeout_s=5.0).connect()

    with pytest.raises(RuntimeError, match="intentional failure"):
        client.act({"ego": {}}, "frame.png")
    client.shutdown()
    thread.join(timeout=5.0)

    assert not thread.is_alive()


def _wait_for_port(server: PolicyServer) -> int:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if server.bound_port is not None:
            return server.bound_port
        time.sleep(0.01)
    raise RuntimeError("server did not bind a port")
