"""VLM-side policy server for the two-process CARLA bridge."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path
from typing import Any, Protocol

from vlm_driving.carla.observations import NormalizedAction
from vlm_driving.eval.protocol import make_action_response, make_error, make_ok, make_ready, recv_message, send_message


class PolicyAgent(Protocol):
    previous_action: tuple[float, float]

    def act(self, record: dict[str, Any], image: str | Path | None = None) -> NormalizedAction:
        ...


class PolicyServer:
    def __init__(
        self,
        agent: PolicyAgent,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout_s: float = 60.0,
    ) -> None:
        self.agent = agent
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.bound_port: int | None = None

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(4)
            self.bound_port = server_sock.getsockname()[1]
            while True:
                conn, _ = server_sock.accept()
                with conn:
                    conn.settimeout(self.timeout_s)
                    send_message(conn, make_ready())
                    if self._handle_connection(conn):
                        return

    def serve_once(self) -> None:
        self.serve_forever()

    def _handle_connection(self, conn: socket.socket) -> bool:
        while True:
            try:
                request = recv_message(conn)
            except (ConnectionError, TimeoutError, OSError):
                return False
            request_type = request.get("type")
            if request_type == "reset":
                self.agent.previous_action = (0.0, 0.0)
                send_message(conn, make_ok())
            elif request_type == "act":
                try:
                    record = request["record"]
                    frame_path = request["frame_path"]
                    action = self.agent.act(record, image=frame_path)
                    send_message(conn, make_action_response(action.steer, action.acceleration))
                except Exception as exc:  # server must report errors over the protocol.
                    send_message(conn, make_error(str(exc)))
            elif request_type == "shutdown":
                send_message(conn, make_ok())
                return True
            else:
                send_message(conn, make_error(f"unsupported request type: {request_type!r}"))


def build_agent(
    checkpoint_path: str | Path,
    command_text: str,
    device: str | None,
):
    from vlm_driving.carla.bc_agent import BCAgent

    return BCAgent(
        checkpoint_path=checkpoint_path,
        hidden_source="live",
        command_text=command_text,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLM-side BC policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/bc_smoke/bc_checkpoint.pt"))
    parser.add_argument("--command-text", default="You are driving in CARLA. Keep lane and continue safely.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    agent = build_agent(args.checkpoint, args.command_text, args.device)
    server = PolicyServer(agent=agent, host=args.host, port=args.port, timeout_s=args.timeout_s)
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = ["PolicyAgent", "PolicyServer", "build_agent"]
