"""Minimal client for the Ableton MCP Remote Script (raw JSON over TCP, one request per connection)."""

from __future__ import annotations

import json
import socket
from typing import Any


class BridgeError(RuntimeError):
    """Ableton is unreachable or rejected the command."""


class AbletonBridge:
    """Talks to ``127.0.0.1:<port>`` (default 9877), the same bridge the Ableton MCP server uses."""

    def __init__(self, port: int = 9877, host: str = "127.0.0.1", timeout: float = 15.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout

    def send(self, command: str, **params: Any) -> Any:
        """Send one command and return its ``result``; raises ``BridgeError`` with a readable message."""
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.sendall(json.dumps({"type": command, "params": params}).encode())
                data = b""
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    try:
                        response = json.loads(data)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    response = json.loads(data)
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeError(f"Cannot talk to Ableton on {self.host}:{self.port} ({exc}). Is Live open with the AbletonMCP control surface?") from exc
        if response.get("status") == "error":
            raise BridgeError(f"Ableton rejected '{command}': {response.get('message')}")
        return response.get("result")

    def device_parameters(self, track_index: int, device_index: int) -> list[dict[str, Any]]:
        """All parameters of a device (name, value, min, max, index)."""
        result = self.send("get_device_parameters", track_index=track_index, device_index=device_index)
        return result["device"]["parameters"] if isinstance(result, dict) else json.loads(result)["device"]["parameters"]

    def set_parameter(self, track_index: int, device_index: int, parameter_index: int, value: float) -> Any:
        return self.send("set_device_parameter", track_index=track_index, device_index=device_index,
                         parameter_index=parameter_index, value=value)
