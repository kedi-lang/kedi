from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _log(message: dict[str, Any]) -> None:
    path = os.getenv("KEDI_ACP_FAKE_LOG")
    if path:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, separators=(",", ":")) + "\n")


_log({"method": "process/cwd", "cwd": os.getcwd()})
pending = {}
sessions = 0


for line in sys.stdin:
    request = json.loads(line)
    _log(request)
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": 1}})
    elif method == "session/new":
        sessions += 1
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": f"session-{sessions}"}})
    elif method == "session/prompt":
        session_id = params["sessionId"]
        prompt = params["prompt"][0]["text"]
        if prompt == "wait":
            pending[session_id] = request_id
            _send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "waiting"},
                        },
                    },
                }
            )
            continue
        _send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "echo:"},
                    },
                },
            }
        )
        _send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": prompt},
                    },
                },
            }
        )
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})
    elif method == "session/cancel":
        pending_id = pending.pop(params["sessionId"], None)
        if pending_id is not None:
            _send({"jsonrpc": "2.0", "id": pending_id, "result": {"stopReason": "cancelled"}})
    elif method == "session/close":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    else:
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }
        )
