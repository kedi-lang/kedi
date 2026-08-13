from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def send(message: dict[str, Any]) -> None:
    print(json.dumps(message, ensure_ascii=False, separators=(",", ":")), flush=True)


def record(message: dict[str, Any]) -> None:
    path = os.getenv("KEDI_CODEX_FAKE_LOG")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")


def complete_turn(turn_id: str, text: str) -> None:
    send(
        {
            "method": "item/completed",
            "params": {
                "turnId": turn_id,
                "item": {"type": "agentMessage", "text": text},
            },
        }
    )
    send(
        {
            "method": "turn/completed",
            "params": {"turn": {"id": turn_id, "status": "completed"}},
        }
    )


thread_count = 0
turn_count = 0
pending_tools: dict[int, str] = {}
pending_approvals: dict[int, str] = {}


def request_native_approval(
    *,
    request_id: int,
    turn_id: str,
    thread_id: str,
    kind: str,
) -> None:
    method: str
    params: dict[str, Any] = {
        "threadId": thread_id,
        "turnId": turn_id,
        "itemId": f"approval-{turn_id}",
        "startedAtMs": 0,
    }
    if kind == "command-read":
        method = "item/commandExecution/requestApproval"
        params.update(
            {
                "command": "cat README.md",
                "cwd": "/tmp/project",
                "commandActions": [
                    {
                        "type": "read",
                        "command": "cat README.md",
                        "name": "cat",
                        "path": "/tmp/project/README.md",
                    }
                ],
            }
        )
    elif kind == "command-write":
        method = "item/commandExecution/requestApproval"
        params.update(
            {
                "command": "rm notes.txt",
                "cwd": "/tmp/project",
                "commandActions": [{"type": "unknown", "command": "rm notes.txt"}],
            }
        )
    elif kind == "file":
        method = "item/fileChange/requestApproval"
        params["grantRoot"] = "/tmp/project"
    elif kind == "permissions":
        method = "item/permissions/requestApproval"
        params["cwd"] = "/tmp/project"
        params["reason"] = "network access"
        params["permissions"] = {"network": {"enabled": True}, "fileSystem": None}
    else:
        raise ValueError(f"unsupported fake approval kind: {kind}")
    pending_approvals[request_id] = turn_id
    send({"id": request_id, "method": method, "params": params})


for raw_line in sys.stdin:
    message = json.loads(raw_line)
    record(message)
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        send({"id": request_id, "result": {"serverInfo": {"name": "fake-codex"}}})
        continue
    if method == "initialized":
        continue
    if method == "thread/start":
        reconnect_file = os.getenv("KEDI_CODEX_FAKE_RECONNECT_FILE")
        if reconnect_file and not Path(reconnect_file).exists():
            Path(reconnect_file).touch()
            sys.exit(0)
        thread_count += 1
        send({"id": request_id, "result": {"thread": {"id": f"thread-{thread_count}"}}})
        continue
    if method == "thread/resume":
        send(
            {
                "id": request_id,
                "result": {"thread": {"id": message["params"]["threadId"]}},
            }
        )
        continue
    if method == "turn/start":
        turn_count += 1
        turn_id = f"turn-{turn_count}"
        send(
            {
                "id": request_id,
                "result": {"turn": {"id": turn_id, "status": "inProgress"}},
            }
        )
        prompt = message["params"]["input"][0]["text"]
        if prompt == "timeout":
            continue
        if prompt == "disconnect-turn":
            print("forced disconnect", file=sys.stderr, flush=True)
            sys.exit(0)
        if prompt == "malformed-json":
            print("not-json", flush=True)
            continue
        if prompt == "no-message":
            send(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": turn_id, "status": "completed"}},
                }
            )
            continue
        if prompt == "failed-turn":
            send(
                {
                    "method": "turn/completed",
                    "params": {
                        "turn": {
                            "id": turn_id,
                            "status": "failed",
                            "error": {"message": "fake turn failed"},
                        }
                    },
                }
            )
            continue
        if prompt == "missing-turn-data":
            send(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"status": "completed"}},
                }
            )
            continue
        if prompt == "invalid-output":
            complete_turn(turn_id, "not json")
            continue
        if prompt == "compacted":
            send(
                {
                    "method": "item/completed",
                    "params": {
                        "turnId": turn_id,
                        "item": {"type": "contextCompaction"},
                    },
                }
            )
            complete_turn(turn_id, json.dumps({"answer": prompt}, ensure_ascii=False))
            continue
        if prompt.startswith("tool:"):
            _, tool_name, arguments = prompt.split(":", 2)
            tool_request_id = 10_000 + turn_count
            pending_tools[tool_request_id] = turn_id
            send(
                {
                    "id": tool_request_id,
                    "method": "item/tool/call",
                    "params": {
                        "threadId": message["params"]["threadId"],
                        "turnId": turn_id,
                        "callId": f"call-{turn_count}",
                        "tool": tool_name,
                        "arguments": json.loads(arguments),
                    },
                }
            )
            continue
        if prompt.startswith("approval:"):
            _, kind = prompt.split(":", 1)
            request_native_approval(
                request_id=20_000 + turn_count,
                turn_id=turn_id,
                thread_id=message["params"]["threadId"],
                kind=kind,
            )
            continue
        complete_turn(turn_id, json.dumps({"answer": prompt}, ensure_ascii=False))
        continue
    if method == "turn/interrupt":
        send({"id": request_id, "result": {}})
        continue
    if method == "thread/delete":
        send({"id": request_id, "result": {}})
        continue
    if request_id in pending_tools and "result" in message:
        turn_id = pending_tools.pop(request_id)
        result = message["result"]
        content = result["contentItems"][0]["text"]
        complete_turn(
            turn_id,
            json.dumps({"answer": content, "success": result["success"]}, ensure_ascii=False),
        )
        continue
    if request_id in pending_approvals and "result" in message:
        turn_id = pending_approvals.pop(request_id)
        complete_turn(
            turn_id,
            json.dumps({"answer": json.dumps(message["result"]), "success": True}),
        )
        continue
    if request_id is not None:
        send(
            {
                "id": request_id,
                "error": {"code": -32601, "message": f"unsupported: {method}"},
            }
        )
