"""A scripted JSON-RPC bridge for the JsonRpcAdapter tests: ``python jsonrpc_peer.py SCENARIO``.

Each scenario is one way a bridge behaves, well or badly. Runs are handled on their own
threads, so a bridge can wait on Kedi while Kedi sends it another run.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

SCENARIO = sys.argv[1]
CAPABILITIES = {"structuredOutput": True, "tools": True}
_write = threading.Lock()
_pending: dict[int, Any] = {}
_ids = iter(range(1, 1_000_000))


def send(message: dict[str, Any]) -> None:
    with _write:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        sys.stdout.flush()


def call_kedi(method: str, params: Any) -> dict[str, Any]:
    request_id = next(_ids)
    event = threading.Event()
    _pending[request_id] = event
    send({"id": request_id, "method": method, "params": params})
    event.wait(30)
    return _pending.pop(request_id)


def handshake() -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocolVersion": 1,
        "backend": {"name": "peer", "version": "1"},
        "kind": "agent-harness",
        "capabilities": CAPABILITIES,
    }
    if SCENARIO == "version":
        result["protocolVersion"] = 2
    elif SCENARIO == "kind":
        result["kind"] = "robot"
    elif SCENARIO == "framework":
        result["kind"] = "agent-framework"
    elif SCENARIO == "events":
        result["capabilities"] = {**CAPABILITIES, "events": True}
    elif SCENARIO == "sessions":
        result["capabilities"] = {**CAPABILITIES, "sessions": True}
    elif SCENARIO.startswith("media"):
        result["capabilities"] = {**CAPABILITIES, "mediaInput": True}
        result["mediaKinds"] = {"media": ["image"], "media-bad-kinds": ["smell"]}.get(SCENARIO, [])
    elif SCENARIO == "nameless":
        result["backend"] = {}
    elif SCENARIO == "capabilities-list":
        result["capabilities"] = ["tools"]
    elif SCENARIO == "text-only":
        result["capabilities"] = {"media": True, "events": True}
    elif SCENARIO in ADMISSION:
        result["capabilities"] = {**CAPABILITIES, **ADMISSION[SCENARIO]}
    return result


ADMISSION = {
    "governed": {"governed": True, "nativeTools": True},
    "governed-kedi-tools": {"governed": True},
    "native-ungoverned": {"nativeTools": True},
    "abandon-approval": {"nativeTools": True},
    "subagents": {"governed": True, "subagents": True, "sessions": True, "nativeTools": True},
    "subagents-stateless": {"governed": True, "subagents": True},
    "subagents-die": {"governed": True, "subagents": True},
    "subagents-ungoverned": {"subagents": True},
    # Claims governed, then retries a failed request without asking Kedi.
    "retries-unadmitted": {"governed": True},
}


def admission_probe(request_id: int, params: dict[str, Any]) -> None:
    """Ask for each admission once, and report Kedi's answers as the run's text."""

    run_id = params["runId"]
    replies = {"request": call_kedi("budget/admit", {"runId": run_id, "dimension": "request"})}
    if replies["request"].get("result", {}).get("admitted") is False:
        send({"id": request_id, "error": {"code": -32002, "message": "the budget is exhausted"}})
        return
    shell = {"runId": run_id, "tool": "shell", "arguments": {"command": "ls"}}
    replies |= {
        "tool attempt": call_kedi("budget/admit", {"runId": run_id, "dimension": "toolAttempt"}),
        "bad dimension": call_kedi("budget/admit", {"runId": run_id, "dimension": "tokens"}),
        "approval": call_kedi("approval/request", {**shell, "risk": "mutating"}),
        "bad risk": call_kedi("approval/request", {**shell, "risk": "harmless"}),
    }
    send({"id": request_id, "result": {"text": json.dumps(replies, sort_keys=True)}})


def subagent_probe(request_id: int, params: dict[str, Any]) -> None:
    """A parent delegates through Kedi's tool; a child reports what Kedi sent it."""

    run_id = params["runId"]
    if params["subagent"] is None and "delegate_task" in kedi_tools(params):
        background = "background" in params["user"] or "cancel" in params["user"]
        work = "Wait for it." if "cancel" in params["user"] else "Find q."
        task = {"subagent": "worker", "task": work, "background": background}
        reply = call_kedi(
            "tool/call", {"runId": run_id, "name": "delegate_task", "arguments": task}
        )
        parent: dict[str, Any] = {"runId": run_id, "reply": reply}
        if background:
            follow = "cancel_subagent" if "cancel" in params["user"] else "wait_subagent"
            arguments = {"run_id": reply["result"]["result"]["run_id"]}
            parent[follow] = call_kedi(
                "tool/call", {"runId": run_id, "name": follow, "arguments": arguments}
            )
        send({"id": request_id, "result": {"text": json.dumps(parent)}})
        return
    if "Wait for it." in params["user"]:
        time.sleep(5)  # Kedi cancels this child; its run/cancel reaches the bridge's stderr
    replies: dict[str, Any] = {"subagent": params["subagent"], "session": params["session"]}
    admissions = []
    for _ in range(2):
        admissions.append(call_kedi("budget/admit", {"runId": run_id, "dimension": "request"}))
        if SCENARIO == "subagents-die":
            sys.stderr.write("bridge crashed during a child\n")
            sys.stderr.flush()
            import os

            os._exit(3)
        if admissions[-1].get("result", {}).get("admitted") is False:
            send({"id": request_id, "error": {"code": -32002, "message": "the limit is spent"}})
            return
    lookup = {"runId": run_id, "name": "lookup", "arguments": {"query": "q"}}
    replies |= {"admissions": admissions, "lookups": [call_kedi("tool/call", lookup) for _ in "ab"]}
    replies["user"] = params["user"]
    if SCENARIO == "subagents":
        shell = {
            "runId": run_id,
            "tool": "shell",
            "arguments": {"command": "ls"},
            "risk": "mutating",
        }
        replies["approval"] = call_kedi("approval/request", shell)
    # The bridge's own request count is not Kedi's: Kedi counted what it admitted.
    usage = {"requests": 5, "inputTokens": 3, "outputTokens": 1}
    result: dict[str, Any] = {"text": json.dumps(replies, sort_keys=True), "usage": usage}
    if params["output"] is not None:
        summary = params["continuation"]["message"] if params["continuation"] else "first try"
        data: dict[str, Any] = {"task_summary": summary[:900]}
        if params["continuation"]:  # the first answer leaves out final_result
            data["final_result"] = {"ocean": "Pacific"}
        result = {"data": data, "usage": usage}
    if params["session"] is not None:
        session = params["session"]
        result["state"] = {"turn": session["turn"], "previous": session["state"]}
    send({"id": request_id, "result": result})


def unadmitted_retry(request_id: int, params: dict[str, Any]) -> None:
    """Admit the first model request only, and retry it once when the endpoint fails it."""

    import urllib.error
    import urllib.request

    call_kedi("budget/admit", {"runId": params["runId"], "dimension": "request"})
    body = {"model": "m", "messages": [{"role": "user", "content": params["user"]}]}
    post = urllib.request.Request(
        f"{sys.argv[2]}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(post, timeout=10) as response:
                message = json.load(response)["choices"][0]["message"]
            send({"id": request_id, "result": {"text": message["content"]}})
            return
        except urllib.error.HTTPError:
            if attempt:
                send({"id": request_id, "error": {"code": -32002, "message": "endpoint failed"}})


def kedi_tools(params: dict[str, Any]) -> list[str]:
    return [tool["name"] for tool in params["tools"] if "artifact" not in tool["name"]]


def run(request_id: int, params: dict[str, Any]) -> None:
    user = params["user"]
    if SCENARIO == "retries-unadmitted":
        unadmitted_retry(request_id, params)
        return
    if SCENARIO in ("subagents", "subagents-stateless", "subagents-die"):
        subagent_probe(request_id, params)
        return
    if SCENARIO == "abandon-approval":
        # Answer the run while Kedi is still deciding an approval: Kedi cancels the decision.
        def wait_for_approval() -> None:
            shell = {"tool": "shell", "arguments": {}, "risk": "mutating"}
            reply = call_kedi("approval/request", {"runId": params["runId"], **shell})
            sys.stderr.write(f"approval reply {reply.get('error', {}).get('code')}\n")
            sys.stderr.flush()

        threading.Thread(target=wait_for_approval, daemon=True).start()
        time.sleep(0.2)
        send({"id": request_id, "result": {"text": "answered early"}})
        return
    if SCENARIO in ADMISSION:
        admission_probe(request_id, params)
        return
    if SCENARIO == "garbage":
        sys.stdout.write("hello from a stray print\n")
        sys.stdout.flush()
        return
    if SCENARIO == "oversized":
        sys.stdout.write('{"jsonrpc": "2.0", "padding": "' + "x" * (8 * 1024 * 1024) + '"}\n')
        sys.stdout.flush()
        return
    if SCENARIO == "not-jsonrpc":
        sys.stdout.write(json.dumps({"id": request_id, "result": {"text": "no version"}}) + "\n")
        sys.stdout.flush()
        return
    if SCENARIO == "silent":
        return
    if SCENARIO == "die":
        sys.stderr.write("bridge crashed while waiting\n")
        sys.stderr.flush()
        import os

        os._exit(3)
    if SCENARIO == "error":
        send({"id": request_id, "error": {"code": -32002, "message": "model is down"}})
        return
    if SCENARIO == "abandon":
        # Answer the run while its tool call is still running: Kedi cancels the call.
        def wait_for_tool() -> None:
            reply = call_kedi(
                "tool/call", {"runId": params["runId"], "name": kedi_tools(params)[0]}
            )
            sys.stderr.write(f"tool reply {reply.get('error', {}).get('code')}\n")
            sys.stderr.flush()

        threading.Thread(target=wait_for_tool, daemon=True).start()
        time.sleep(0.2)
        send({"id": request_id, "result": {"text": "answered early"}})
        return
    if SCENARIO in ("events", "events-unadvertised"):
        run_id = params["runId"]
        for event in (
            {"type": "message.delta", "messageId": "m1", "text": "Looking"},
            {"type": "message.delta", "messageId": "m1", "text": " it up."},
            {"type": "message.completed", "messageId": "m1", "phase": "commentary"},
            {"type": "tool.boundary"},
            {"type": "message.delta", "messageId": "m2"},  # no text: ignored
            {"type": "telepathy", "messageId": "m3", "text": "?"},  # unknown: ignored
        ):
            send({"method": "run/event", "params": {"runId": run_id, **event}})
        send({"method": "run/event", "params": {"runId": "other", "type": "tool.boundary"}})
        send({"method": "run/event", "params": []})
        final = f"echo: {user}"
        send(
            {
                "method": "run/event",
                "params": {
                    "runId": run_id,
                    "type": "message.completed",
                    "messageId": "m4",
                    "text": final,
                },
            }
        )
        send({"id": request_id, "result": {"text": final}})
        return
    if SCENARIO == "sessions":
        session = params["session"]
        turns = ((session or {}).get("state") or {}).get("turns", [])
        text = f"{user} after {turns}" + ("" if session else " (no session)")
        reply: dict[str, Any] = {"text": text}
        if session is not None:
            reply["state"] = {"turns": [*turns, user], "id": session["id"], "turn": session["turn"]}
        send({"id": request_id, "result": reply})
        return
    if SCENARIO == "media":
        parts = [
            {key: value for key, value in part.items() if key != "data"} | {"bytes": "data" in part}
            for part in params["userContent"] or []
        ]
        send({"id": request_id, "result": {"text": json.dumps(parts, sort_keys=True)}})
        return
    if SCENARIO == "bad-error-code":
        send({"id": request_id, "error": {"code": "x", "message": "untyped failure"}})
        return
    if SCENARIO == "not-an-object":
        send({"id": request_id, "result": "plain"})
        return
    if SCENARIO == "missing-answer":
        send({"id": request_id, "result": {}})
        return
    if SCENARIO == "stderr-flood":
        sys.stderr.write(("diagnostic " * 100 + "\n") * 6000)
        sys.stderr.flush()
    if SCENARIO == "corrections":
        continuation = params["continuation"]
        if continuation is None:
            answer = {"invalid": "the answer is three", "error": "Expected a JSON value"}
        elif "invalid" in continuation["answer"]:
            answer = {"data": "three", "usage": {"inputTokens": 3, "outputTokens": 2}}
        else:
            answer = {"data": 3, "echo": continuation["message"]}
        send({"id": request_id, "result": answer})
        return
    if SCENARIO == "required":
        if params["continuation"] is None:
            send({"id": request_id, "result": {"text": "skipped the tool"}})
            return
        call_kedi("tool/call", {"runId": params["runId"], "name": "lookup", "arguments": {}})
        send({"id": request_id, "result": {"text": params["continuation"]["message"]}})
        return
    if SCENARIO == "callbacks":
        run_id = params["runId"]
        replies = {
            "unknown method": call_kedi("kedi/unknown", {}),
            "unknown run": call_kedi("tool/call", {"runId": "nope", "name": "x", "arguments": {}}),
            "bad params": call_kedi("tool/call", {"runId": run_id, "name": 1}),
            "list run id": call_kedi("tool/call", {"runId": [], "name": "x"}),
            "unknown tool": call_kedi("tool/call", {"runId": run_id, "name": "ghost"}),
            "no params": call_kedi("tool/call", None),
            "failing tool": call_kedi("tool/call", {"runId": run_id, "name": "explode"}),
        }
        with _write:  # blank lines between frames are skipped
            sys.stdout.write("\n")
            sys.stdout.flush()
        send({"method": "kedi/notice", "params": {}})  # notifications to Kedi are ignored
        send({"id": 999_999, "result": {"late": True}})  # a response nobody waits for
        send({"id": request_id, "result": {"text": json.dumps(replies, sort_keys=True)}})
        return
    if SCENARIO == "overload":
        results: list[dict[str, Any]] = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    call_kedi("tool/call", {"runId": params["runId"], "name": "slow"})
                )
            )
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        send({"id": request_id, "result": {"text": json.dumps(results, sort_keys=True)}})
        return
    # The default bridge: call the run's first Kedi tool, if any, then echo.
    tools = kedi_tools(params)
    if tools:
        reply = call_kedi(
            "tool/call", {"runId": params["runId"], "name": tools[0], "arguments": {}}
        )
        text = f"{user} -> {json.dumps(reply.get('result'), sort_keys=True)}"
    else:
        text = f"echo: {user}"
    if params["output"] is not None:
        send({"id": request_id, "result": {"data": {"answer": text}}})
        return
    send({"id": request_id, "result": {"text": text, "usage": {"requests": 2, "inputTokens": 5}}})


def main() -> None:
    if SCENARIO == "exit-on-init":
        sys.stderr.write("boom: missing API key\n")
        sys.stderr.flush()
        sys.exit(3)
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method is None:
            event = _pending.get(message["id"])
            if event is not None:
                _pending[message["id"]] = message
                event.set()
            continue
        if method == "initialize":
            if SCENARIO == "hang-on-init":
                continue
            send({"id": message["id"], "result": handshake()})
        elif method == "run":
            threading.Thread(
                target=run, args=(message["id"], message["params"]), daemon=True
            ).start()
        elif method == "run/cancel":
            sys.stderr.write(f"cancel {message['params']['runId']}\n")
            sys.stderr.flush()
        elif method == "shutdown":
            if SCENARIO == "stubborn":
                time.sleep(60)
            send({"id": message["id"], "result": {}})
            return


if __name__ == "__main__":
    main()
