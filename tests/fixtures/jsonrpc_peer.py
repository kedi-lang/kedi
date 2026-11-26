"""A scripted JSON-RPC bridge for the JsonRpcAdapter tests: ``python jsonrpc_peer.py SCENARIO``.

Each scenario is one way a bridge behaves, well or badly. Runs are handled on their own
threads, so a bridge can wait on Kedi while Kedi sends it another run.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from typing import Any

SCENARIO = sys.argv[1]
CAPABILITIES = {"structuredOutput": True, "tools": True}
_write = threading.Lock()
_pending: dict[int, Any] = {}
_live_lock = threading.Lock()
_inputs: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
_interrupted: set[str] = set()
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
    elif SCENARIO == "transcript":
        result["capabilities"] = {**CAPABILITIES, "transcript": True, "mediaInput": True}
        result["mediaKinds"] = ["image"]
    elif SCENARIO == "test-model":
        result["capabilities"] = {**CAPABILITIES, "testModel": True}
    elif SCENARIO == "both-histories":
        result["capabilities"] = {**CAPABILITIES, "transcript": True, "sessions": True}
    elif SCENARIO == "harness-errors":
        result["capabilities"] = {**CAPABILITIES, "governed": True}
    elif SCENARIO == "hooked":
        result["capabilities"] = {**CAPABILITIES, "nativeTools": True}
    elif SCENARIO == "mcp-not-native":
        result["mcp"] = {"transports": ["stdio"]}
    elif SCENARIO == "mcp-unknown-transport":
        result["capabilities"] = {**CAPABILITIES, "nativeTools": True}
        result["mcp"] = {"transports": ["carrier-pigeon"]}
    elif SCENARIO == "drawing":
        result["capabilities"] = {
            **CAPABILITIES,
            "mediaInput": True,
            "imageOutput": True,
            "audioOutput": True,
        }
        result["mediaKinds"] = ["image"]
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
    elif SCENARIO == "configured":
        result["capabilities"] = {**CAPABILITIES, "modelFallback": True}
        result["settings"] = ["temperature"]
        result["effort"] = ["low", "high"]
    elif SCENARIO == "settings-own":
        result["settings"] = ["context_limit"]
    elif SCENARIO == "settings-not-names":
        result["settings"] = [1]
    elif SCENARIO == "effort-unknown":
        result["effort"] = ["turbo"]
    return result


ADMISSION = {
    "governed": {"governed": True, "nativeTools": True},
    "governed-kedi-tools": {"governed": True},
    "native-ungoverned": {"nativeTools": True},
    "abandon-approval": {"nativeTools": True},
    "subagents": {"governed": True, "subagents": True, "sessions": True, "nativeTools": True},
    "subagents-stateless": {"governed": True, "subagents": True},
    "subagents-die": {"governed": True, "subagents": True},
    "live": {
        "governed": True,
        "subagents": True,
        "sessions": True,
        "liveInput": True,
        "interrupt": True,
        "usage": True,
    },
    "live-without-subagents": {"liveInput": True},
    "interrupt-without-live": {"governed": True, "subagents": True, "interrupt": True},
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


def until(condition: Any, seconds: float = 10) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def live_probe(request_id: int, params: dict[str, Any]) -> None:
    """A parent sends input to, or interrupts, a background child that waits for it."""

    run_id, user = params["runId"], params["user"]
    if params["subagent"] is None:
        call = lambda name, **arguments: call_kedi(  # noqa: E731
            "tool/call", {"runId": run_id, "name": name, "arguments": arguments}
        )
        work = "Count tokens." if "Count" in user else "Wait for input."
        if "Interrupt" in user:
            work = "Wait for interrupt."
        started = call("delegate_task", subagent="worker", task=work, background=True)
        task_id = started["result"]["result"]["run_id"]
        replies: dict[str, Any] = {}
        if work != "Count tokens.":
            replies["sent"] = call(
                "send_subagent_input",
                task_id=task_id,
                message="Use the blue one.",
                interrupt="Interrupt" in user,
            )
        replies["waited"] = call("wait_subagent", run_id=task_id)
        send({"id": request_id, "result": {"text": json.dumps(replies)}})
        return
    call_kedi("budget/admit", {"runId": run_id, "dimension": "request"})
    seen: dict[str, Any] = {"user": user, "session": params["session"]}
    if user == "Wait for input.":
        until(lambda: run_id in _inputs)
        with _live_lock:
            batches = _inputs.pop(run_id, [])
        for input_id, _items in batches:
            send({"id": input_id, "result": {"delivered": True}})
        seen["inputs"] = [item["text"] for _, items in batches for item in items]
    elif user == "Wait for interrupt.":
        until(lambda: run_id in _interrupted)
        error = {"code": -32006, "message": "interrupted", "data": {"state": {"partial": run_id}}}
        send({"id": request_id, "error": error})
        return
    elif user == "Count tokens.":
        call_kedi("budget/admit", {"runId": run_id, "dimension": "request"})
        for _ in range(2):
            send(
                {
                    "method": "run/usage",
                    "params": {"runId": run_id, "inputTokens": 3, "outputTokens": 1},
                }
            )
        # Usage for a run that is not in progress is dropped.
        send({"method": "run/usage", "params": {"runId": "gone", "inputTokens": 50}})
    result: dict[str, Any] = {"text": json.dumps(seen), "usage": {"requests": 9, "inputTokens": 99}}
    if params["session"] is not None:
        result["state"] = {"after": run_id}
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


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
WAV = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode()
_drawn: dict[str, Any] = {}


def drawing(request_id: int, params: dict[str, Any]) -> None:
    """The model draws and speaks: answers with media for a media output."""

    output, user = params["output"] or {}, params["user"]
    if user == "What did you get?":
        send({"id": request_id, "result": {"text": json.dumps(_drawn)}})
        return
    _drawn.update(output=output, tools=[tool["name"] for tool in params["tools"]])
    sound = {"type": "media", "kind": "audio", "mediaType": "audio/wav", "data": WAV}
    drawn = {"type": "media", "kind": "image", "mediaType": "image/png", "data": PNG}
    answers = {
        "Pick one.": {"data": "photo"},
        "Answer with sound.": {"content": [sound]},
        "Answer with garbage.": {"content": [{**drawn, "data": "not base64!"}]},
        "Draw two.": {"content": [drawn, drawn]},
        "Answer with parts.": {"content": "not a list"},
        "Link it.": {
            "content": [{"type": "media", "kind": "image", "url": "https://example.com/cat.png"}]
        },
    }
    # The model's text may come with what it made.
    made = [
        {"type": "text", "text": "Here it is."},
        sound if output.get("media") == "audio" else drawn,
    ]
    if user.startswith("Pick one."):
        user = "Pick one."
    send({"id": request_id, "result": answers.get(user, {"content": made})})


HARNESS = {
    "type": "ProviderError",
    "message": "429 rate limited",
    "cause": {"type": "HTTPError", "message": "Too Many Requests"},
}


def harness_errors(request_id: int, params: dict[str, Any]) -> None:
    """The harness fails, or Kedi stops it: the run's error carries the harness's own."""

    sys.stderr.write("[harness] retrying the provider\n[harness] giving up\n")
    sys.stderr.flush()
    if params["user"] == "Spend.":
        for _ in range(2):
            reply = call_kedi("budget/admit", {"runId": params["runId"], "dimension": "request"})
            if not reply["result"]["admitted"]:
                break
    error = {"code": -32002, "message": "the harness failed", "data": {"harness": HARNESS}}
    send({"id": request_id, "error": error})


def hooked(request_id: int, params: dict[str, Any]) -> None:
    """The bridge's own shell tool, with the program's hooks around each call."""

    run_id, user = params["runId"], params["user"]
    call = {"runId": run_id, "callId": "c1", "tool": "shell", "origin": "adapter_native"}
    if user == "Admit it.":
        admitted = call_kedi("artifact/admit", {**call, "result": "a b"})
        send({"id": request_id, "result": {"text": json.dumps(admitted)}})
        return
    if user == "Admit early.":
        threading.Thread(
            target=lambda: sys.stderr.write(
                f"admit reply {call_kedi('artifact/admit', {**call, 'result': 'x'}).get('error', {}).get('code')}\n"
            ),
            daemon=True,
        ).start()
        time.sleep(0.2)
        send({"id": request_id, "result": {"text": "answered early"}})
        return
    if user == "Answer early.":
        # Answer the run while Kedi still runs a hook: Kedi cancels it.
        def wait_for_hook() -> None:
            hook = {**call, "event": "pre_tool_use", "arguments": {"command": "ls"}}
            reply = call_kedi("hook/invoke", hook)
            sys.stderr.write(f"hook reply {reply.get('error', {}).get('code')}\n")
            sys.stderr.flush()

        threading.Thread(target=wait_for_hook, daemon=True).start()
        time.sleep(0.2)
        send({"id": request_id, "result": {"text": "answered early"}})
        return
    command = {"Run ls.": "ls", "Run rm.": "rm -rf /", "Run a failing command.": "false"}[user]
    # Kedi lists the events with handlers; the bridge reports only those.
    events = params["hooks"]
    seen: dict[str, Any] = {"hooks": events}
    arguments = {"command": command}
    if "pre_tool_use" in events:
        pre = call_kedi("hook/invoke", {**call, "event": "pre_tool_use", "arguments": arguments})
        seen["pre"] = pre
        if "error" in pre or pre["result"]["decision"] == "deny":
            send({"id": request_id, "result": {"text": json.dumps(seen)}})
            return
        arguments = pre["result"].get("arguments", arguments)
    approval = {"runId": run_id, "callId": "c1", "tool": "shell", "risk": "mutating"}
    seen["approval"] = call_kedi("approval/request", {**approval, "arguments": arguments})
    if arguments["command"] == "false":
        error = {"type": "CalledProcessError", "message": "exit status 1"}
        post = {**call, "event": "post_tool_use_failure", "arguments": arguments, "error": error}
    else:
        post = {**call, "event": "post_tool_use", "arguments": arguments, "result": "a b"}
    if post["event"] in events:
        seen["post"] = call_kedi("hook/invoke", post)
    send({"id": request_id, "result": {"text": json.dumps(seen)}})


def transcript_probe(request_id: int, params: dict[str, Any]) -> None:
    """Kedi holds the conversation: answer with this turn's messages in its schema."""

    user, history = params["user"], params.get("history", [])
    said = [part["text"] for message in history for part in message["parts"] if "text" in part]
    reply: dict[str, Any] = {"text": f"{user} after {said}"}
    turn = [
        {"role": "user", "parts": [{"type": "text", "text": user}]},
        {"role": "assistant", "parts": [{"type": "text", "text": reply["text"]}]},
    ]
    if params.get("processHistory") and user == "Process early.":
        # Answer the run while Kedi still runs the processor: Kedi cancels it.
        def wait_for_processor() -> None:
            current = {"runId": params["runId"], "messages": [*history, turn[0]]}
            reply = call_kedi("history/process", current)
            sys.stderr.write(f"process reply {reply.get('error', {}).get('code')}\n")
            sys.stderr.flush()

        threading.Thread(target=wait_for_processor, daemon=True).start()
        time.sleep(0.2)
        send({"id": request_id, "result": {"text": "answered early", "transcript": turn}})
        return
    if params.get("processHistory"):
        # Kedi's processor sees the conversation so far; the run answers with all of it.
        current = [*history, turn[0]]
        processed = call_kedi("history/process", {"runId": params["runId"], "messages": current})
        if user == "Forget it." or "error" in processed:
            send({"id": request_id, "result": {"text": json.dumps(processed)}})
            return
        kept = processed["result"]["messages"]
        reply["text"] = json.dumps([part["text"] for m in kept for part in m["parts"]])
        reply["transcript"] = [*kept, turn[1]]
    elif user == "Process anyway.":
        refused = call_kedi("history/process", {"runId": params["runId"], "messages": []})
        reply["text"] = json.dumps(refused)
        reply["messages"] = turn
    elif user == "Forget it.":
        pass
    elif user == "Break a part.":
        reply["messages"] = [{"role": "user", "parts": [{"type": "hologram"}]}]
    elif user == "Name a stranger.":
        reply["messages"] = [{"role": "user", "parts": [{"type": "media", "ref": "stranger"}]}]
    elif user == "No role.":
        reply["messages"] = [{"parts": []}]
    elif user == "Not a list.":
        reply["messages"] = "everything"
    elif user.startswith("Look at"):
        # The turn names its photo by ref; a later turn fetches the bytes with media/get.
        media = [part for part in params["userContent"] if part["type"] == "media"]
        turn[0]["parts"] += [{"type": "media", "ref": part["name"]} for part in media]
        reply["messages"] = turn
    elif user == "Fetch the photo.":
        photo = call_kedi("media/get", {"runId": params["runId"], "ref": "photo"})
        missing = call_kedi("media/get", {"runId": params["runId"], "ref": "nothing"})
        reply["text"] = json.dumps({"photo": photo, "missing": missing})
        reply["messages"] = turn
    else:
        reply["messages"] = turn
    send({"id": request_id, "result": reply})


def kedi_tools(params: dict[str, Any]) -> list[str]:
    return [tool["name"] for tool in params["tools"] if "artifact" not in tool["name"]]


def run(request_id: int, params: dict[str, Any]) -> None:
    user = params["user"]
    if SCENARIO == "retries-unadmitted":
        unadmitted_retry(request_id, params)
        return
    if SCENARIO == "live":
        live_probe(request_id, params)
        return
    if SCENARIO == "drawing":
        drawing(request_id, params)
        return
    if SCENARIO == "hooked":
        hooked(request_id, params)
        return
    if SCENARIO == "harness-errors":
        harness_errors(request_id, params)
        return
    if SCENARIO == "configured":
        keys = ("model", "fallbackModels", "settings", "effort")
        send({"id": request_id, "result": {"text": json.dumps({key: params[key] for key in keys})}})
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
    if SCENARIO == "transcript":
        transcript_probe(request_id, params)
        return
    if SCENARIO == "test-model":
        call = {"runId": params["runId"], "model": "m", "system": None, "tools": [], "output": None}
        call["messages"] = [{"role": "user", "parts": [{"type": "text", "text": params["user"]}]}]
        if params["user"] == "Answer early.":
            # Answer the run while Kedi's test model still runs: Kedi cancels it.
            def wait_for_model() -> None:
                reply = call_kedi("model/complete", call)
                sys.stderr.write(f"model reply {reply.get('error', {}).get('code')}\n")
                sys.stderr.flush()

            threading.Thread(target=wait_for_model, daemon=True).start()
            time.sleep(0.2)
            send({"id": request_id, "result": {"text": "answered early"}})
            return
        reply = call_kedi("model/complete", call)
        send({"id": request_id, "result": {"text": json.dumps(reply)}})
        return
    if SCENARIO == "both-histories":
        # Which mode Kedi chose: a harness keeps its session unless the program needs Kedi's.
        mode = "transcript" if "history" in params else "state"
        reply = {"text": mode, **({"messages": []} if mode == "transcript" else {"state": {}})}
        send({"id": request_id, "result": reply})
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
        elif method == "run/input" and SCENARIO != "live":
            send({"id": message["id"], "error": {"code": -32601, "message": "no live input"}})
        elif method == "run/input":
            with _live_lock:
                runs = _inputs.setdefault(message["params"]["runId"], [])
                runs.append((message["id"], message["params"]["items"]))
        elif method == "run/interrupt":
            with _live_lock:
                _interrupted.add(message["params"]["runId"])
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
