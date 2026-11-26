"""JsonRpcAdapter (#197): Kedi's adapter contract over JSON-RPC to a bridge process."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter import JsonRpcAdapter, ToolSpec
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command
from kedi.agent_adapter.adapters.jsonrpc.connection import Connection
from kedi.agent_adapter.adapters.jsonrpc.protocol import (
    CANCELLED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    OVERLOADED,
    RUN_FAILED,
    UNKNOWN_RUN,
    UNSUPPORTED,
    JsonRpcDisconnected,
    JsonRpcError,
)
from kedi.agent_adapter.approvals.policy import (
    ApprovalCancelled,
    ApprovalDecision,
    ApprovalPolicy,
)
from kedi.agent_adapter.conformance import check_adapter
from kedi.agent_adapter.conformance.check import SCENARIOS, _check_run, _run
from kedi.agent_adapter.conformance.endpoint import RecordingEndpoint, Script
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.agent_adapter.subagents import SubagentExecutionRequest, SubagentUsageLimits
from kedi.agent_adapter.subagents.control import (
    AgentInput,
    LiveInput,
    RunInterrupted,
    live_input_scope,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compile import build_capture_env
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope

PEER = Path(__file__).parent / "fixtures" / "jsonrpc_peer.py"


def peer(scenario: str, **options: Any) -> JsonRpcAdapter:
    return JsonRpcAdapter([sys.executable, str(PEER), scenario], model="m", **options)


@pytest.fixture
def adapters() -> Iterator[list[JsonRpcAdapter]]:
    opened: list[JsonRpcAdapter] = []
    yield opened
    for adapter in opened:
        adapter.close()


def run(source: str, adapter: JsonRpcAdapter, **globals_: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter, runtime_globals=globals_)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def failure(source: str, adapter: JsonRpcAdapter) -> BaseException:
    with pytest.raises(Exception) as caught:
        run(source, adapter)
    return getattr(caught.value, "original", None) or caught.value


ECHO = "[reply] << Say hello.\n= <reply>\n"


@pytest.mark.parametrize("mode", ["native", "protocol"])
def test_the_reference_bridge_in_javascript_passes_the_conformance_kit(mode: str) -> None:
    report = check_adapter(
        lambda base_url: JsonRpcAdapter(reference_bridge_command(f"{base_url}/v1", mode=mode))
    )
    assert report.passed, report.render()
    assert report.harness_text == []
    assert {result.name: result.status for result in report.results}["media"] == "passed"


def test_for_conformance_runs_the_reference_bridge() -> None:
    adapter = JsonRpcAdapter.for_conformance("http://127.0.0.1:9")
    try:
        assert adapter.backend == {"name": "openai-chat-bridge", "version": "1"}
        assert adapter.default_model() == "conformance"
    finally:
        adapter.close()


def test_the_handshake_sets_kind_identity_and_instance_capabilities(adapters: list) -> None:
    harness = peer("default")
    plain = peer("text-only")
    adapters += [harness, plain]
    assert harness.kind == "agent-harness"
    assert harness.backend == {"name": "peer", "version": "1"}
    assert harness.capabilities.supports_structured_output
    assert harness.capabilities.supports_tool_registration
    # Unknown and version-2 capabilities stay unavailable whatever the bridge says.
    assert not plain.capabilities.supports_structured_output
    assert not plain.capabilities.supports_tool_registration
    assert not plain.capabilities.supports_media_input


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("version", "the bridge speaks protocol version 2; Kedi speaks 1"),
        ("kind", "unknown bridge kind 'robot'"),
        ("nameless", "initialize must name the backend"),
        ("capabilities-list", "capabilities must be an object"),
        ("exit-on-init", "boom: missing API key"),
        # Kedi admits each of a child's requests against its limits: only a governed bridge.
        ("subagents-ungoverned", "subagents need governed"),
        ("live-without-subagents", "liveInput needs subagents"),
        ("interrupt-without-live", "interrupt needs liveInput"),
        ("settings-own", "settings lists Kedi's own settings: context_limit"),
        ("settings-not-names", "settings must list setting names"),
        ("mcp-not-native", "mcp needs nativeTools: MCP tools are the bridge's own"),
        ("mcp-unknown-transport", "mcp.transports must list transports of"),
        ("effort-unknown", "effort must list levels of"),
    ],
)
def test_a_bridge_that_fails_the_handshake_is_never_exposed(scenario: str, message: str) -> None:
    with pytest.raises(JsonRpcError, match=message):
        peer(scenario)


def test_a_bridge_that_never_initializes_times_out() -> None:
    with pytest.raises(TimeoutError, match="'initialize' timed out"):
        peer("hang-on-init", initialize_timeout=0.3)


def test_a_missing_executable_is_named() -> None:
    with pytest.raises(FileNotFoundError, match="bridge executable not found: no-such-bridge"):
        JsonRpcAdapter(["no-such-bridge"])
    with pytest.raises(ValueError, match="cannot be empty"):
        Connection([], handler=lambda *_: None)


def test_text_runs_report_usage_and_unsupported_operations_fail_before_dispatch(
    adapters: list,
) -> None:
    adapter = peer("text-only")
    adapters.append(adapter)
    usage: list[InvocationUsage] = []
    with observe_invocations(usage.append):
        assert run(ECHO, adapter) == "echo: Say hello."
    # The peer reports requests and input tokens only; what it leaves out is unknown.
    assert usage == [
        InvocationUsage(
            requests=2,
            input_tokens=5,
            cache_read_tokens=None,
            cache_write_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    ]
    error = failure("[n: int] << How many?\n= `n`\n", adapter)
    assert "structured output" in str(error)


def test_structured_output_is_validated_and_corrected_by_kedi(adapters: list) -> None:
    adapter = peer("corrections", output_retries=2)
    adapters.append(adapter)
    assert run("[n: int] << How many sides has a triangle?\n= `n`\n", adapter) == 3
    exhausted = peer("corrections", output_retries=1)
    adapters.append(exhausted)
    error = failure("[n: int] << How many sides has a triangle?\n= `n`\n", exhausted)
    assert isinstance(error, JsonRpcError) and error.code == INVALID_PARAMS
    assert "gave no valid answer after 1 corrections" in str(error)
    assert "The structured output does not match the required schema" in str(error)


def test_required_tools_and_direct_calls_use_kedis_text(adapters: list) -> None:
    adapter = peer("required")
    adapters.append(adapter)
    lookup = ToolSpec(fn=lambda: "found", name="lookup", metadata={"required_before_output": True})
    # A direct call outside a Kedi program gets the request Kedi would make from its arguments.
    with adapter.tool_scope((lookup,)):
        reply = adapter.invoke_sync(prompt="Use lookup.")
    assert reply == (
        "Call required tool lookup and use its result before returning the parent response."
    )
    with pytest.raises(ValueError, match="needs output_schema or output_type"):
        adapter.produce_sync(template="Nothing to fill.")


def test_the_bridge_calls_kedi_tools_of_its_own_run(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    source = "@city() -> str:\n  = Ankara\n\n> approval: allow\n> use: city\n[reply] << Where?\n= <reply>\n"
    assert run(source, adapter) == 'Where? -> {"ok": true, "result": "Ankara"}'


def test_bridge_requests_kedi_cannot_serve_get_typed_errors(adapters: list) -> None:
    adapter = peer("callbacks")
    adapters.append(adapter)
    source = (
        "@explode() -> str:\n  = `int('x')`\n\n"
        "> approval: allow\n> use: explode\n[reply] << Probe.\n= <reply>\n"
    )
    replies = json.loads(run(source, adapter))
    assert replies["unknown method"]["error"]["code"] == METHOD_NOT_FOUND
    assert replies["unknown run"]["error"]["code"] == UNKNOWN_RUN
    assert replies["bad params"]["error"]["code"] == INVALID_PARAMS
    assert replies["list run id"]["error"]["code"] == INVALID_PARAMS
    assert replies["no params"]["error"]["code"] == INVALID_REQUEST
    assert replies["unknown tool"]["result"] == {"ok": False, "error": "Unknown Kedi tool: ghost"}
    failed = replies["failing tool"]["result"]
    assert failed["ok"] is False
    assert failed["error"].startswith("Tool explode failed: ValueError: invalid literal")
    assert "Kedi stack trace" not in failed["error"]


def test_callbacks_beyond_the_admission_limit_are_refused(adapters: list) -> None:
    adapter = peer("overload", max_inbound_requests=1)
    adapters.append(adapter)
    source = "@slow() -> str:\n  = `__import__('time').sleep(0.5) or 'done'`\n\n> approval: allow\n> use: slow\n"
    results = json.loads(run(f"{source}[reply] << Twice.\n= <reply>\n", adapter))
    outcomes = sorted(json.dumps(result.get("error", {}).get("code", "ok")) for result in results)
    assert outcomes == sorted([json.dumps(OVERLOADED), '"ok"'])


def test_nested_runs_complete_while_the_bridge_waits_on_kedi(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    source = (
        "@inner() -> str:\n  [answer] << Inner question.\n  = <answer>\n\n"
        "> approval: allow\n> use: inner\n[reply] << Outer question.\n= <reply>\n"
    )
    started = time.monotonic()
    reply = run(source, adapter)
    assert reply == 'Outer question. -> {"ok": true, "result": "echo: Inner question."}'
    assert time.monotonic() - started < 20


def test_concurrent_runs_keep_their_own_tools(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    results: dict[str, Any] = {}

    def program(name: str) -> None:
        source = f"@tool_{name}() -> str:\n  = {name}\n\n> approval: allow\n> use: tool_{name}\n[r] << Ask {name}.\n= <r>\n"
        results[name] = run(source, adapter)

    threads = [threading.Thread(target=program, args=(name,)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert results == {
        name: f'Ask {name}. -> {{"ok": true, "result": "{name}"}}' for name in ("a", "b", "c")
    }


def test_an_async_host_loop_runs_the_callbacks(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    source = "@city() -> str:\n  = Izmir\n\n> approval: allow\n> use: city\n[reply] << Where?\n= <reply>\n"

    async def host() -> Any:
        with use_async_owner_loop():
            return await asyncio.to_thread(run, source, adapter)

    assert asyncio.run(host()) == 'Where? -> {"ok": true, "result": "Izmir"}'
    assert (
        asyncio.run(adapter.launch([sys.executable, str(PEER), "default"])).kind == "agent-harness"
    )


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("garbage", "not JSON to stdout; stdout carries only protocol frames"),
        ("oversized", "frame over the 8388608-byte limit"),
        ("not-jsonrpc", "not JSON-RPC 2.0"),
        ("die", "bridge crashed while waiting"),
    ],
)
def test_a_broken_stream_ends_the_connection_and_fails_waiters(scenario: str, message: str) -> None:
    adapter = peer(scenario)
    try:
        error = failure(ECHO, adapter)
        assert isinstance(error, JsonRpcDisconnected)
        assert message in str(error)
        # A disconnected adapter refuses calls before dispatching them.
        again = failure(ECHO, adapter)
        assert again is error or str(again) == str(error)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("scenario", "code", "message"),
    [
        ("error", RUN_FAILED, "model is down"),
        ("bad-error-code", INTERNAL_ERROR, "untyped failure"),
        ("not-an-object", INVALID_REQUEST, "without an object result"),
        ("missing-answer", INVALID_PARAMS, "a text run must answer with a string 'text'"),
    ],
)
def test_bad_run_answers_raise_typed_errors(scenario: str, code: int, message: str) -> None:
    adapter = peer(scenario)
    try:
        error = failure(ECHO, adapter)
    finally:
        adapter.close()
    assert isinstance(error, JsonRpcError) and error.code == code
    assert message in str(error)


def test_a_timed_out_run_is_cancelled_at_the_bridge() -> None:
    adapter = peer("silent", request_timeout=0.3)
    try:
        error = failure(ECHO, adapter)
        assert isinstance(error, TimeoutError)
        deadline = time.monotonic() + 5
        while not any(line.startswith("cancel ") for line in adapter._connection.stderr_tail):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        adapter.close()


def test_stderr_floods_do_not_stall_the_bridge(adapters: list) -> None:
    adapter = peer("stderr-flood")
    adapters.append(adapter)
    assert run(ECHO, adapter) == "echo: Say hello."
    assert len(adapter._connection.stderr_tail) == 50


def test_close_stops_a_bridge_that_ignores_shutdown() -> None:
    adapter = peer("stubborn")
    process = adapter._connection._process
    started = time.monotonic()
    adapter.close(timeout=0.3)
    assert process.poll() is not None
    assert time.monotonic() - started < 5
    with pytest.raises(JsonRpcDisconnected, match="connection is closed"):
        adapter._connection.start_request("run", {})


def test_profiles_pass_the_model_and_refuse_what_the_protocol_lacks(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    assert run('> model: "m2"\n' + ECHO, adapter) == "echo: Say hello."
    error = failure("> effort: high\n" + ECHO, adapter)
    assert "does not support profile fields: effort" in str(error)


def test_unsupported_code_constants_are_distinct() -> None:
    assert len({UNSUPPORTED, RUN_FAILED, CANCELLED, OVERLOADED, UNKNOWN_RUN}) == 5


def test_async_tools_and_the_context_manager() -> None:
    async def forecast() -> str:
        await asyncio.sleep(0)
        return "sunny"

    with peer("default") as adapter, adapter.tool_scope((ToolSpec(fn=forecast, name="forecast"),)):
        assert (
            adapter.invoke_sync(prompt="Weather?") == 'Weather? -> {"ok": true, "result": "sunny"}'
        )
    assert adapter._connection.disconnected is not None


def test_a_structured_run_without_data_is_a_protocol_error() -> None:
    with peer("missing-answer") as adapter:
        error = failure("[n: int] << How many?\n= `n`\n", adapter)
    assert isinstance(error, JsonRpcError) and error.code == INVALID_PARAMS
    assert "a structured run must answer with 'data'" in str(error)


def test_an_oversized_request_is_refused_and_the_connection_stays_open() -> None:
    with peer("default") as adapter:
        with pytest.raises(JsonRpcError, match="exceeds the 8388608-byte frame limit") as caught:
            adapter.invoke_sync(prompt="x" * (8 * 1024 * 1024))
        assert caught.value.code == INVALID_REQUEST
        assert adapter.invoke_sync(prompt="small") == "echo: small"


def test_answers_and_notifications_after_the_bridge_is_gone_are_dropped() -> None:
    adapter = peer("default")
    connection = adapter._connection
    # A request Kedi admitted is answered once; a second answer is ignored.
    assert connection._slots.acquire(blocking=False)
    connection._inbound.add("late")
    connection.respond("late", result={})
    connection.respond("late", result={})
    # A closed stdin disconnects; notifications and answers are then dropped quietly.
    assert connection._process.stdin is not None
    connection._process.stdin.close()
    connection.notify("run/cancel", {"runId": "r"})
    assert "closed its stdin" in str(connection.disconnected)
    assert connection._slots.acquire(blocking=False)
    connection._inbound.add("after")
    connection.respond("after", error=JsonRpcError(RUN_FAILED, "gone"))
    adapter.close(timeout=1)


def test_a_tool_call_for_a_run_whose_loop_closed_is_refused() -> None:
    from kedi.agent_adapter.adapters.jsonrpc.adapter import _Run

    with peer("default") as adapter:
        loop = asyncio.new_event_loop()
        loop.close()
        adapter._runs["ended"] = _Run(
            run_id="ended", loop=loop, context=contextvars.copy_context(), tools={}
        )
        sent: list[Any] = []
        adapter._connection.respond = lambda request_id, **reply: sent.append((request_id, reply))  # type: ignore[method-assign]
        adapter._on_request(5, "tool/call", {"runId": "ended", "name": "x", "arguments": {}})
    ((request_id, reply),) = sent
    assert request_id == 5 and reply["error"].code == UNKNOWN_RUN


def test_denied_tools_and_fatal_tool_failures(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    denied = run(
        "@city() -> str:\n  = Ankara\n\n> approval: deny\n> use: city\n[r] << Where?\n= <r>\n",
        adapter,
    )
    reply = json.loads(denied.split(" -> ", 1)[1])
    assert reply["ok"] is False and reply["error"].startswith("Tool 'city' was denied")
    assert "Kedi stack trace" not in reply["error"]
    # A failure that is not the model's to correct stops the run with Kedi's own error.
    fatal = failure(
        "@broken() -> str:\n  = `1 / 0`\n\n> approval: allow\n> use: broken\n[r] << Go.\n= <r>\n",
        adapter,
    )
    assert isinstance(fatal, ZeroDivisionError)
    assert run(ECHO, adapter) == "echo: Say hello."


def test_requires_checks_the_negotiated_peer(adapters: list) -> None:
    source = "> requires: structured_output\n" + ECHO
    rich, plain = peer("default"), peer("text-only")
    adapters += [rich, plain]
    assert run(source, rich) == "echo: Say hello."
    assert "structured_output" in str(failure(source, plain))


def test_a_handler_failure_ends_the_connection_instead_of_the_reader() -> None:
    script = (
        "import json, sys, time\n"
        'print(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "x", "params": {}}), flush=True)\n'
        "time.sleep(30)\n"
    )

    def handler(*_: Any) -> None:
        raise RuntimeError("handler bug")

    connection = Connection([sys.executable, "-c", script], handler=handler)
    try:
        deadline = time.monotonic() + 10
        while connection.disconnected is None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert "could not handle a bridge message: RuntimeError('handler bug')" in str(
            connection.disconnected
        )
    finally:
        connection.close(timeout=0.5)


def test_an_answer_over_the_frame_limit_becomes_an_error_reply() -> None:
    with peer("default") as adapter:
        connection = adapter._connection
        sent: list[Any] = []
        connection._send_error = lambda request_id, code, message: sent.append((request_id, code))  # type: ignore[method-assign]
        assert connection._slots.acquire(blocking=False)
        connection._inbound.add("big")
        connection.respond("big", result={"blob": "x" * (8 * 1024 * 1024)})
        assert sent == [("big", INVALID_REQUEST)]


def test_tool_calls_still_running_when_the_run_ends_are_cancelled() -> None:
    adapter = peer("abandon")
    try:
        source = (
            "@slow() -> str:\n  = `__import__('time').sleep(2) or 'late'`\n\n"
            "> approval: allow\n> use: slow\n[r] << Go.\n= <r>\n"
        )
        assert run(source, adapter) == "answered early"
        deadline = time.monotonic() + 10
        while f"tool reply {CANCELLED}" not in adapter._connection.stderr_tail:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("scenario", "messages"),
    [
        ("events", [("commentary", "Looking it up."), ("final", "echo: Say hello.")]),
        # A bridge that did not negotiate events is not listened to.
        ("events-unadvertised", [("final", "echo: Say hello.")]),
    ],
)
def test_bridge_events_reach_kedis_observers_attributed_to_the_run(
    scenario: str, messages: list[tuple[str, str]]
) -> None:
    from kedi.agent_adapter.events import (
        AgentMessageEvent,
        AgentRunStateEvent,
        observe_agent_events,
    )

    events: list[Any] = []
    with peer(scenario) as adapter, observe_agent_events(events.append):
        assert run(ECHO, adapter) == "echo: Say hello."
    assert [(e.phase, e.content) for e in events if isinstance(e, AgentMessageEvent)] == messages
    assert [e.state for e in events if isinstance(e, AgentRunStateEvent)] == [
        "started",
        "completed",
    ]
    assert len({event.run_id for event in events}) == 1
    assert {event.adapter for event in events} == {"rpc"}


def test_the_reference_bridge_reports_commentary_events() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from kedi.agent_adapter.events import AgentMessageEvent, observe_agent_events

    class Model(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if any(message["role"] == "tool" for message in body["messages"]):
                message: dict[str, Any] = {"role": "assistant", "content": "Sunny."}
            else:
                call = {"id": "c1", "type": "function", "function": {"name": "forecast"}}
                call["function"]["arguments"] = "{}"
                message = {"role": "assistant", "content": "Checking.", "tool_calls": [call]}
            data = json.dumps({"choices": [{"index": 0, "message": message}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    events: list[Any] = []
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    source = "@forecast() -> str:\n  = sunny\n\n> approval: allow\n> use: forecast\n" + ECHO
    try:
        with (
            JsonRpcAdapter(reference_bridge_command(base_url), model="m") as adapter,
            observe_agent_events(events.append),
        ):
            assert run(source, adapter) == "Sunny."
    finally:
        server.shutdown()
    messages = [(e.phase, e.content) for e in events if isinstance(e, AgentMessageEvent)]
    assert messages == [("commentary", "Checking."), ("final", "Sunny.")]


PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_prompt_media_crosses_the_bridge_in_kedis_layout(adapters: list) -> None:
    adapter = peer("media")
    adapters.append(adapter)
    assert adapter.capabilities.supports_media_input and adapter.media_kinds == {"image"}
    source = (
        f'> import: media\n[chart: Image] = `image("{PIXEL}")`\n'
        '[icon: Image] = `image("https://example.com/icon.png")`\n'
        "[reply] << Compare <<chart>> with <<icon>>.\n= <reply>\n"
    )
    parts = json.loads(run(source, adapter))
    assert [part["type"] for part in parts] == ["text", "media", "text", "media", "text"]
    chart, icon = parts[1], parts[3]
    assert chart == {
        "type": "media",
        "name": "chart",
        "kind": "image",
        "mediaType": "image/png",
        "bytes": True,
    }
    assert icon["url"] == "https://example.com/icon.png" and icon["bytes"] is False
    assert parts[0]["text"].startswith('<attachments>\n<image id="chart" type="image/png">')
    assert parts[-1]["text"].endswith("Compare chart with icon.")


def test_media_the_bridge_cannot_send_is_refused_before_the_run(adapters: list) -> None:
    from kedi.media import Audio, Image, MediaError

    adapter = peer("media")
    adapters.append(adapter)
    sound = Audio(media_type="audio/wav", data=b"RIFF0000WAVEfmt ")
    stored = Image(media_type="image/png", file_id="file-1", provider="openai")
    for kind, value, message in (
        ("Audio", sound, "sound: bridge 'peer' does not take audio input"),
        ("Image", stored, "a provider file id cannot cross a JSON-RPC bridge"),
    ):
        source = (
            f"> import: media\n[sound: {kind}] = `value`\n[reply] << Hear <<sound>>.\n= <reply>\n"
        )
        with pytest.raises(Exception) as caught:
            run(source, adapter, value=value)
        error = getattr(caught.value, "original", None) or caught.value
        assert isinstance(error, MediaError) and message in str(error)
    # A bridge that did not negotiate media gets none: Kedi refuses before the call.
    plain = peer("default")
    adapters.append(plain)
    error = failure(
        f'> import: media\n[c: Image] = `image("{PIXEL}")`\n[r] << See <<c>>.\n= <r>\n', plain
    )
    assert "does not accept media input" in str(error)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("media-no-kinds", "mediaInput needs mediaKinds"),
        ("media-bad-kinds", "mediaKinds must list"),
    ],
)
def test_media_capabilities_must_name_their_kinds(scenario: str, message: str) -> None:
    with pytest.raises(JsonRpcError, match=message):
        peer(scenario)


def test_sessions_carry_the_bridge_state_kedi_committed(adapters: list) -> None:
    adapter = peer("sessions")
    adapters.append(adapter)
    assert adapter.capabilities.supports_stateful_history
    source = (
        "> history: enabled\n[a] << First.\n[b] << Second.\n"
        "> model: other\n[c] << Third.\n= `[a, b, c]`\n"
    )
    first, second, third = run(source, adapter)
    assert first == "First. after []"
    assert second == "Second. after ['First.']"
    # Another model is another lane: its history starts empty.
    assert third == "Third. after []"
    # Without history, no session is sent.
    assert run(ECHO, adapter) == "Say hello. after [] (no session)"


def test_a_bridge_without_sessions_refuses_history(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    error = failure("> history: enabled\n" + ECHO, adapter)
    assert "does not support stateful history" in str(error)


def test_the_reference_bridge_replays_its_session_state() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[dict[str, Any]] = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            message = {"role": "assistant", "content": f"Answer {len(requests)}."}
            data = json.dumps({"choices": [{"index": 0, "message": message}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with JsonRpcAdapter(reference_bridge_command(base_url), model="m") as adapter:
            source = "> history: enabled\n[a] << First.\n[b] << Second.\n= `[a, b]`\n"
            assert run(source, adapter) == ["Answer 1.", "Answer 2."]
    finally:
        server.shutdown()
    second = requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert [message["content"] for message in second[1:]] == [
        "First.",
        "Answer 1.",
        "Second.",
    ]


ALLOW = "> approval: allow\n" + ECHO


def test_governed_bridges_ask_kedi_before_each_step(adapters: list) -> None:
    adapter = peer("governed")
    adapters.append(adapter)
    assert adapter.capabilities.supports_request_budget is True
    assert adapter.capabilities.supports_tool_attempt_budget is True
    replies = json.loads(run(ALLOW, adapter))
    assert replies["request"]["result"] == {"admitted": True}
    assert replies["tool attempt"]["result"] == {"admitted": True}
    assert replies["bad dimension"]["error"]["code"] == INVALID_PARAMS
    assert replies["approval"]["result"] == {"decision": "allow", "reason": "static allow policy"}
    assert replies["bad risk"]["error"]["code"] == INVALID_PARAMS
    denied = json.loads(run("> approval: deny\n" + ECHO, adapter))
    assert denied["approval"]["result"] == {"decision": "deny", "reason": "static deny policy"}
    # Without a policy, the bridge's tools are denied by default, as Kedi's tools are.
    assert json.loads(run(ECHO, adapter))["approval"]["result"]["decision"] == "deny"


def test_a_spent_budget_stops_the_bridge_before_the_step(adapters: list) -> None:
    adapter = peer("governed")
    adapters.append(adapter)
    error = failure("> budget:\n    request_limit: 0\n" + ECHO, adapter)
    assert isinstance(error, RunBudgetExceeded) and error.dimension == "request"
    # A bridge that takes the denied step anyway still fails: Kedi has already decided.
    error = failure("> budget:\n    tool_attempt_limit: 0\n" + ALLOW, adapter)
    assert isinstance(error, RunBudgetExceeded) and error.dimension == "tool_attempt"
    assert json.loads(run(ALLOW, adapter))["request"]["result"] == {"admitted": True}


def test_admission_needs_the_capability_it_belongs_to(adapters: list) -> None:
    kedi_tools, ungoverned, plain = (
        peer("governed-kedi-tools"),
        peer("native-ungoverned"),
        peer("default"),
    )
    adapters += [kedi_tools, ungoverned, plain]
    replies = json.loads(run(ALLOW, kedi_tools))
    assert replies["request"]["result"] == {"admitted": True}
    assert replies["tool attempt"]["error"]["code"] == UNSUPPORTED
    assert replies["approval"]["error"]["code"] == UNSUPPORTED
    replies = json.loads(run(ALLOW, ungoverned))
    assert replies["request"]["error"]["code"] == UNSUPPORTED
    assert replies["approval"]["result"]["decision"] == "allow"
    # Kedi refuses a ceiling it cannot enforce before the run: an ungoverned bridge's model
    # requests, and the attempts of tools of its own.
    assert ungoverned.capabilities.supports_tool_attempt_budget is False
    for adapter, budget in ((ungoverned, "tool_attempt"), (plain, "request")):
        error = failure(f"> budget:\n    {budget}_limit: 3\n" + ECHO, adapter)
        assert f"cannot enforce {budget} run budget" in str(error)
    # Kedi's own tools are admitted at tool/call, so any bridge enforces tool attempts.
    assert plain.capabilities.supports_tool_attempt_budget is True


def test_approval_handlers_decide_the_bridges_tools(adapters: list) -> None:
    adapter = peer("governed")
    adapters.append(adapter)
    seen: list[tuple[Any, ...]] = []

    def safer(request: Any) -> ApprovalDecision:
        seen.append((request.tool_name, dict(request.arguments), request.risk))
        return ApprovalDecision.edit({"command": "pwd"}, reason="read only")

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(safer))):
        replies = json.loads(adapter.invoke_sync(prompt="Probe."))
    assert replies["approval"]["result"] == {
        "decision": "edit",
        "reason": "read only",
        "arguments": {"command": "pwd"},
    }
    assert seen == [("shell", {"command": "ls"}, "mutating")]

    with adapter.profile_override(
        AgentProfile(approval=ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.allow()))
    ):
        replies = json.loads(adapter.invoke_sync(prompt="Probe."))
    assert replies["approval"]["result"] == {"decision": "allow"}

    def closed(_request: Any) -> ApprovalDecision:
        raise ApprovalCancelled(message="the user closed the prompt")

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(closed))):
        with pytest.raises(ApprovalCancelled):
            adapter.invoke_sync(prompt="Probe.")
        # The run fails with the first reason Kedi stopped it: here the spent budget, before
        # the bridge went on to ask for an approval the user then cancelled.
        with run_budget_scope(RunBudget(tool_attempt_limit=0)):
            with pytest.raises(RunBudgetExceeded):
                adapter.invoke_sync(prompt="Probe.")


def test_an_approval_still_pending_when_the_run_ends_is_cancelled() -> None:
    adapter = peer("abandon-approval")

    async def undecided(_request: Any) -> ApprovalDecision:
        await asyncio.sleep(30)
        raise AssertionError("the decision should have been cancelled")

    try:
        with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(undecided))):
            assert adapter.invoke_sync(prompt="Go.") == "answered early"
        deadline = time.monotonic() + 10
        while f"approval reply {CANCELLED}" not in adapter._connection.stderr_tail:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        adapter.close()


def test_the_conformance_kit_catches_a_retry_kedi_did_not_admit() -> None:
    scenario = next(scenario for scenario in SCENARIOS if scenario.name == "request-budget")
    variant = scenario.variants[1]  # the endpoint fails the first request
    with RecordingEndpoint() as endpoint:
        result = _run(
            endpoint,
            lambda url: JsonRpcAdapter(
                [sys.executable, str(PEER), "retries-unadmitted", url], model="m"
            ),
            variant,
        )
    problems = _check_run(scenario, variant, result)
    assert "2 requests reached the model endpoint, expected 1" in problems


SUBAGENTS = """
@lookup(query: str) -> str:
    = found-<query>

> profile: worker:
    ###
    Handles one isolated worker task.
    ###
    > approval: allow
    > use: lookup

> profile: manager:
    > approval: allow
    > subagent: worker
"""


def delegation_tools(runtime: Any, adapter: JsonRpcAdapter) -> dict[str, Any]:
    profile = eval_profile_definition(
        runtime, runtime._profile_registry["manager"], build_capture_env(runtime)
    )
    return {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}


def test_kedi_runs_its_children_on_a_governed_bridge(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    assert adapter.capabilities.supports_subagents is True
    assert adapter.capabilities.supports_background_subagents is True
    assert adapter.supports_native_subagent_resume is True
    runtime = compile_program(parse_program(SUBAGENTS), adapter=adapter)
    tools = delegation_tools(runtime, adapter)
    first = asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Find q."))
    replies = json.loads(first["task_summary"])
    lineage = replies["subagent"]
    assert lineage["id"] == first["run_id"] and lineage["name"] == "worker"
    assert lineage["parentRunId"] is None  # started by Python, not from a bridge run
    assert lineage["limits"] == {
        "requests": 8,
        "toolCalls": 16,
        "inputTokens": None,
        "outputTokens": None,
        "totalTokens": None,
    }
    assert all(reply["result"] == {"admitted": True} for reply in replies["admissions"])
    assert all(reply["result"]["result"] == "found-q" for reply in replies["lookups"])
    assert replies["approval"]["result"]["decision"] == "allow"
    assert replies["session"] == {"id": lineage["conversationId"], "turn": 0, "state": None}
    # Kedi counted the child's two admitted requests and two tool calls, nothing twice: the
    # bridge's own request count is ignored, its token counts are kept.
    record = runtime._subagent_coordinator.runs.snapshot()[-1]
    assert (record.requests, record.tool_calls, record.total_tokens) == (2, 2, 4)
    # A continuation resumes from the bridge state the child's last run committed.
    second = asyncio.run(tools["continue_subagent"].fn(run_id=first["run_id"], task="Again."))
    session = json.loads(second["task_summary"])["session"]
    assert session == {
        "id": lineage["conversationId"],
        "turn": 1,
        "state": {"turn": 0, "previous": None},
    }
    # A continuation is a new task with limits of its own, counted on its own record.
    assert json.loads(second["task_summary"])["subagent"]["limits"] == lineage["limits"]
    record = runtime._subagent_coordinator.runs.snapshot()[-1]
    assert (record.run_id, record.requests, record.tool_calls) == (second["run_id"], 2, 2)


def test_a_child_started_from_a_bridge_run_names_that_run(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    parent = json.loads(
        run(SUBAGENTS + "\n> use: manager\n[r] << Delegate one task.\n= <r>\n", adapter)
    )
    delegated = parent["reply"]["result"]
    assert delegated["ok"] is True
    child = json.loads(delegated["result"]["task_summary"])
    assert child["subagent"]["parentRunId"] == parent["runId"]


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (SubagentUsageLimits(request_limit=1), "request limit 1 exceeded"),
        (SubagentUsageLimits(tool_calls_limit=1), "tool call limit 1 exceeded"),
    ],
)
def test_a_childs_limits_stop_the_step_before_the_bridge_takes_it(
    adapters: list, limits: SubagentUsageLimits, message: str
) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    runtime = compile_program(
        parse_program(SUBAGENTS), adapter=adapter, subagent_usage_limits=limits
    )
    tools = delegation_tools(runtime, adapter)
    with pytest.raises(KediExecutionError, match=message):
        asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Find q."))
    assert runtime._subagent_coordinator.runs.snapshot()[-1].status == "failed"


def test_a_childs_bridge_tools_answer_to_its_parents_policy(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    source = SUBAGENTS.replace(
        "> profile: manager:\n    > approval: allow", "> profile: manager:\n    > approval: deny"
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    tools = delegation_tools(runtime, adapter)
    replies = json.loads(
        asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Find q."))["task_summary"]
    )
    # The worker allows; its parent's deny is the ceiling, for Kedi's tools and the bridge's.
    assert replies["approval"]["result"] == {"decision": "deny", "reason": "static deny policy"}
    assert all(reply["result"]["ok"] is False for reply in replies["lookups"])


def test_structured_children_are_corrected_with_kedis_feedback(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    runtime = compile_program(parse_program(SUBAGENTS), adapter=adapter)
    tools = delegation_tools(runtime, adapter)
    schema = {
        "type": "object",
        "properties": {"ocean": {"type": "string"}},
        "required": ["ocean"],
    }
    result = asyncio.run(
        tools["delegate_task"].fn(subagent="worker", task="Name an ocean.", final_schema=schema)
    )
    assert result["final_result"] == {"ocean": "Pacific"}
    # The bridge's second answer quotes Kedi's correction of its first.
    assert "task_summary and final_result" in result["task_summary"]


def test_without_sessions_kedi_replays_the_childs_turns(adapters: list) -> None:
    adapter = peer("subagents-stateless")
    adapters.append(adapter)
    assert adapter.supports_native_subagent_resume is False
    runtime = compile_program(parse_program(SUBAGENTS), adapter=adapter)
    tools = delegation_tools(runtime, adapter)
    first = asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Find q."))
    second = asyncio.run(tools["continue_subagent"].fn(run_id=first["run_id"], task="Again."))
    replies = json.loads(second["task_summary"])
    assert replies["session"] is None
    assert replies["user"].startswith("Continue the same child task using this verified prior")


def test_only_the_childs_own_turn_is_the_child(adapters: list) -> None:
    adapter = peer("subagents-stateless")
    adapters.append(adapter)
    request = SubagentExecutionRequest(
        profile=AgentProfile(name="worker"),
        tools=(),
        instructions=None,
        approval_ceiling=(),
        usage_limits=SubagentUsageLimits(),
    )
    with adapter.subagent_scope(request):
        assert adapter._child.get() is request
        with adapter.tool_scope(()):
            # A Kedi call made by one of the child's tools is an ordinary call.
            assert adapter._child.get() is None
        replies = json.loads(adapter.invoke_sync(prompt="Direct."))
    # Called outside Kedi's coordinator: no Kedi run id, and the request's own limits.
    assert replies["subagent"]["id"] is None
    assert replies["subagent"]["limits"]["requests"] == 8
    with pytest.raises(KediExecutionError, match="only inside subagent_scope"):
        asyncio.run(adapter.produce_subagent(template="x", output_schema={"type": "object"}))


def manager_run(adapter: JsonRpcAdapter, prompt: str) -> dict[str, Any]:
    return json.loads(run(SUBAGENTS + f"\n> use: manager\n[r] << {prompt}\n= <r>\n", adapter))


def test_a_background_child_runs_beside_its_bridge_parent(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    parent = manager_run(adapter, "Delegate in the background.")
    started = parent["reply"]["result"]["result"]
    assert started["status"] == "running"
    waited = parent["wait_subagent"]["result"]["result"]
    assert waited["run_id"] == started["run_id"]
    child = json.loads(waited["task_summary"])
    # The child ran on the owner loop while its parent's run waited on the same connection.
    assert child["subagent"]["parentRunId"] == parent["runId"]
    assert all(reply["result"]["ok"] for reply in child["lookups"])


def test_cancelling_a_background_child_cancels_its_bridge_run(adapters: list) -> None:
    adapter = peer("subagents")
    adapters.append(adapter)
    parent = manager_run(adapter, "Delegate, then cancel.")
    assert parent["cancel_subagent"]["result"]["result"]["cancel_requested"] is True
    deadline = time.monotonic() + 10
    while not any(line.startswith("cancel ") for line in adapter._connection.stderr_tail):
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_bridge_that_dies_during_a_child_releases_its_slot_once() -> None:
    adapter = peer("subagents-die")
    try:
        runtime = compile_program(parse_program(SUBAGENTS), adapter=adapter)
        tools = delegation_tools(runtime, adapter)
        with pytest.raises(Exception) as caught:
            asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Find q."))
        assert "bridge crashed during a child" in str(caught.value)
        coordinator = runtime._subagent_coordinator
        assert coordinator.runs.snapshot()[-1].status == "failed"
        # The slot was released exactly once: a second release would raise "slot underflow".
        assert coordinator._active_runs == 0
    finally:
        adapter.close()


def test_live_input_reaches_a_running_child(adapters: list) -> None:
    adapter = peer("live")
    adapters.append(adapter)
    assert adapter.capabilities.supports_live_subagent_input is True
    assert adapter.capabilities.supports_subagent_interrupt is True
    parent = manager_run(adapter, "Send input.")
    assert parent["sent"]["result"]["result"]["status"] == "queued"
    child = json.loads(parent["waited"]["result"]["result"]["task_summary"])
    # Kedi's rendering of the input reached the child's bridge run while it ran.
    (delivered,) = child["inputs"]
    assert delivered.startswith("Kedi agent input:") and "Use the blue one." in delivered


def test_an_interrupt_restarts_the_child_from_its_interrupted_state(adapters: list) -> None:
    adapter = peer("live")
    adapters.append(adapter)
    parent = manager_run(adapter, "Interrupt.")
    assert parent["sent"]["result"]["result"]["status"] == "interrupt_requested"
    child = json.loads(parent["waited"]["result"]["result"]["task_summary"])
    # The next attempt carries the input, and resumes from the interrupted attempt's state.
    assert child["user"].startswith("Continue with the updated task input.")
    assert "Use the blue one." in child["user"]
    assert child["session"]["state"]["partial"]


def test_per_request_usage_counts_tokens_and_holds_a_childs_token_limits() -> None:
    adapter = peer("live")
    try:
        assert adapter.capabilities.supports_token_usage is True
        source = SUBAGENTS + "\n> use: manager\n[r] << Count tokens.\n= <r>\n"
        runtime = compile_program(parse_program(source), adapter=adapter)
        try:
            runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())
        record = runtime._subagent_coordinator.runs.snapshot()[-1]
        # Two admitted requests; tokens from run/usage, not from the result's usage.
        assert (record.requests, record.input_tokens, record.total_tokens) == (2, 6, 8)
        limited = compile_program(
            parse_program(source),
            adapter=adapter,
            subagent_usage_limits=SubagentUsageLimits(total_tokens_limit=5),
        )
        try:
            waited = json.loads(limited.run_main())["waited"]["result"]["result"]
        finally:
            asyncio.run(limited.aclose())
        assert waited["status"] == "failed"
        assert "total token limit 5 exceeded" in waited["error"]
    finally:
        adapter.close()


class HeldEndpoint(RecordingEndpoint):
    """Holds the first model request until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.arrived = threading.Event()
        self.release = threading.Event()

    def _answer(self, protocol: Any, body: dict[str, Any]) -> dict[str, Any]:
        if not self.arrived.is_set():
            self.arrived.set()
            self.release.wait(10)
        return super()._answer(protocol, body)


def lookup_tool() -> ToolSpec:
    def lookup(query: str) -> str:
        return f"found-{query}"

    return ToolSpec(
        fn=lookup,
        name="lookup",
        json_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        risk="read_only",
    )


def test_the_reference_bridge_gives_live_input_to_its_next_request() -> None:
    with HeldEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="lookup", arguments={"query": "q"})
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        control = LiveInput("child")
        control.configure(input=True, interrupt=True)

        async def attempt() -> str:
            with live_input_scope(control), adapter.tool_scope((lookup_tool(),)):
                running = asyncio.ensure_future(adapter.invoke(prompt="Look it up."))
                await asyncio.to_thread(endpoint.arrived.wait, 10)
                receipt = control.admit("Use the blue one.", sender_id="parent")
                await asyncio.sleep(0.3)  # Kedi sends run/input while the request is held
                endpoint.release.set()
                answer = await running
            assert receipt.message_id in control.delivered
            return answer

        try:
            assert asyncio.run(attempt()) == "done"
        finally:
            endpoint.release.set()
            adapter.close()
        first, second = endpoint.take()
        assert "Use the blue one." not in json.dumps(first.body)
        assert "Use the blue one." in json.dumps(second.body)


def test_the_reference_bridge_stops_an_interrupted_attempt() -> None:
    with HeldEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        control = LiveInput("child")
        control.configure(input=True, interrupt=True)

        async def attempt() -> None:
            with live_input_scope(control):
                running = asyncio.ensure_future(adapter.invoke(prompt="Take your time."))
                await asyncio.to_thread(endpoint.arrived.wait, 10)
                control.admit("Stop.", sender_id="parent", interrupt=True)
                await running

        try:
            with pytest.raises(RunInterrupted):
                asyncio.run(attempt())
        finally:
            endpoint.release.set()
            adapter.close()


def test_the_reference_bridge_reports_each_requests_usage(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="lookup", arguments={"query": "q"})
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        assert adapter.capabilities.supports_token_usage is True
        usage: list[InvocationUsage] = []
        with observe_invocations(usage.append), adapter.tool_scope((lookup_tool(),)):
            assert asyncio.run(adapter.invoke(prompt="Look it up.")) == "done"
        total = sum(usage, InvocationUsage())
        # Two requests, each admitted and each reporting the endpoint's 1 + 1 tokens.
        assert (total.requests, total.tool_calls, total.total_tokens) == (2, 1, 4)


def test_input_a_bridge_refuses_waits_for_the_next_attempt(adapters: list) -> None:
    adapter = peer("default")
    adapters.append(adapter)
    item = AgentInput(
        message_id="m1", target_id="child", sender_id="parent", revision=0, message="Hi."
    )
    # Kedi counts the input undelivered, so the coordinator gives it to the next attempt.
    assert asyncio.run(adapter._deliver("run-1", (item,))) is False


CONFIGURED = (
    """> model: primary
> fallback:
    backup-1
    backup-2
> effort: high
> settings:
    temperature: 0.2
    context_limit: 40000
"""
    + ECHO
)


def test_settings_effort_and_fallbacks_reach_the_bridge_that_named_them(adapters: list) -> None:
    adapter, plain = peer("configured"), peer("default")
    adapters += [adapter, plain]
    assert adapter.capabilities.settings == frozenset({"temperature"})
    assert adapter.capabilities.supports_effort and adapter.capabilities.supports_model_fallback
    sent = json.loads(run(CONFIGURED, adapter))
    # Kedi applies its own settings (context_limit); the bridge gets the names it listed.
    assert sent == {
        "model": "primary",
        "fallbackModels": ["backup-1", "backup-2"],
        "settings": {"temperature": 0.2},
        "effort": "high",
    }
    for directive, refused in (
        ("> effort: max\n", "effort"),
        ("> settings:\n    top_p: 0.5\n", "top_p"),
    ):
        error = failure(directive + ECHO, adapter)
        assert f"does not support profile fields: {refused}" in str(error)
    # A bridge that named none takes none, and Kedi refuses > fallback: before the run.
    assert "does not support profile fields: effort" in str(
        failure("> effort: low\n" + ECHO, plain)
    )
    assert "model_fallback" in str(failure("> fallback: backup-1\n" + ECHO, plain))


class BrokenModelEndpoint(RecordingEndpoint):
    """Fails every request for the model named broken."""

    def _answer(self, protocol: Any, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("model") == "broken":
            raise RuntimeError("the model is down")
        return super()._answer(protocol, body)


def test_the_reference_bridge_falls_back_and_applies_settings_and_effort() -> None:
    source = (
        """> model: broken
> fallback: working
> effort: low
> settings:
    temperature: 0.2
    stop_sequences: `["END"]`
"""
        + ECHO
    )
    with BrokenModelEndpoint() as endpoint:
        endpoint.script = Script(final="Hello.", text=True)
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        try:
            assert run(source, adapter) == "Hello."
            broken, working = endpoint.take()
            assert (broken.body["model"], working.body["model"]) == ("broken", "working")
            assert working.body["temperature"] == 0.2 and working.body["stop"] == ["END"]
            assert working.body["reasoning_effort"] == "low"
            # Each attempt was admitted: a one-request budget stops the fallback.
            error = failure("> budget:\n    request_limit: 1\n" + source, adapter)
            assert isinstance(error, RunBudgetExceeded)
            assert [exchange.body["model"] for exchange in endpoint.take()] == ["broken"]
        finally:
            adapter.close()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def test_the_reference_bridge_keeps_line_separators_inside_a_frame(adapters: list) -> None:
    # Kedi writes U+2028 and U+2029 unescaped; a frame ends at LF only.
    prompt = "one two three\rfour"
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        assert asyncio.run(adapter.invoke(prompt=prompt)) == "done"
        (exchange,) = endpoint.take()
        assert prompt in _strings(exchange.body["messages"])


def test_a_bridge_model_draws_and_speaks(adapters: list) -> None:
    from kedi.media import Audio, Image

    adapter = peer("drawing")
    adapters.append(adapter)
    capabilities = adapter.capabilities
    assert capabilities.supports_image_output and capabilities.supports_audio_output
    image = run("> import: media\n[img: Image] << Draw a cat.\n= `img`\n", adapter)
    assert isinstance(image, Image) and image.media_type == "image/png"
    assert image.data is not None and image.data.startswith(b"\x89PNG")
    # With nothing to name, the model can only draw: the output holds no schema.
    seen = json.loads(run("[seen] << What did you get?\n= <seen>\n", adapter))
    assert seen["output"] == {"media": "image"}
    # With an image in the request, the model may name it instead of drawing.
    source = (
        f'> import: media\n[photo: Image] = `image("{PIXEL}")`\n'
        "[img: Image] << Pick one. <<photo>>\n= `img`\n"
    )
    picked = run(source, adapter)
    # The model named the request's photo: Kedi returns that image, not a drawing.
    assert isinstance(picked, Image) and picked.data == base64.b64decode(PIXEL.split(",")[1])
    seen = json.loads(run("[seen] << What did you get?\n= <seen>\n", adapter))
    assert seen["output"]["media"] == "image" and seen["output"]["schema"] == {"type": "string"}

    async def speak() -> Any:
        profile = AgentProfile(settings={"voice": "nova"})
        with adapter.profile_override(profile), adapter.tool_scope((lookup_tool(),)):
            return await adapter.produce(template="Say hello.", output_type=Audio)

    speech = asyncio.run(speak())
    assert isinstance(speech, Audio) and speech.media_type == "audio/wav"
    seen = json.loads(run("[seen] << What did you get?\n= <seen>\n", adapter))
    # A speech request carries the voice, and no tools.
    assert seen == {"output": {"media": "audio", "voice": "nova"}, "tools": []}


def test_a_bridge_answers_with_the_media_it_was_asked_for(adapters: list) -> None:
    from kedi.media import Image

    adapter = peer("drawing")
    adapters.append(adapter)

    def draw(prompt: str) -> Any:
        return asyncio.run(adapter.produce(template=prompt, output_type=Image))

    linked = draw("Link it.")
    assert isinstance(linked, Image) and linked.url == "https://example.com/cat.png"
    for prompt, message in (
        ("Answer with sound.", "the run asked for one image; its 'content' holds 0"),
        ("Draw two.", "the run asked for one image; its 'content' holds 2"),
        ("Answer with parts.", "'content' must list text and media parts"),
        ("Answer with garbage.", "a media part needs base64 'data' or a 'url'"),
    ):
        with pytest.raises(JsonRpcError, match=re.escape(message)):
            draw(prompt)


def test_a_bridge_without_media_output_still_only_names_media(adapters: list) -> None:
    from kedi.media import MediaError

    adapter = peer("media")
    adapters.append(adapter)
    assert not adapter.capabilities.supports_image_output
    error = failure("> import: media\n[img: Image] << Draw a cat.\n= `img`\n", adapter)
    assert isinstance(error, MediaError) and "holds no image to name" in str(error)
    error = failure("> import: media\n[speech: Audio] << Say hello.\n= `speech`\n", adapter)
    assert isinstance(error, MediaError) and "cannot produce speech" in str(error)


class MediaEndpoint(RecordingEndpoint):
    """Draws for image requests and speaks for audio ones, as chat completions report it."""

    def _answer(self, protocol: Any, body: dict[str, Any]) -> dict[str, Any]:
        answer = super()._answer(protocol, body)
        message = answer["choices"][0]["message"]
        modalities = body.get("modalities") or []
        if "image" in modalities and "Draw" in json.dumps(body["messages"]):
            png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
            message["images"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}
            ]
        if "audio" in modalities:
            message["audio"] = {"data": base64.b64encode(b"RIFF\x00WAVE").decode()}
        return answer


def test_the_reference_bridge_draws_and_speaks_when_its_endpoint_does(adapters: list) -> None:
    from kedi.media import Audio, Image

    with MediaEndpoint() as endpoint:
        endpoint.script = Script(final="photo", text=True)
        command = reference_bridge_command(
            f"{endpoint.base_url}/v1", image_output=True, audio_output=True
        )
        adapter = JsonRpcAdapter(command)
        adapters.append(adapter)
        assert adapter.capabilities.supports_image_output
        drawn = run("> import: media\n[img: Image] << Draw a cat.\n= `img`\n", adapter)
        assert isinstance(drawn, Image) and drawn.data == b"\x89PNG\r\n\x1a\n"
        (drawing,) = endpoint.take()
        assert drawing.body["modalities"] == ["image", "text"]
        assert "response_format" not in drawing.body
        # A model that writes a name instead picks the request's image.
        source = (
            f'> import: media\n[photo: Image] = `image("{PIXEL}")`\n'
            "[img: Image] << Pick <<photo>>.\n= `img`\n"
        )
        picked = run(source, adapter)
        assert isinstance(picked, Image)
        assert picked.data == base64.b64decode(PIXEL.split(",")[1])
        endpoint.take()
        profile = AgentProfile(settings={"voice": "nova"})

        async def speak() -> Any:
            with adapter.profile_override(profile):
                return await adapter.produce(template="Say hello.", output_type=Audio)

        speech = asyncio.run(speak())
        assert isinstance(speech, Audio) and speech.data == b"RIFF\x00WAVE"
        (spoken,) = endpoint.take()
        assert spoken.body["audio"] == {"voice": "nova", "format": "wav"}
        assert "tools" not in spoken.body


def test_the_reference_bridge_claims_no_media_output_by_default(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        assert not adapter.capabilities.supports_image_output
        assert not adapter.capabilities.supports_audio_output


def test_the_programs_hooks_run_around_a_bridges_own_tool(adapters: list) -> None:
    from kedi.agent_adapter.hooks import PreToolUseDecision

    adapter = peer("hooked")
    adapters.append(adapter)
    hooks = adapter.capabilities.hooks
    assert hooks is not None and hooks.pre_tool_use is not None
    assert hooks.pre_tool_use.origins == {"kedi", "adapter_native", "provider_builtin"}
    source = "> approval: allow\n[r] << {prompt}\n= <r>\n"
    # With no handlers, the run lists no events and the bridge sends no hook/invoke.
    quiet = json.loads(run(source.format(prompt="Run ls."), adapter))
    assert quiet == {"hooks": [], "approval": quiet["approval"]}
    events: list[Any] = []

    def pre(event: Any) -> Any:
        events.append(("pre", event.tool_name, event.origin, dict(event.arguments)))
        if event.arguments["command"].startswith("rm"):
            return PreToolUseDecision.deny("no deleting")
        if event.arguments["command"] == "ls":
            return PreToolUseDecision.edit({"command": "ls -la"})
        return None

    adapter.on("pre_tool_use")(pre)
    adapter.on("post_tool_use")(lambda event: events.append(("post", event.result)))
    adapter.on("post_tool_use_failure")(
        lambda event: events.append(("failed", event.error_type, event.error_message))
    )
    ran = json.loads(run(source.format(prompt="Run ls."), adapter))
    assert ran["hooks"] == ["post_tool_use", "post_tool_use_failure", "pre_tool_use"]
    # The edit reaches the bridge, which asks approval for the edited arguments.
    assert ran["pre"]["result"] == {"decision": "edit", "arguments": {"command": "ls -la"}}
    assert ran["approval"]["result"]["decision"] == "allow"
    assert ran["post"]["result"] == {}
    denied = json.loads(run(source.format(prompt="Run rm."), adapter))
    assert denied["pre"]["result"]["decision"] == "deny"
    assert "no deleting" in denied["pre"]["result"]["reason"]
    failed = json.loads(run(source.format(prompt="Run a failing command."), adapter))
    assert failed["post"]["result"] == {}
    assert events == [
        ("pre", "shell", "adapter_native", {"command": "ls"}),
        ("post", "a b"),
        ("pre", "shell", "adapter_native", {"command": "rm -rf /"}),
        ("pre", "shell", "adapter_native", {"command": "false"}),
        ("failed", "CalledProcessError", "exit status 1"),
    ]


def test_a_failing_hook_stops_the_bridge_run(adapters: list) -> None:
    adapter = peer("hooked")
    adapters.append(adapter)

    def broken(event: Any) -> Any:
        raise RuntimeError("audit store is down")

    adapter.on("pre_tool_use")(broken)
    error = failure("> approval: allow\n[r] << Run ls.\n= <r>\n", adapter)
    assert "audit store is down" in f"{error} {error.__cause__}"


def test_hook_invoke_needs_a_declared_origin_and_a_well_formed_call(adapters: list) -> None:
    adapter = peer("hooked")
    adapters.append(adapter)
    base = {
        "runId": "r",
        "event": "pre_tool_use",
        "callId": "c",
        "tool": "shell",
        "origin": "adapter_native",
        "arguments": {},
    }
    for params, message in (
        ({"origin": "mcp"}, "hook/invoke for 'mcp' tools: the bridge runs no such tools"),
        ({"event": "teleport"}, "hook/invoke needs an event of"),
    ):
        with pytest.raises(JsonRpcError, match=re.escape(message)):
            adapter._parse_hook({**base, **params})


def test_kedi_tool_hooks_and_the_prompt_hook_run_on_any_bridge(adapters: list) -> None:
    from kedi.agent_adapter.hooks import PreToolUseDecision, UserPromptSubmitDecision

    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="lookup", arguments={"query": "q"})
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        seen: list[Any] = []
        adapter.on("user_prompt_submit")(
            lambda event: UserPromptSubmitDecision.edit(event.content + " Be brief.")
        )
        adapter.on("pre_tool_use")(lambda event: PreToolUseDecision.edit({"query": "edited"}))
        adapter.on("post_tool_use")(lambda event: seen.append((event.origin, event.result)))
        with adapter.tool_scope((lookup_tool(),)):
            assert asyncio.run(adapter.invoke(prompt="Look it up.")) == "done"
        first, second = endpoint.take()
        assert "Look it up. Be brief." in _strings(first.body["messages"])
        # The hook's edit is what the tool ran with.
        assert seen == [("kedi", "found-edited")]
        assert "found-edited" in json.dumps(second.body)


def test_a_programs_prompt_hook_runs_once_on_a_bridge(adapters: list) -> None:
    from kedi.agent_adapter.hooks import UserPromptSubmitDecision

    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        adapter.on("user_prompt_submit")(
            lambda event: UserPromptSubmitDecision.edit(event.content + " Be brief.")
        )
        assert run("[r] << Say hi.\n= <r>\n", adapter) == "done"
        (exchange,) = endpoint.take()
        texts = _strings(exchange.body["messages"])
        assert any(text.endswith("Say hi. Be brief.") for text in texts)
        assert not any("Be brief. Be brief." in text for text in texts)


def test_a_hook_still_running_when_the_run_ends_is_cancelled(adapters: list) -> None:
    import kedi.agent_adapter.adapters.jsonrpc.adapter as module

    adapter = peer("hooked")
    adapters.append(adapter)

    async def slow(event: Any) -> Any:
        await asyncio.sleep(30)
        raise AssertionError("the hook should have been cancelled")

    adapter.on("pre_tool_use")(slow)
    assert adapter.invoke_sync(prompt="Answer early.") == "answered early"
    deadline = time.monotonic() + 10
    while f"hook reply {CANCELLED}" not in adapter._connection.stderr_tail:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    # Kedi only describes a bridge tool for its hooks; the bridge runs it.
    with pytest.raises(RuntimeError, match="a bridge tool runs in the bridge"):
        module._bridge_tool(command="ls")


MCP_SERVER = Path(__file__).parent / "fixtures" / "mcp_tools_server.py"


def mcp_program(prompt: str, *, profile: str = "> approval: allow\n") -> str:
    server = f"    command: {sys.executable}\n    args: `[{str(MCP_SERVER)!r}]`\n"
    return f"{profile}> mcp:\n{server}[r] << {prompt}\n= <r>\n"


def mcp_bridge(endpoint: RecordingEndpoint, adapters: list) -> JsonRpcAdapter:
    adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
    adapters.append(adapter)
    return adapter


def tool_results(exchange: Any) -> list[Any]:
    messages = exchange.body["messages"]
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


def test_the_reference_bridge_runs_mcp_tools_under_kedis_controls(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="shout", arguments={"text": "hi"})
        adapter = mcp_bridge(endpoint, adapters)
        capabilities = adapter.capabilities
        assert capabilities.supports_mcp and adapter.mcp_transports == {"stdio"}
        assert capabilities.supports_native_artifacts
        seen: list[Any] = []
        adapter.on("pre_tool_use")(
            lambda event: seen.append(("pre", event.origin, event.tool_name))
        )
        adapter.on("post_tool_use")(lambda event: seen.append(("post", event.result)))
        assert run(mcp_program("Shout hi."), adapter) == "done"
        first, second = endpoint.take()
        assert "shout" in {tool["function"]["name"] for tool in first.body["tools"]}
        assert tool_results(second) == [{"ok": True, "result": {"result": "HI"}}]
        assert seen == [("pre", "mcp", "shout"), ("post", {"result": "HI"})]
        # Without an approval policy, Kedi denies the mutating MCP call, as for Kedi tools.
        assert run(mcp_program("Shout hi.", profile=""), adapter) == "done"
        _, denied = endpoint.take()
        (result,) = tool_results(denied)
        assert result["ok"] is False


def test_an_mcp_tools_large_result_reaches_the_model_as_an_artifact(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="dump", arguments={"lines": 400})
        adapter = mcp_bridge(endpoint, adapters)
        profile = "> approval: allow\n> artifacts:\n    threshold: 1kb\n"
        assert run(mcp_program("Dump the log.", profile=profile), adapter) == "done"
        _, second = endpoint.take()
        ((result,),) = [[item["result"]] for item in tool_results(second)]
        # Kedi kept the log and gave the model a reference to read it in pieces.
        assert "ref_id" in result and "log line 399" not in json.dumps(result)


def test_an_mcp_tool_failure_reaches_the_programs_failure_hook(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="broken")
        adapter = mcp_bridge(endpoint, adapters)
        failures: list[Any] = []
        adapter.on("post_tool_use_failure")(lambda event: failures.append(event.error_message))
        assert run(mcp_program("Break it."), adapter) == "done"
        _, second = endpoint.take()
        (result,) = tool_results(second)
        assert result["ok"] is False and "the disk is full" in result["error"]
        assert len(failures) == 1 and "the disk is full" in failures[0]


def test_a_bridge_refuses_mcp_transports_it_did_not_list(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        adapter = mcp_bridge(endpoint, adapters)
        source = "> mcp:\n    transport: http\n    url: https://tools.example.com/mcp\n[r] << Hi.\n= <r>\n"
        error = failure(source, adapter)
        assert "does not support profile fields: mcp (http)" in str(error)


def test_artifact_admit_needs_native_tools_and_a_well_formed_result(adapters: list) -> None:
    plain, native = peer("default"), peer("hooked")
    adapters += [plain, native]
    call = {"runId": "r", "callId": "c", "tool": "shell", "origin": "adapter_native"}
    with pytest.raises(JsonRpcError, match="artifact/admit needs the nativeTools capability"):
        plain._parse_artifact({**call, "result": "x"})
    with pytest.raises(JsonRpcError, match="artifact/admit needs a string tool and callId"):
        native._parse_artifact(call)
    with pytest.raises(JsonRpcError, match="an origin of adapter_native, provider_builtin"):
        native._parse_approval({**call, "arguments": {}, "risk": "mutating", "origin": "mcp"})


def test_a_result_kedi_cannot_admit_stops_the_run(adapters: list, monkeypatch: Any) -> None:
    import kedi.agent_adapter.adapters.jsonrpc.adapter as module

    adapter = peer("hooked")
    adapters.append(adapter)
    # Outside a Kedi request there is no artifact scope: the result comes back as is.
    assert json.loads(adapter.invoke_sync(prompt="Admit it."))["result"] == {"result": "a b"}

    async def too_large(*_: Any, **__: Any) -> Any:
        raise RuntimeError("the result is too large to admit")

    monkeypatch.setattr(module, "admit_current_artifact_result", too_large)
    with pytest.raises(Exception, match="too large to admit"):
        adapter.invoke_sync(prompt="Admit it.")


def test_an_admission_still_running_when_the_run_ends_is_cancelled(
    adapters: list, monkeypatch: Any
) -> None:
    import kedi.agent_adapter.adapters.jsonrpc.adapter as module

    async def slow(*_: Any, **__: Any) -> Any:
        await asyncio.sleep(30)

    monkeypatch.setattr(module, "admit_current_artifact_result", slow)
    adapter = peer("hooked")
    adapters.append(adapter)
    assert adapter.invoke_sync(prompt="Admit early.") == "answered early"
    deadline = time.monotonic() + 10
    while f"admit reply {CANCELLED}" not in adapter._connection.stderr_tail:
        assert time.monotonic() < deadline
        time.sleep(0.05)


def wait_for_stderr(adapter: JsonRpcAdapter, line: str) -> None:
    deadline = time.monotonic() + 10
    while line not in adapter._connection.stderr_tail:
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_bridge_failure_quotes_the_harness_error_and_stderr(adapters: list) -> None:
    adapter = peer("harness-errors")
    adapters.append(adapter)
    error = failure("[r] << Fail.\n= <r>\n", adapter)
    assert isinstance(error, JsonRpcError)
    text = str(error)
    assert text.startswith("the harness failed\n")
    assert "harness error: ProviderError: 429 rate limited" in text
    assert "  caused by: HTTPError: Too Many Requests" in text
    assert "bridge stderr:\n  [harness] retrying the provider\n  [harness] giving up" in text
    # What Kedi sends a bridge stays the bare message.
    assert error.to_json() == {"code": -32002, "message": "the harness failed"}


def test_kedis_own_error_keeps_what_the_bridge_reported(adapters: list) -> None:
    adapter = peer("harness-errors")
    adapters.append(adapter)
    runtime = compile_program(
        parse_program("> budget:\n    request_limit: 1\n[r] << Spend.\n= <r>\n"), adapter=adapter
    )
    try:
        with pytest.raises(KediExecutionError) as caught:
            runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    error = caught.value
    assert isinstance(error, RunBudgetExceeded)
    bridge = error.__cause__
    assert isinstance(bridge, JsonRpcError) and bridge.code == RUN_FAILED
    # Kedi's error names its cause first, then what the harness said behind it.
    rendered = error.render()
    assert rendered.index("Run request budget exhausted") < rendered.index("from the bridge:")
    assert "  harness error: ProviderError: 429 rate limited" in rendered
    assert "  bridge stderr:" in rendered


def test_a_bridge_error_without_a_report_renders_as_before() -> None:
    plain = JsonRpcError(RUN_FAILED, "model endpoint answered 500")
    assert plain.bridge_report is None and str(plain) == "model endpoint answered 500"
    odd = JsonRpcError(RUN_FAILED, "failed", {"harness": {"message": "no type"}})
    assert odd.bridge_report == "harness error: Error: no type"


def test_kedis_error_rendering_quotes_at_most_four_bridge_causes() -> None:
    from kedi.error_rendering import _bridge_causes

    top = RuntimeError("Kedi stopped it")
    current: BaseException = top
    for number in range(6):
        # A cause without a report (a plain error) is passed over, not quoted.
        cause: BaseException = (
            ValueError("plain")
            if number == 1
            else JsonRpcError(RUN_FAILED, f"bridge {number}", {"harness": {"message": "x"}})
        )
        current.__cause__ = cause
        current = cause
    lines = _bridge_causes(top)
    assert [line for line in lines if line.startswith("from the bridge:")] == [
        "from the bridge: bridge 0",
        "from the bridge: bridge 2",
        "from the bridge: bridge 3",
    ]


def test_kedi_holds_the_transcript_and_sends_it_each_turn(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    capabilities = adapter.capabilities
    assert capabilities.supports_stateful_history and capabilities.supports_history_replay
    source = (
        "> history: enabled\n[a] << First.\n[b] << Second.\n"
        "> model: other\n[c] << Third.\n= `[a, b, c]`\n"
    )
    first, second, third = run(source, adapter)
    assert first == "First. after []"
    # The second turn carries the first turn's messages, as Kedi recorded them.
    assert second == "Second. after ['First.', 'First. after []']"
    # Another model is another lane: its transcript starts empty.
    assert third == "Third. after []"


def test_a_transcript_turns_media_is_fetched_by_ref(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    source = (
        f'> import: media\n> history: enabled\n[photo: Image] = `image("{PIXEL}")`\n'
        "[a] << Look at <<photo>>.\n[b] << Fetch the photo.\n= <b>\n"
    )
    fetched = json.loads(run(source, adapter))
    photo = fetched["photo"]["result"]
    assert photo["kind"] == "image" and photo["mediaType"] == "image/png"
    assert photo["data"] == PIXEL.split(",")[1]
    assert "the conversation has no 'nothing'" in fetched["missing"]["error"]["message"]


def test_kedi_checks_the_messages_a_transcript_run_returns(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    for prompt, message in (
        ("Forget it.", "a transcript run must answer with 'messages': this turn's messages"),
        ("Break a part.", "messages[0].parts[0] must be one of text, media, toolCall"),
        ("Name a stranger.", "messages[0].parts[0]: the conversation has no 'stranger'"),
        ("No role.", "messages[0] needs a role of user, assistant or tool, and parts"),
        ("Not a list.", "'messages' must list the turn's messages"),
    ):
        error = failure(f"> history: enabled\n[r] << {prompt}\n= <r>\n", adapter)
        assert message in str(error), prompt


def test_a_harness_keeps_its_session_unless_the_program_needs_kedis_transcript(
    adapters: list,
) -> None:
    adapter = peer("both-histories")
    adapters.append(adapter)
    assert run("> history: enabled\n[r] << Hi.\n= <r>\n", adapter) == "state"
    needs = "> requires: history_replay\n> history: enabled\n[r] << Hi.\n= <r>\n"
    assert run(needs, adapter) == "transcript"


def test_media_get_needs_the_transcript_capability(adapters: list) -> None:
    adapter = peer("sessions")
    adapters.append(adapter)
    with pytest.raises(JsonRpcError, match="media/get needs the transcript capability"):
        adapter._parse_media({"runId": "r", "ref": "photo"})
    with pytest.raises(JsonRpcError, match="media/get needs a string ref"):
        peer_with_transcript = peer("transcript")
        adapters.append(peer_with_transcript)
        peer_with_transcript._parse_media({"runId": "r"})


def test_the_reference_bridge_rebuilds_tool_turns_and_media_from_kedis_transcript(
    adapters: list,
) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="lookup", arguments={"query": "q"})
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        source = (
            f'> import: media\n> history: enabled\n[photo: Image] = `image("{PIXEL}")`\n'
            "@lookup(query: str) -> str:\n  > tool:\n    risk: read_only\n  = `'found ' + query`\n"
            "> use: lookup\n[a] << Look at <<photo>>.\n[b] << Again.\n= `[a, b]`\n"
        )
        assert run(source, adapter) == ["done", "done"]
        exchanges = endpoint.take()
        # The second turn's first request replays the first turn from Kedi's transcript.
        replay = exchanges[2].body["messages"]
        roles = [message["role"] for message in replay]
        assert roles[:5] == ["system", "user", "assistant", "tool", "assistant"]
        image = replay[1]["content"][1]["image_url"]["url"]
        assert image == PIXEL
        assert replay[2]["tool_calls"][0]["function"]["name"] == "lookup"
        assert "found q" in replay[3]["content"]


PROCESSED = """> history:
    enabled: true
    processor: `recent`
[a] << First.
[b] << Second.
[c] << Third.
= `[a, b, c]`
"""


def keep_recent(ctx: Any) -> list[Any]:
    return ctx.keep_recent_groups(1)


def test_kedi_runs_the_programs_history_processor_for_a_bridge(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    assert adapter.capabilities.supports_history_processing
    first, second, third = run(PROCESSED, adapter, recent=keep_recent)
    assert json.loads(first) == ["First."]
    # The processor keeps the last finished exchange and the current turn.
    assert json.loads(second) == ["First.", "First. after []", "Second."]
    # Before the third request it drops the first exchange.
    assert json.loads(third) == [
        "Second.",
        "Second. after ['First.', 'First. after []']",
        "Third.",
    ]


def test_a_processing_run_answers_with_the_whole_conversation(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    source = "> history:\n    enabled: true\n    processor: `recent`\n[r] << Forget it.\n= <r>\n"
    with pytest.raises(Exception, match="a transcript run must answer with 'transcript'"):
        run(source, adapter, recent=keep_recent)


def test_a_failing_history_processor_stops_the_bridge_run(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)

    def broken(ctx: Any) -> list[Any]:
        raise RuntimeError("the processor is broken")

    source = "> history:\n    enabled: true\n    processor: `recent`\n[r] << Hi.\n= <r>\n"
    with pytest.raises(Exception) as caught:
        run(source, adapter, recent=broken)
    assert "the processor is broken" in str(caught.value) or "the processor is broken" in str(
        caught.value.__cause__
    )


def test_history_process_needs_a_processing_run_and_a_message_list(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)
    refused = json.loads(run("> history: enabled\n[r] << Process anyway.\n= <r>\n", adapter))
    assert refused["error"]["code"] == UNSUPPORTED
    assert "the run processes no history" in refused["error"]["message"]
    with pytest.raises(JsonRpcError, match="history/process needs the list of messages"):
        adapter._parse_history({"runId": "r", "messages": "all"})


def test_the_reference_bridge_sends_the_model_what_the_processor_kept(adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1"))
        adapters.append(adapter)
        assert run(PROCESSED, adapter, recent=keep_recent) == ["done", "done", "done"]
        _, _, third = endpoint.take()
        sent = _strings(third.body["messages"])
        # The first exchange was dropped before the third request.
        assert "First." not in sent and "Second." in sent and "Third." in sent


def test_the_transcript_codec_pairs_tool_calls_and_marks_what_must_stay() -> None:
    from kedi.agent_adapter.adapters.jsonrpc.history import TranscriptHistoryCodec

    codec = TranscriptHistoryCodec()
    messages = [
        {"role": "user", "parts": [{"type": "text", "text": "Look it up."}]},
        {
            "role": "assistant",
            "parts": [{"type": "toolCall", "id": "c1", "name": "f", "arguments": {}}],
        },
        {"role": "tool", "parts": [{"type": "toolResult", "id": "c1", "result": "found"}]},
        {"role": "assistant", "parts": [{"type": "reasoning"}], "providerData": {"sealed": "x"}},
        {"role": "assistant", "parts": [{"type": "text", "text": "Done."}]},
        {"role": "robot"},
    ]
    snapshot = codec.inspect(messages)
    kinds = [entry.kind for entry in snapshot.entries]
    assert kinds == ["user", "tool-call", "tool-result", "reasoning", "assistant", "unknown"]
    call, result, sealed = snapshot.entries[1], snapshot.entries[2], snapshot.entries[3]
    assert call.lifecycle_ids == ("c1",) and result.closes_lifecycle_ids == ("c1",)
    # A backend's sealed reasoning goes back as it was.
    assert sealed.provider_required and snapshot.storage_format == "native"
    copied = codec.clone(messages)
    assert copied == messages and copied[0] is not messages[0]
    with pytest.raises(TypeError, match="all values must be transcript messages"):
        codec.clone(["text"])  # type: ignore[list-item]


def test_a_processor_still_running_when_the_run_ends_is_cancelled(adapters: list) -> None:
    adapter = peer("transcript")
    adapters.append(adapter)

    async def slow(ctx: Any) -> list[Any]:
        await asyncio.sleep(30)
        raise AssertionError("the processor should have been cancelled")

    source = (
        "> history:\n    enabled: true\n    processor: `recent`\n[r] << Process early.\n= <r>\n"
    )
    assert run(source, adapter, recent=slow) == "answered early"
    wait_for_stderr(adapter, f"process reply {CANCELLED}")


def test_a_test_model_answers_the_reference_bridges_model_requests(adapters: list) -> None:
    from kedi.agent_adapter.adapters.jsonrpc import ModelCall

    calls: list[ModelCall] = []

    async def script(call: ModelCall) -> Any:
        calls.append(call)
        if not any(message["role"] == "tool" for message in call.messages):
            return {
                "parts": [
                    {"type": "toolCall", "id": "t1", "name": "lookup", "arguments": {"query": "q"}}
                ]
            }
        return "It is found-q."

    # No model endpoint at all: the bridge sends every request to Kedi.
    adapter = JsonRpcAdapter(reference_bridge_command("http://127.0.0.1:9/v1"), test_model=script)
    adapters.append(adapter)
    with adapter.tool_scope((lookup_tool(),)):
        assert asyncio.run(adapter.invoke(prompt="Look it up.")) == "It is found-q."
    first, second = calls
    # The script sees what the bridge would send its model, in Kedi's schema.
    assert [tool["name"] for tool in first.tools] == ["lookup"]
    assert first.messages[-1] == {
        "role": "user",
        "parts": [{"type": "text", "text": "Look it up."}],
    }
    assert [message["role"] for message in second.messages] == ["user", "assistant", "tool"]
    assert "found-q" in json.dumps(second.messages[-1])


def test_a_test_model_gives_structured_answers_and_sees_the_output_schema(adapters: list) -> None:
    from kedi.agent_adapter.adapters.jsonrpc import ModelCall

    seen: list[ModelCall] = []

    def script(call: ModelCall) -> str:
        seen.append(call)
        return json.dumps({"capital": "Ankara"})

    adapter = JsonRpcAdapter(reference_bridge_command("http://127.0.0.1:9/v1"), test_model=script)
    adapters.append(adapter)
    assert run(">> The capital of Turkey is [capital: str].\n= <capital>\n", adapter) == "Ankara"
    (call,) = seen
    assert call.output is not None and "capital" in json.dumps(call.output["schema"])
    assert call.system is not None


def test_a_bridge_without_a_test_model_refuses_one(adapters: list) -> None:
    with pytest.raises(JsonRpcError, match="bridge 'peer' has no test model"):
        peer("default", test_model=lambda call: "hi")
    plain = peer("default")
    adapters.append(plain)
    with pytest.raises(JsonRpcError, match="model/complete: Kedi asked for no test model"):
        plain._parse_model_call({"runId": "r", "messages": []})


def test_a_test_models_answer_is_checked_and_its_failure_stops_the_run(adapters: list) -> None:
    adapter = peer("test-model", test_model=lambda call: 42)
    adapters.append(adapter)
    with pytest.raises(JsonRpcError, match="model/complete needs lists of messages and tools"):
        adapter._parse_model_call({"runId": "r", "messages": "hi"})
    with pytest.raises(Exception, match="a test model returns the assistant's text"):
        adapter.invoke_sync(prompt="Ask.")
    checked = peer("test-model", test_model=lambda call: {"parts": [{"type": "hologram"}]})
    adapters.append(checked)
    with pytest.raises(Exception, match=r"parts\[0\] must be one of"):
        checked.invoke_sync(prompt="Ask.")


def test_a_test_model_still_running_when_the_run_ends_is_cancelled(adapters: list) -> None:
    async def slow(call: Any) -> str:
        await asyncio.sleep(30)
        raise AssertionError("the test model should have been cancelled")

    adapter = peer("test-model", test_model=slow)
    adapters.append(adapter)
    assert adapter.invoke_sync(prompt="Answer early.") == "answered early"
    wait_for_stderr(adapter, f"model reply {CANCELLED}")


def test_an_interrupt_stops_a_test_model_request_the_bridge_waits_on(adapters: list) -> None:
    started = threading.Event()

    async def slow(call: Any) -> str:
        started.set()
        await asyncio.sleep(30)
        raise AssertionError("the interrupt should have stopped the request")

    adapter = JsonRpcAdapter(reference_bridge_command("http://127.0.0.1:9/v1"), test_model=slow)
    adapters.append(adapter)
    control = LiveInput("child")
    control.configure(input=True, interrupt=True)

    async def attempt() -> None:
        with live_input_scope(control):
            running = asyncio.ensure_future(adapter.invoke(prompt="Take your time."))
            await asyncio.to_thread(started.wait, 10)
            control.admit("Stop.", sender_id="parent", interrupt=True)
            await running

    # In test mode the bridge waits on model/complete, not a fetch; the interrupt still stops it.
    with pytest.raises(RunInterrupted):
        asyncio.run(attempt())


HANDSHAKES = json.loads((Path(__file__).parent / "fixtures/jsonrpc_handshakes.json").read_text())
"""kedi-lang/json-rpc's fixtures/handshake/v1.json: the handshakes Kedi refuses, and those it takes."""


@pytest.mark.parametrize("case", HANDSHAKES["cases"], ids=lambda case: case["name"])
def test_kedi_refuses_the_handshakes_the_protocol_fixtures_name(case: dict[str, Any]) -> None:
    command = [sys.executable, str(Path(__file__).parent / "fixtures/jsonrpc_hello.py")]
    if case["kediSays"] is None:
        JsonRpcAdapter([*command, json.dumps(case["handshake"])]).close()
        return
    with pytest.raises(JsonRpcError, match=re.escape(case["kediSays"])):
        JsonRpcAdapter([*command, json.dumps(case["handshake"])])


def test_a_model_request_kedi_answers_with_an_error_falls_back_in_test_mode() -> None:
    # A model/complete error is the model failing, as a failed fetch is: the bridge tries the
    # next model. (Kedi's own test model stops the run when it raises; this drives the bridge
    # over the wire, as kedi-lang/json-rpc's model-fallback fixture does.)
    command = reference_bridge_command("http://127.0.0.1:9/v1")
    bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert bridge.stdin is not None and bridge.stdout is not None

    def send(message: dict[str, Any]) -> None:
        bridge.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        bridge.stdin.flush()

    try:
        send(
            {
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": 1, "client": {"name": "test"}, "testModel": True},
            }
        )
        assert "result" in json.loads(bridge.stdout.readline())
        run_params = {
            "runId": "r",
            "callKind": "invoke",
            "promptKind": "raw",
            "model": "primary",
            "fallbackModels": ["backup"],
            "system": None,
            "instructionParts": [],
            "user": "Hi.",
            "userContent": None,
            "output": None,
            "tools": [],
            "continuation": None,
            "session": None,
            "subagent": None,
        }
        send({"id": 2, "method": "run", "params": run_params})
        models = []
        while True:
            message = json.loads(bridge.stdout.readline())
            if message.get("method") == "budget/admit":
                send({"id": message["id"], "result": {"admitted": True}})
            elif message.get("method") == "model/complete":
                models.append(message["params"]["model"])
                if message["params"]["model"] == "primary":
                    send(
                        {
                            "id": message["id"],
                            "error": {"code": -32002, "message": "primary is down"},
                        }
                    )
                else:
                    reply = {
                        "role": "assistant",
                        "parts": [{"type": "text", "text": "from backup"}],
                    }
                    send({"id": message["id"], "result": {"message": reply}})
            elif message.get("id") == 2:
                assert message.get("result", {}).get("text") == "from backup", message
                break
        assert models == ["primary", "backup"]
    finally:
        bridge.stdin.close()
        bridge.wait(timeout=10)
