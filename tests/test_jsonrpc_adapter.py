"""JsonRpcAdapter (#197): Kedi's adapter contract over JSON-RPC to a bridge process."""

from __future__ import annotations

import asyncio
import contextvars
import json
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
from kedi.agent_adapter.conformance import check_adapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.lang import compile_program, parse_program

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
    assert usage == [InvocationUsage(requests=2, input_tokens=5, total_tokens=5)]
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
