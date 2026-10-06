from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import signal
import sys
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi.agent_adapter import ACPAdapter as ExportedACPAdapter
from kedi.agent_adapter.adapters import ACPAdapter
from kedi.agent_adapter.adapters.acp import (
    ACPClient,
    ACPDisconnected,
    ACPProtocolError,
    _SessionState,
)
from kedi.agent_adapter.events import AgentMessageEvent, AgentRunStateEvent, observe_agent_events
from kedi.agent_adapter.registry import adapter_capabilities, build_adapter
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import ACP_AGENT_SETTING_KEYS, AgentProfile
from kedi.agent_profile.errors import AgentProfileOverrideUnsupported, AgentToolOverrideUnsupported
from kedi.app import _build_runtime_adapter

FAKE_ACP_AGENT = Path(__file__).parent / "files" / "fake_acp_agent.py"


class FakeACPClient:
    def __init__(self) -> None:
        self.sync_calls: list[dict[str, Any]] = []
        self.async_calls: list[dict[str, Any]] = []
        self.closed = False

    def run_prompt_sync(self, *, prompt: str, cwd: str, timeout: float) -> str:
        self.sync_calls.append({"prompt": prompt, "cwd": cwd, "timeout": timeout})
        return f"sync:{prompt}:{cwd}:{timeout:g}"

    async def run_prompt(self, *, prompt: str, cwd: str, timeout: float) -> str:
        self.async_calls.append({"prompt": prompt, "cwd": cwd, "timeout": timeout})
        return f"async:{prompt}:{cwd}:{timeout:g}"

    def close(self) -> None:
        self.closed = True


class BrokenStdin:
    def close(self) -> None:
        pass

    def write(self, line: str) -> None:
        raise BrokenPipeError

    def flush(self) -> None:
        raise AssertionError("flush should not be called after write fails")


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 1234
        self.stdin: Any = BrokenStdin()
        self.stdout: Any = io.StringIO()
        self.stderr: Any = io.StringIO()
        self.terminated = False
        self.killed = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> None:
        raise TimeoutError

    def kill(self) -> None:
        self.killed = True


def _messages(log_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_acp_adapter_metadata_public_exports_and_registry() -> None:
    assert ACPAdapter is ExportedACPAdapter
    assert ACPAdapter.kind == "agent-harness"
    assert ACPAdapter.shortname == "acp"
    assert not ACPAdapter.capabilities.supports("structured_output")
    assert not ACPAdapter.capabilities.supports("tool_registration")
    assert not ACPAdapter.capabilities.supports("mcp")
    assert ACPAdapter.capabilities.supports("profile_override")
    assert ACPAdapter.capabilities.settings == ACP_AGENT_SETTING_KEYS

    assert adapter_capabilities("acp") == ACPAdapter.capabilities
    assert isinstance(build_adapter("acp"), ACPAdapter)
    assert isinstance(_build_runtime_adapter("acp", None, acp_command="fake acp"), ACPAdapter)


def test_acp_adapter_invoke_uses_profile_command_settings_and_env() -> None:
    client = FakeACPClient()
    adapter = ACPAdapter(
        client=cast(Any, client),
        model_settings={"timeout": 9},
    )
    profile = AgentProfile(
        agent_command="profile acp",
        settings={
            "cwd": "/tmp/kedi-acp",
            "timeout": 3,
            "env": {"ACP_TEST": "ok"},
        },
    )

    with adapter.profile_override(profile):
        assert adapter.invoke_sync(prompt="hello") == "sync:hello:/tmp/kedi-acp:3"
        assert asyncio.run(adapter.invoke(prompt="world")) == "async:world:/tmp/kedi-acp:3"

    assert client.sync_calls == [{"prompt": "hello", "cwd": "/tmp/kedi-acp", "timeout": 3.0}]
    assert client.async_calls == [{"prompt": "world", "cwd": "/tmp/kedi-acp", "timeout": 3.0}]
    adapter.close()
    assert client.closed


def test_acp_adapter_default_model_and_async_close() -> None:
    client = FakeACPClient()
    adapter = ACPAdapter(model="acp-model", client=cast(Any, client))

    assert adapter.default_model() == "acp-model"
    asyncio.run(adapter.aclose())
    assert client.closed


def test_acp_adapter_does_not_resolve_command_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_ACP_AGENT_COMMAND", "env-acp --stdio")
    adapter = ACPAdapter()

    with pytest.raises(RuntimeError, match="requires an explicit command"):
        adapter.invoke_sync(prompt="env")


def test_acp_adapter_accepts_pathlike_cwd(tmp_path: Path) -> None:
    client = FakeACPClient()
    adapter = ACPAdapter(client=cast(Any, client))

    assert adapter.invoke_sync(prompt="path", cwd=tmp_path) == f"sync:path:{tmp_path}:30"


def test_acp_adapter_uses_real_client_from_constructor_command(tmp_path: Path) -> None:
    log_path = tmp_path / "acp.jsonl"
    adapter = ACPAdapter(
        command=[sys.executable, str(FAKE_ACP_AGENT)],
        model_settings={"env": {"KEDI_ACP_FAKE_LOG": str(log_path)}},
    )

    try:
        assert adapter.invoke_sync(prompt="first", cwd=str(tmp_path)) == "echo:first"
        assert adapter.invoke_sync(prompt="second", cwd=str(tmp_path)) == "echo:second"
    finally:
        adapter.close()

    methods = [message["method"] for message in _messages(log_path)]
    process_cwd = next(
        message["cwd"] for message in _messages(log_path) if message["method"] == "process/cwd"
    )
    assert process_cwd == str(tmp_path)
    assert methods.count("initialize") == 1
    assert methods.count("session/prompt") == 2


def test_acp_adapter_buffers_chunks_and_emits_one_final_message(
    tmp_path: Path,
) -> None:
    adapter = ACPAdapter(command=[sys.executable, str(FAKE_ACP_AGENT)])
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    try:
        with observe_agent_events(streamed.append):
            assert adapter.invoke_sync(prompt="chunked", cwd=tmp_path) == "echo:chunked"
    finally:
        adapter.close()

    messages = [event for event in streamed if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [("final", "echo:chunked")]


def test_acp_adapter_async_invoke_emits_final_message(
    tmp_path: Path,
) -> None:
    adapter = ACPAdapter(command=[sys.executable, str(FAKE_ACP_AGENT)])
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    async def scenario() -> str:
        with observe_agent_events(streamed.append):
            return await adapter.invoke(prompt="async-event", cwd=tmp_path)

    try:
        assert asyncio.run(scenario()) == "echo:async-event"
    finally:
        adapter.close()

    messages = [event for event in streamed if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [("final", "echo:async-event")]


@pytest.mark.parametrize("as_string", [False, True], ids=["argv", "quoted-string"])
def test_acp_adapter_uses_explicit_profile_command(tmp_path: Path, as_string: bool) -> None:
    profile_log_path = tmp_path / "profile-acp.jsonl"
    adapter = ACPAdapter(
        model_settings={
            "env": {"KEDI_ACP_FAKE_LOG": str(profile_log_path)},
        }
    )
    command = (sys.executable, str(FAKE_ACP_AGENT))
    profile = AgentProfile(agent_command=shlex.join(command) if as_string else command)

    try:
        with adapter.profile_override(profile):
            assert adapter.invoke_sync(prompt="profile", cwd=str(tmp_path)) == "echo:profile"
    finally:
        adapter.close()

    assert profile_log_path.exists()


def test_acp_adapter_uses_constructor_command_when_settings_omit_command(tmp_path: Path) -> None:
    log_path = tmp_path / "constructor-acp.jsonl"
    adapter = ACPAdapter(
        command=[sys.executable, str(FAKE_ACP_AGENT)],
        model_settings={"env": {"KEDI_ACP_FAKE_LOG": str(log_path)}},
    )

    try:
        assert adapter.invoke_sync(prompt="constructor", cwd=str(tmp_path)) == "echo:constructor"
    finally:
        adapter.close()

    assert log_path.exists()


def test_acp_adapter_requires_command_when_no_client_or_setting() -> None:
    adapter = ACPAdapter()

    with pytest.raises(RuntimeError, match="requires an explicit command"):
        adapter.invoke_sync(prompt="missing")


def test_acp_adapter_rejects_unsupported_settings_and_invalid_values() -> None:
    with pytest.raises(TypeError, match="Unsupported ACP setting"):
        ACPAdapter(not_a_setting=True)

    adapter = ACPAdapter(client=cast(Any, FakeACPClient()))
    with pytest.raises(TypeError, match="cwd"):
        adapter.invoke_sync(prompt="bad", cwd=1)
    with pytest.raises(TypeError, match="timeout"):
        adapter.invoke_sync(prompt="bad", timeout=0)
    with pytest.raises(TypeError, match="env"):
        adapter.invoke_sync(prompt="bad", env={"X": 1})
    with pytest.raises(TypeError, match="env"):
        adapter.invoke_sync(prompt="bad", env=object())
    with pytest.raises(TypeError, match="command"):
        ACPAdapter(command=cast(Any, [object()]))
    with pytest.raises(TypeError, match="command"):
        ACPAdapter(command=cast(Any, object()))
    assert ACPAdapter(command="")._command is None


def test_acp_adapter_structured_output_is_explicitly_unsupported() -> None:
    adapter = ACPAdapter(client=cast(Any, FakeACPClient()))

    with pytest.raises(NotImplementedError, match="structured template output"):
        adapter.produce_sync("template", output_schema={"answer": str})
    with pytest.raises(NotImplementedError, match="structured template output"):
        asyncio.run(adapter.produce("template", output_schema={"answer": str}))


def test_acp_adapter_rejects_registered_tools() -> None:
    adapter = ACPAdapter(client=cast(Any, FakeACPClient()))

    with pytest.raises(AgentToolOverrideUnsupported):
        adapter.register_tool(ToolSpec(fn=lambda: "ok", name="noop"))
    with pytest.raises(AgentToolOverrideUnsupported):
        adapter.tool_scope((ToolSpec(fn=lambda: "scoped", name="scoped"),))
    with adapter.tool_scope():
        assert adapter.scoped_tools == ()


@pytest.mark.parametrize("profile", [AgentProfile(model="unused"), AgentProfile(effort="high")])
def test_acp_adapter_rejects_unsupported_model_configuration(profile):
    client = FakeACPClient()
    adapter = ACPAdapter(client=cast(Any, client))
    with pytest.raises(AgentProfileOverrideUnsupported, match="model or effort"):
        with adapter.profile_override(profile):
            pytest.fail("Unsupported profile was accepted")
    assert not client.sync_calls and not client.async_calls


@pytest.mark.parametrize("field", ["mcp_servers", "tools", "tool_bindings"])
def test_acp_profile_rejects_local_tool_projection(field):
    adapter = ACPAdapter(client=cast(Any, FakeACPClient()))
    from kedi.agent_profile import McpServerSpec

    values = {
        "mcp_servers": (McpServerSpec(command="unused"),),
        "tools": ("noop",),
        "tool_bindings": {"noop": ToolSpec(fn=lambda: "ok", name="noop")},
    }
    profile = AgentProfile(**{field: values[field]})
    error = (
        AgentProfileOverrideUnsupported if field == "mcp_servers" else AgentToolOverrideUnsupported
    )
    with pytest.raises(error), adapter.profile_override(profile):
        pytest.fail("Unsupported tool projection was accepted")


@pytest.mark.parametrize(
    "reason", [None, "max_tokens", "max_turn_requests", "refusal", "cancelled", "unknown"]
)
def test_acp_partial_answer_is_not_reported_as_completed(monkeypatch, reason):
    client = ACPClient(("fake",))
    monkeypatch.setattr(client, "_new_session", lambda cwd: "s")

    def request(method, params, **kwargs):
        if method == "session/prompt":
            client._sessions["s"].messages.append("partial")
            return {"stopReason": reason}
        return {}

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(ACPProtocolError, match="did not complete"):
        client.run_prompt_sync(prompt="test", cwd="/tmp", timeout=1)
    assert not client._sessions


def test_acp_client_runs_prompt_through_stdio_agent(tmp_path: Path) -> None:
    log_path = tmp_path / "acp.jsonl"
    client = ACPClient(
        (sys.executable, str(FAKE_ACP_AGENT)),
        env={"KEDI_ACP_FAKE_LOG": str(log_path)},
        request_timeout=1,
    )

    try:
        assert client.run_prompt_sync(prompt="hello", cwd=str(tmp_path), timeout=1) == "echo:hello"
    finally:
        client.close()

    methods = [
        message["method"] for message in _messages(log_path) if message["method"] != "process/cwd"
    ]
    assert methods == ["initialize", "session/new", "session/prompt", "session/close"]


def test_acp_client_async_prompt_and_close(tmp_path: Path) -> None:
    client = ACPClient((sys.executable, str(FAKE_ACP_AGENT)), request_timeout=1)

    assert asyncio.run(client.run_prompt(prompt="async", cwd=str(tmp_path), timeout=1)) == (
        "echo:async"
    )
    asyncio.run(client.aclose())
    with pytest.raises(ACPDisconnected, match="closed"):
        client.request("initialize")


def test_acp_client_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="empty"):
        ACPClient(())


def test_acp_client_protocol_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ACPClient(("fake",))

    monkeypatch.setattr(client, "request", lambda *args, **kwargs: None)
    with pytest.raises(ACPProtocolError, match="initialize"):
        client._initialize()
    with pytest.raises(ACPProtocolError, match="session/new"):
        client._new_session("/tmp")

    assert "disconnected" in client._disconnect_message()
    client._stderr_tail.append("boom")
    assert "boom" in client._disconnect_message()


def test_acp_client_write_and_reader_edge_paths() -> None:
    client = ACPClient(("fake",))

    with pytest.raises(ACPDisconnected):
        client._write({"method": "x"})
    client._read_stdout()
    client._read_stderr()

    process = FakeProcess()
    client._process = cast(Any, process)
    with pytest.raises(ACPDisconnected):
        client._write({"method": "x"})
    process.stdout = ["not-json\n"]
    client._read_stdout()
    process.stdout = ["[]\n"]
    client._read_stdout()
    process.stderr = ["one\n", "two\n"]
    client._read_stderr()
    assert list(client._stderr_tail)[-2:] == ["one", "two"]


def test_acp_client_close_without_process_is_safe() -> None:
    client = ACPClient(("fake",))

    client.close()
    client.close()

    assert client._closed


@pytest.mark.parametrize("running", [False, True])
def test_acp_close_windows_process_without_pipes(monkeypatch, running):
    from kedi.agent_adapter.adapters import acp

    client = ACPClient(("fake",))
    process = FakeProcess()
    process.stdin = process.stdout = process.stderr = None
    monkeypatch.setattr(process, "poll", lambda: None if running else 0)
    monkeypatch.setattr(acp, "os", SimpleNamespace(name="nt"))
    client._process = process
    client._reader = current_thread()
    client.close()
    assert process.terminated is running
    assert process.killed is running


@pytest.mark.parametrize("method", ["initialize", "session/prompt"])
def test_acp_expired_deadline_without_session_does_not_cancel(monkeypatch, method):
    from kedi.agent_adapter.adapters import acp

    client = ACPClient(("fake",))
    messages = []
    clock = iter((0, 2))
    monkeypatch.setattr(client, "_ensure_started", lambda **kwargs: None)
    monkeypatch.setattr(client, "_write", messages.append)
    monkeypatch.setattr(acp, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        client.request(method, timeout=1)
    assert [m["method"] for m in messages] == [method]
    assert not client._pending


@pytest.mark.parametrize("drain_error", [FutureTimeoutError, ACPDisconnected])
def test_acp_cancel_closes_transport_only_when_handshake_stalls(monkeypatch, drain_error):
    from kedi.agent_adapter.adapters import acp

    class FailedHandshake(Future):
        def result(self, timeout=None):
            assert timeout == 5
            raise drain_error("handshake failed")

    client = ACPClient(("fake",))
    cancelled = Event()
    cancelled.set()
    messages = []
    monkeypatch.setattr(client, "_ensure_started", lambda **kwargs: None)
    monkeypatch.setattr(client, "_write", messages.append)
    monkeypatch.setattr(acp, "Future", FailedHandshake)
    with pytest.raises(acp.FutureCancelledError):
        client.request("session/prompt", {"sessionId": "owned"}, _cancelled=cancelled)
    assert [m["method"] for m in messages] == ["session/prompt", "session/cancel"]
    assert messages[-1]["params"] == {"sessionId": "owned"}
    assert client._closed is (drain_error is FutureTimeoutError)
    assert not client._pending


@pytest.mark.asyncio
async def test_acp_repeated_cancel_drains_worker_failure(monkeypatch):
    client = ACPClient(("fake",))
    entered, release = Event(), Event()

    def prompt(**kwargs):
        entered.set()
        assert release.wait(5)
        assert kwargs["_cancelled"].is_set()
        raise ACPDisconnected("worker stopped")

    monkeypatch.setattr(client, "run_prompt_sync", prompt)
    task = asyncio.create_task(client.run_prompt(prompt="wait", cwd="/tmp", timeout=1))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)


def test_acp_client_close_fails_pending_requests() -> None:
    client = ACPClient(("fake",))
    pending: Future[dict[str, Any] | None] = Future()
    with client._state_lock:
        client._pending[1] = pending

    client.close()

    with pytest.raises(ACPDisconnected, match="closed"):
        pending.result()


def test_acp_client_write_failure_cleans_pending_request() -> None:
    client = ACPClient(("fake",))
    client._ensure_started = lambda **_kwargs: None  # type: ignore[method-assign]

    def fail_write(_payload: object) -> None:
        raise ACPDisconnected("broken pipe")

    client._write = fail_write  # type: ignore[method-assign]

    with pytest.raises(ACPDisconnected, match="broken pipe"):
        client.request("session/prompt")

    assert client._pending == {}


def test_acp_client_rejects_duplicate_active_session_id() -> None:
    client = ACPClient(("fake",))
    first = _SessionState()
    client._register_session("same", first)

    with pytest.raises(ACPProtocolError, match="reused active session ID"):
        client._register_session("same", _SessionState())

    assert client._sessions == {"same": first}


def test_acp_client_preserves_a_replaced_session_during_prompt_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ACPClient(("fake",))
    replacement = _SessionState()

    monkeypatch.setattr(client, "_new_session", lambda _cwd: "same")

    def request(method: str, _params: object, *, timeout: float) -> dict[str, object]:
        del timeout
        if method == "session/prompt":
            client._sessions["same"] = replacement
        return {"stopReason": "end_turn"}

    monkeypatch.setattr(client, "request", request)

    assert client.run_prompt_sync(prompt="hello", cwd="/tmp", timeout=1) == ""
    assert client._sessions["same"] is replacement


def test_acp_client_can_start_without_initializing(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess()
    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.acp.subprocess.Popen", lambda *_, **__: process
    )
    client = ACPClient(("fake",))

    client._ensure_started(initialize=False)

    assert client._process is process
    assert not client._initialized


def test_acp_client_running_uninitialized_process_initializes() -> None:
    client = ACPClient(("fake",))
    process = FakeProcess()
    client._process = cast(Any, process)
    called: list[bool] = []

    def initialize() -> None:
        called.append(True)
        client._initialized = True

    client._initialize = cast(Any, initialize)
    client._ensure_started()

    assert called == [True]


def test_acp_client_close_terminates_and_kills_running_process(monkeypatch) -> None:
    client = ACPClient(("fake",))
    process = FakeProcess()
    client._process = cast(Any, process)

    def kill_group(pid, sig):
        assert pid == process.pid
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    if os.name == "posix":
        monkeypatch.setattr(os, "killpg", kill_group)

    client.close()

    assert process.terminated
    assert process.killed


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_acp_cancellation_notifies_and_drains_only_owned_session(tmp_path, cancel):
    log_path = tmp_path / "acp.jsonl"
    client = ACPClient(
        (sys.executable, str(FAKE_ACP_AGENT)),
        env={"KEDI_ACP_FAKE_LOG": str(log_path)},
        request_timeout=2,
    )

    async def scenario():
        blocked = asyncio.create_task(
            client.run_prompt(prompt="wait", cwd=str(tmp_path), timeout=2 if cancel else 0.1)
        )

        async def wait_for_message():
            while not any(state.messages for state in client._sessions.values()):
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_message(), 3)
        # A second call uses its own session even while the first is active.
        assert (
            await client.run_prompt(prompt="sibling", cwd=str(tmp_path), timeout=1)
            == "echo:sibling"
        )
        if cancel:
            blocked.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await blocked
        assert not client._sessions
        assert not client._pending
        assert await client.run_prompt(prompt="after", cwd=str(tmp_path), timeout=1) == "echo:after"

    try:
        asyncio.run(scenario())
    finally:
        client.close()
    messages = _messages(log_path)
    cancellations = [m for m in messages if m.get("method") == "session/cancel"]
    assert len(cancellations) == 1
    assert "id" not in cancellations[0]
    assert not client._reader.is_alive()
    assert not client._stderr_reader.is_alive()
    assert client._process.stdout.closed
    assert client._process.stderr.closed


def test_acp_inbound_request_cannot_resolve_outbound_request(monkeypatch):
    client = ACPClient(("fake",))
    pending = Future()
    client._pending[1] = pending
    replies = []
    monkeypatch.setattr(client, "_write", replies.append)
    client._handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}}
    )
    assert not pending.done()
    assert replies == [{"jsonrpc": "2.0", "id": 1, "result": {"outcome": {"outcome": "cancelled"}}}]
    client._handle_message(
        {"jsonrpc": "2.0", "id": "remote", "method": "fs/read_text_file", "params": {}}
    )
    assert replies[-1]["error"]["code"] == -32601
    assert not pending.done()


def test_acp_cancel_before_session_setup_does_not_submit_prompt(monkeypatch):
    from threading import Event

    client = ACPClient(("fake",))
    entered, release = Event(), Event()
    calls = []

    def new_session(cwd):
        entered.set()
        assert release.wait(2)
        return "new"

    monkeypatch.setattr(client, "_new_session", new_session)
    monkeypatch.setattr(client, "request", lambda method, *args, **kwargs: calls.append(method))

    async def scenario():
        task = asyncio.create_task(client.run_prompt(prompt="unused", cwd="/tmp", timeout=1))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not client._sessions

    asyncio.run(scenario())
    assert calls == ["session/close"]


def test_acp_client_message_routing_edge_paths() -> None:
    client = ACPClient(("fake",))

    client._handle_message({"id": "not-int", "result": {}})
    client._handle_message({"id": 999, "result": {}})

    done: Future[dict[str, Any] | None] = Future()
    done.set_result({})
    pending: Future[dict[str, Any] | None] = Future()
    with client._state_lock:
        client._pending[1] = done
        client._pending[2] = pending
    client._handle_message({"id": 1, "result": {}})
    client._handle_message({"id": 2, "error": {"message": "bad"}})
    with pytest.raises(ACPProtocolError, match="bad"):
        pending.result()

    client._handle_message({"method": "other", "params": {}})
    client._handle_session_update({})
    client._handle_session_update({"sessionId": "s", "update": "bad"})
    client._handle_session_update({"sessionId": "s", "update": {"sessionUpdate": "other"}})
    client._handle_session_update(
        {"sessionId": "s", "update": {"sessionUpdate": "agent_message_chunk"}}
    )
    client._handle_session_update(
        {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "image"},
            },
        }
    )
    client._handle_session_update(
        {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": 1},
            },
        }
    )
    client._handle_session_update(
        {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "orphan"},
            },
        }
    )
    client._sessions["s"] = session_state = _SessionState()
    client._handle_session_update(
        {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "chunk"},
            },
        }
    )
    assert session_state.messages == ["chunk"]

    future: Future[dict[str, Any] | None] = Future()
    done_future: Future[dict[str, Any] | None] = Future()
    done_future.set_result({})
    with client._state_lock:
        client._pending[3] = future
        client._pending[4] = done_future
    client._fail_all(RuntimeError("fail"))
    with pytest.raises(RuntimeError, match="fail"):
        future.result()
