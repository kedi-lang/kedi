from __future__ import annotations

import asyncio
import contextvars
import json
import subprocess
import sys
from concurrent.futures import Future
from contextvars import ContextVar
from datetime import date, datetime, time, timedelta
from pathlib import Path
from re import Pattern
from threading import Thread, current_thread
from typing import Any, cast

import pytest
from pydantic import BaseModel, EmailStr, FileUrl, HttpUrl

from kedi.agent_adapter import CodexAdapter as ExportedCodexAdapter
from kedi.agent_adapter.adapters import CodexAdapter
from kedi.agent_adapter.adapters import _codex_app_server as codex_server_module
from kedi.agent_adapter.adapters import codex as codex_adapter_module
from kedi.agent_adapter.adapters._codex_app_server import (
    CodexAppServerClient,
    CodexAppServerDisconnected,
    CodexAppServerError,
    CodexAppServerProtocolError,
    CodexTurnResult,
)
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.tool_override import ToolSpec, arguments_target_secret_file
from kedi.agent_profile import CODEX_AGENT_SETTING_KEYS, AgentProfile
from kedi.agent_profile.mcp import McpServerSpec

FAKE_APP_SERVER = Path(__file__).parent / "files" / "fake_codex_app_server.py"


class Answer(BaseModel):
    answer: str


class ToolAnswer(BaseModel):
    answer: str
    success: bool


class DumpedToolResult(BaseModel):
    value: int


class FormattedOutput(BaseModel):
    created_at: datetime
    elapsed: timedelta
    day: date
    clock: time
    email: EmailStr


class UnsupportedFormattedOutput(BaseModel):
    matcher: Pattern[str]
    site: HttpUrl
    file: FileUrl


def _adapter(
    tmp_path: Path,
    *,
    request_timeout: float = 1,
    turn_timeout: float = 1,
    extra_env: dict[str, str] | None = None,
    **kwargs: Any,
) -> tuple[CodexAdapter, Path]:
    log_path = tmp_path / "app-server.jsonl"
    env = {"KEDI_CODEX_FAKE_LOG": str(log_path), **(extra_env or {})}
    adapter = CodexAdapter(
        command=(sys.executable, str(FAKE_APP_SERVER)),
        env=env,
        request_timeout=request_timeout,
        turn_timeout=turn_timeout,
        **kwargs,
    )
    return adapter, log_path


def _messages(log_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def _method(messages: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("method") == method]


def test_codex_adapter_metadata_and_public_exports() -> None:
    assert CodexAdapter is ExportedCodexAdapter
    assert CodexAdapter.kind == "agent-harness"
    assert CodexAdapter.shortname == "codex"
    assert CodexAdapter.capabilities.supports("structured_output")
    assert CodexAdapter.capabilities.supports("tool_registration")
    assert not CodexAdapter.capabilities.supports("mcp")
    assert CodexAdapter.capabilities.settings == CODEX_AGENT_SETTING_KEYS


def test_codex_adapter_sync_run_reuses_connection_and_maps_profile(
    tmp_path: Path,
) -> None:
    adapter, log_path = _adapter(
        tmp_path,
        model="gpt-base",
        model_settings={"sandbox": "readOnly", "ignored": True},
        approval_policy="onRequest",
    )
    assert adapter.default_model() == "gpt-base"

    profile = AgentProfile(
        model="gpt-profile",
        effort="max",
        system="Profile instructions",
        settings={
            "cwd": str(tmp_path),
            "model_provider": "openai",
            "personality": "friendly",
            "service_tier": "fast",
            "summary": "concise",
            "config": {"features.web_search": True},
        },
    )
    with adapter.profile_override(profile):
        first = adapter.produce_sync(
            "first",
            output_type=Answer,
            instructions="Runtime instructions",
        )
        assert adapter.active_profile is profile
    second = adapter.produce_sync("second", output_schema={"answer": str})
    assert adapter.active_profile == AgentProfile.empty()
    adapter.close()

    assert first == Answer(answer="first")
    assert second.answer == "second"
    messages = _messages(log_path)
    assert len(_method(messages, "initialize")) == 1
    starts = _method(messages, "thread/start")
    assert len(starts) == 2
    first_params = starts[0]["params"]
    assert first_params == {
        "ephemeral": True,
        "approvalPolicy": "onRequest",
        "sandbox": "read-only",
        "cwd": str(tmp_path),
        "serviceName": "kedi",
        "model": "gpt-profile",
        "developerInstructions": "Runtime instructions",
        "modelProvider": "openai",
        "personality": "friendly",
        "serviceTier": "fast",
        "config": {"features.web_search": True},
    }
    assert "dynamicTools" not in first_params
    turns = _method(messages, "turn/start")
    assert turns[0]["params"]["effort"] == "xhigh"
    assert turns[0]["params"]["summary"] == "concise"
    assert turns[0]["params"]["outputSchema"]["title"] == "Answer"
    assert set(turns[1]["params"]["outputSchema"]["properties"]) == {"answer"}
    assert len(_method(messages, "thread/delete")) == 2


def test_codex_adapter_async_run_and_close(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, model=None)
    result = asyncio.run(adapter.produce("async", output_type=Answer))
    asyncio.run(adapter.aclose())
    assert result == Answer(answer="async")


def test_codex_adapter_invoke_returns_raw_text_without_output_schema(
    tmp_path: Path,
) -> None:
    adapter, log_path = _adapter(tmp_path, model="gpt-base")

    assert adapter.invoke_sync(prompt="raw prompt") == '{"answer": "raw prompt"}'
    assert asyncio.run(adapter.invoke(prompt="raw async")) == '{"answer": "raw async"}'
    adapter.close()

    turns = _method(_messages(log_path), "turn/start")
    assert turns[0]["params"]["input"] == [{"type": "text", "text": "raw prompt"}]
    assert "outputSchema" not in turns[0]["params"]
    assert turns[1]["params"]["input"] == [{"type": "text", "text": "raw async"}]
    assert "outputSchema" not in turns[1]["params"]


def test_codex_adapter_runs_sync_and_async_dynamic_tools(tmp_path: Path) -> None:
    adapter, log_path = _adapter(tmp_path)
    context_value: ContextVar[str] = ContextVar("codex_test_context", default="missing")

    def lookup(query: str, limit: int = 2) -> dict[str, Any]:
        return {"query": query, "limit": limit, "context": context_value.get()}

    async def async_lookup(query: str) -> str:
        await asyncio.sleep(0)
        return f"async:{query}:{context_value.get()}"

    context_value.set("visible")
    adapter.register_tool(
        ToolSpec(
            fn=lookup,
            name="lookup",
            description="Look up records.",
            risk="read_only",
        )
    )
    with adapter.tool_scope(
        (
            ToolSpec(
                fn=async_lookup,
                name="async_lookup",
                description="Look up asynchronously.",
                json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                risk="read_only",
            ),
        )
    ):
        async_result = adapter.produce_sync(
            'tool:async_lookup:{"query":"docs"}', output_type=ToolAnswer
        )
        assert adapter.scoped_tools[0].name == "async_lookup"
    sync_result = None
    with adapter.tool_scope((ToolSpec(fn=lookup, name="lookup", risk="read_only"),)):
        sync_result = adapter.produce_sync(
            'tool:lookup:{"query":"items","limit":3}', output_type=ToolAnswer
        )
    adapter.close()

    assert async_result == ToolAnswer(answer="async:docs:visible", success=True)
    assert sync_result is not None
    assert json.loads(sync_result.answer) == {
        "query": "items",
        "limit": 3,
        "context": "visible",
    }
    messages = _messages(log_path)
    specs = [message["params"]["dynamicTools"][0] for message in _method(messages, "thread/start")]
    starts = _method(messages, "thread/start")
    assert all(start["params"]["sandbox"] == "workspace-write" for start in starts)
    # The App Server must ask Kedi before it can run native actions as well. Kedi then
    # auto-allows the explicitly read-only registered tools above.
    assert all(start["params"]["approvalPolicy"] == "untrusted" for start in starts)
    assert specs[0]["inputSchema"]["required"] == ["query"]
    assert specs[1]["inputSchema"]["properties"]["limit"]["default"] == 2
    tool_responses = [message for message in messages if message.get("id", 0) >= 10_000]
    assert all(message["result"]["success"] for message in tool_responses)


def test_codex_adapter_bridges_native_app_server_approvals(tmp_path: Path) -> None:
    adapter, log_path = _adapter(tmp_path)

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        command = adapter.produce_sync("approval:command-write", output_type=ToolAnswer)
        permission = adapter.produce_sync("approval:permissions", output_type=ToolAnswer)
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        denied = adapter.produce_sync("approval:command-write", output_type=ToolAnswer)
        read_only = adapter.produce_sync("approval:command-read", output_type=ToolAnswer)

    def edit_command(_request: Any) -> ApprovalDecision:
        return ApprovalDecision.edit({"command": "echo replacement"})

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(edit_command))):
        edited = adapter.produce_sync("approval:file", output_type=ToolAnswer)
    adapter.close()

    assert json.loads(command.answer) == {"decision": "accept"}
    assert json.loads(permission.answer) == {
        "permissions": {"network": {"enabled": True}, "fileSystem": None},
        "scope": "turn",
    }
    assert json.loads(denied.answer) == {"decision": "decline"}
    assert json.loads(read_only.answer) == {"decision": "accept"}
    assert json.loads(edited.answer) == {"decision": "decline"}
    starts = _method(_messages(log_path), "thread/start")
    assert all(start["params"]["approvalPolicy"] == "untrusted" for start in starts)
    assert CodexAdapter.capabilities.supports("native_approvals")
    assert not CodexAdapter.capabilities.supports("native_approval_handler")


def test_codex_adapter_keeps_async_tool_and_approval_on_caller_loop(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)

    async def run() -> tuple[ToolAnswer, ToolAnswer]:
        loop = asyncio.get_running_loop()
        tool_gate: asyncio.Future[None] = loop.create_future()
        approval_gate: asyncio.Future[None] = loop.create_future()

        async def lookup(query: str) -> str:
            assert asyncio.get_running_loop() is loop
            await tool_gate
            return f"lookup:{query}"

        async def approve(_request: Any) -> ApprovalDecision:
            assert asyncio.get_running_loop() is loop
            await approval_gate
            return ApprovalDecision.allow()

        loop.call_later(0.01, tool_gate.set_result, None)
        loop.call_later(0.01, approval_gate.set_result, None)
        with adapter.tool_scope((ToolSpec(fn=lookup, name="lookup", risk="read_only"),)):
            tool_result = await adapter.produce(
                'tool:lookup:{"query":"docs"}', output_type=ToolAnswer
            )
        with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))):
            approval_result = await adapter.produce(
                "approval:command-write", output_type=ToolAnswer
            )
        return tool_result, approval_result

    tool_result, approval_result = asyncio.run(run())
    adapter.close()

    assert tool_result == ToolAnswer(answer="lookup:docs", success=True)
    assert json.loads(approval_result.answer) == {"decision": "accept"}


def test_codex_adapter_returns_tool_failures_to_app_server(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)

    def broken(query: str) -> str:
        raise ValueError(f"bad {query}")

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        with adapter.tool_scope((ToolSpec(fn=broken, name="broken"),)):
            result = adapter.produce_sync('tool:broken:{"query":"input"}', output_type=ToolAnswer)
    adapter.close()
    assert result == ToolAnswer(answer="ValueError: bad input", success=False)


def test_codex_adapter_reconnects_only_before_thread_creation(tmp_path: Path) -> None:
    reconnect_file = tmp_path / "disconnected-once"
    adapter, log_path = _adapter(
        tmp_path,
        extra_env={"KEDI_CODEX_FAKE_RECONNECT_FILE": str(reconnect_file)},
    )
    assert adapter.produce_sync("after reconnect", output_type=Answer).answer == "after reconnect"
    adapter.close()

    messages = _messages(log_path)
    assert len(_method(messages, "initialize")) == 2
    assert len(_method(messages, "thread/start")) == 2


@pytest.mark.parametrize(
    ("prompt", "error_type", "message"),
    [
        ("invalid-output", CodexAppServerProtocolError, "invalid structured output"),
        ("no-message", CodexAppServerProtocolError, "without a final agent message"),
        ("failed-turn", CodexAppServerError, "fake turn failed"),
        ("disconnect-turn", CodexAppServerDisconnected, "stdout disconnected"),
        ("malformed-json", CodexAppServerDisconnected, "invalid JSON"),
    ],
)
def test_codex_adapter_protocol_failures(
    tmp_path: Path,
    prompt: str,
    error_type: type[BaseException],
    message: str,
) -> None:
    adapter, _ = _adapter(tmp_path)
    with pytest.raises(error_type, match=message):
        adapter.produce_sync(prompt, output_type=Answer)
    adapter.close()


def test_codex_adapter_timeout_interrupts_turn(tmp_path: Path) -> None:
    adapter, log_path = _adapter(tmp_path, turn_timeout=0.05)
    with pytest.raises(TimeoutError, match="did not complete"):
        adapter.produce_sync("timeout", output_type=Answer)
    adapter.close()
    assert len(_method(_messages(log_path), "turn/interrupt")) == 1


def test_codex_adapter_validation_and_mcp_rejection(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with pytest.raises(ValueError, match="Output schema cannot be empty"):
        CodexAdapter.type_builder({})
    with pytest.raises(ValueError, match="Output spec must be provided"):
        adapter.produce_sync("missing")
    with pytest.raises(TypeError, match="instructions must be a string"):
        adapter.produce_sync("bad", output_type=Answer, instructions=1)
    with pytest.raises(TypeError, match="Unsupported Codex setting.*unknown"):
        adapter.produce_sync("bad", output_type=Answer, unknown=True)
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        adapter.produce_sync("bad", output_type=Answer, timeout=False)
    with pytest.raises(ValueError, match="Codex sandbox"):
        adapter.produce_sync("bad", output_type=Answer, sandbox="bad")
    with adapter.profile_override(
        AgentProfile(mcp_servers=(McpServerSpec(transport="http", url="http://mcp"),))
    ):
        with pytest.raises(NotImplementedError, match="does not yet support"):
            adapter.produce_sync("bad", output_type=Answer)
    adapter.close()

    with pytest.raises(TypeError, match="Unsupported Codex setting.*unknown"):
        CodexAdapter(unknown=True)


def test_codex_rejects_unsupported_output_format_before_starting_server(
    tmp_path: Path,
) -> None:
    adapter, log_path = _adapter(tmp_path)

    with pytest.raises(TypeError, match="JSON Schema format 'regex'"):
        adapter.produce_sync("return formatted data", output_type=UnsupportedFormattedOutput)

    assert not log_path.exists()
    adapter.close()


def test_codex_tool_schema_validation_and_result_helpers() -> None:
    def variadic(*values: str) -> str:
        return "".join(values)

    def positional(value: str, /) -> str:
        return value

    def unannotated(value):
        return value

    def unresolved(value: str) -> str:
        return value

    unresolved.__annotations__["value"] = "MissingType"

    with pytest.raises(TypeError, match="cannot use variadic"):
        codex_adapter_module._dynamic_tool_spec(ToolSpec(fn=variadic, name="variadic"))
    with pytest.raises(TypeError, match="cannot use positional-only"):
        codex_adapter_module._dynamic_tool_spec(ToolSpec(fn=positional, name="positional"))

    explicit = {
        "type": "object",
        "properties": {"file": {"type": "string", "format": "uri"}},
    }
    with pytest.raises(
        TypeError,
        match=r"does not support JSON Schema format 'uri' at properties\.file",
    ):
        codex_adapter_module._dynamic_tool_spec(
            ToolSpec(fn=lambda: None, name="empty", json_schema=explicit)
        )
    assert explicit["properties"]["file"]["format"] == "uri"

    supported_explicit = {
        "type": "object",
        "properties": {"email": {"type": "string", "format": "email"}},
    }
    spec = codex_adapter_module._dynamic_tool_spec(
        ToolSpec(fn=lambda: None, name="empty", json_schema=supported_explicit)
    )
    assert spec["inputSchema"] == {
        "type": "object",
        "properties": {"email": {"type": "string", "format": "email"}},
        "additionalProperties": False,
    }
    assert spec["description"] == "empty"

    assert codex_adapter_module._output_json_schema(str) == {"type": "string"}
    assert codex_adapter_module._output_json_schema(int) == {"type": "integer"}
    assert codex_adapter_module._output_json_schema(float) == {"type": "number"}
    assert codex_adapter_module._output_json_schema(bool) == {"type": "boolean"}
    assert codex_adapter_module._output_json_schema(dict) == {
        "type": "object",
        "additionalProperties": False,
    }
    assert codex_adapter_module._output_json_schema(Answer)["additionalProperties"] is False
    formatted = codex_adapter_module._output_json_schema(FormattedOutput)["properties"]
    assert formatted["created_at"]["format"] == "date-time"
    assert formatted["elapsed"]["format"] == "duration"
    assert formatted["day"]["format"] == "date"
    assert formatted["clock"]["format"] == "time"
    assert formatted["email"]["format"] == "email"
    with pytest.raises(
        TypeError,
        match=r"does not support JSON Schema format 'regex' at properties\.matcher",
    ):
        codex_adapter_module._output_json_schema(UnsupportedFormattedOutput)
    assert (
        codex_adapter_module._parse_structured_result(CodexTurnResult(text='"plain"', turn={}), str)
        == "plain"
    )
    assert (
        codex_adapter_module._dynamic_tool_spec(ToolSpec(fn=unannotated, name="unannotated"))[
            "inputSchema"
        ]["properties"]["value"]["title"]
        == "Value"
    )
    assert (
        codex_adapter_module._dynamic_tool_spec(ToolSpec(fn=unresolved, name="unresolved"))[
            "inputSchema"
        ]["properties"]["value"]["title"]
        == "Value"
    )

    with pytest.raises(CodexAppServerProtocolError, match="object field"):
        codex_server_module._required_object({}, "thread")
    with pytest.raises(CodexAppServerProtocolError, match="string field"):
        codex_server_module._required_string({}, "id")
    assert codex_server_module._tool_result_text(DumpedToolResult(value=3)) == '{"value": 3}'


def test_codex_client_constructor_dependency_and_closed_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        CodexAppServerClient(())

    client = CodexAppServerClient(("missing-codex-test-command",))
    monkeypatch.setattr("shutil.which", lambda executable: None)
    with pytest.raises(ModuleNotFoundError, match="Codex CLI"):
        client.request("test", {})
    client.close()
    client.close()
    with pytest.raises(CodexAppServerDisconnected, match="closed"):
        client.request("test", {})


def test_codex_client_reports_popen_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CodexAppServerClient((sys.executable,))

    def fail_popen(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(ModuleNotFoundError, match="was not found"):
        client.request("test", {})


class _FakeStdin:
    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.closed = False
        self.messages: list[str] = []

    def write(self, value: str) -> int:
        if self.fail_write:
            raise BrokenPipeError
        self.messages.append(value)
        return len(value)

    def flush(self) -> None:
        if self.fail_write:
            raise BrokenPipeError

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: Any = (),
        stderr: Any = (),
        stdin: _FakeStdin | None = None,
        running: bool = True,
        wait_timeouts: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.stdin: Any = _FakeStdin() if stdin is None else stdin
        self.running = running
        self.terminated = False
        self.killed = False
        self.wait_timeouts = wait_timeouts

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_timeouts > 0:
            self.wait_timeouts -= 1
            raise subprocess.TimeoutExpired("fake", 0 if timeout is None else timeout)
        self.running = False
        return 0


def _as_popen(process: _FakeProcess) -> subprocess.Popen[str]:
    return cast(subprocess.Popen[str], process)


def test_codex_client_low_level_request_and_process_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,), request_timeout=0.01)

    with monkeypatch.context() as context:
        context.setattr(client, "_send", lambda message: None)
        with pytest.raises(TimeoutError, match="timed out"):
            client._request_raw("slow", {}, timeout=0.001)

    client._stderr_tail.append("boom")
    assert "boom" in str(client._disconnect_error("disconnected"))

    process = _FakeProcess(stdin=_FakeStdin(fail_write=True))
    client._process = _as_popen(process)
    client._initialized = True
    with pytest.raises(CodexAppServerDisconnected, match="stdin disconnected"):
        client._send({"method": "test"})
    assert isinstance(client._last_disconnect, CodexAppServerDisconnected)

    client._process = None
    with pytest.raises(CodexAppServerDisconnected, match="not running"):
        client._send({"method": "test"})


def test_codex_client_startup_race_and_initialize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,))
    health = iter((False, True))
    monkeypatch.setattr(client, "_is_healthy", lambda: next(health))
    monkeypatch.setattr(client, "_stop_process", lambda: pytest.fail("should not stop"))
    client._ensure_started()

    failing_client = CodexAppServerClient((sys.executable,))
    process = _FakeProcess(stdout=iter(()), stderr=iter(()))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        failing_client,
        "_request_raw",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CodexAppServerProtocolError("initialize failed")
        ),
    )
    with pytest.raises(CodexAppServerProtocolError, match="initialize failed"):
        failing_client._start_process()
    assert process.stdin.closed


def test_codex_client_reader_and_dispatch_protocol_edges() -> None:
    client = CodexAppServerClient((sys.executable,))
    client._read_stdout(_as_popen(_FakeProcess(stdout=None)))
    client._read_stderr(_as_popen(_FakeProcess(stderr=None)))

    stale = _FakeProcess(stderr=iter(("stale\n",)))
    client._process = _as_popen(_FakeProcess())
    client._read_stderr(_as_popen(stale))
    assert not client._stderr_tail

    process = _FakeProcess(stdout=iter(("[1]\n",)))
    client._process = _as_popen(process)
    client._initialized = True
    client._read_stdout(_as_popen(process))
    assert isinstance(client._last_disconnect, CodexAppServerDisconnected)

    future: Future[dict[str, Any]] = Future()
    client._pending[1] = future
    client._dispatch_message({"id": 1, "error": {"code": -1, "message": "bad"}})
    assert isinstance(future.exception(), CodexAppServerProtocolError)

    future = Future()
    client._pending[2] = future
    client._dispatch_message({"id": 2, "result": "bad"})
    assert isinstance(future.exception(), CodexAppServerProtocolError)

    done_future: Future[dict[str, Any]] = Future()
    done_future.set_result({})
    client._pending[3] = done_future
    client._dispatch_message({"id": 3, "result": {"ok": True}})
    client._dispatch_message({"method": "turn/completed", "params": "bad"})
    client._dispatch_message({"jsonrpc": "2.0"})


def test_codex_client_server_request_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "_send", sent.append)

    client._handle_server_request([], "ignored", {})
    assert sent == []

    client._handle_server_request(1, "unsupported", {})
    assert sent[-1]["error"]["code"] == -32601

    client._handle_server_request(2, "item/tool/call", "bad")
    assert sent[-1]["result"]["success"] is False

    client._handle_server_request(3, "item/tool/call", {"threadId": 1, "tool": "x"})
    assert sent[-1]["result"]["success"] is False

    client._handle_server_request(
        4,
        "item/tool/call",
        {"threadId": "t", "tool": "x", "arguments": []},
    )
    assert sent[-1]["result"]["success"] is False

    client._handle_server_request(
        5,
        "item/tool/call",
        {"threadId": "t", "tool": "missing", "arguments": {}},
    )
    assert sent[-1]["result"]["success"] is False


def test_codex_client_tool_request_system_exit_and_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "_send", sent.append)

    def exit_tool() -> None:
        raise SystemExit(3)

    client._thread_tools["thread"] = {
        "exit": codex_server_module._ToolBinding(fn=exit_tool, context=contextvars.copy_context())
    }
    with pytest.raises(SystemExit):
        client._handle_server_request(
            1,
            "item/tool/call",
            {"threadId": "thread", "tool": "exit", "arguments": {}},
        )

    client._handle_notification(
        "item/completed",
        {"turnId": "turn", "item": {"type": "agentMessage", "text": 1}},
    )
    client._handle_notification("item/completed", {"turnId": "turn", "item": "bad"})
    assert client._turn_state("turn").messages == []

    client._handle_notification("turn/completed", {"turn": {"id": 1}})
    client._handle_notification("other", {})
    assert not client._turn_state("turn").done.is_set()

    disconnect = CodexAppServerDisconnected("later")
    client._last_disconnect = disconnect
    state = client._turn_state("late-turn")
    assert state.done.is_set()
    assert state.error is disconnect


def test_codex_client_waiter_and_stop_process_edges() -> None:
    client = CodexAppServerClient((sys.executable,))
    pending: Future[dict[str, Any]] = Future()
    done: Future[dict[str, Any]] = Future()
    done.set_result({})
    waiting_turn = codex_server_module._TurnState()
    done_turn = codex_server_module._TurnState()
    done_turn.done.set()
    client._pending = {1: pending, 2: done}
    client._turns = {"waiting": waiting_turn, "done": done_turn}

    error = CodexAppServerDisconnected("closed")
    client._fail_waiters(error)
    assert pending.exception() is error
    assert done.result() == {}
    assert waiting_turn.error is error
    assert done_turn.error is None

    stale = _FakeProcess()
    client._process = _as_popen(_FakeProcess())
    client._mark_disconnected(_as_popen(stale), error)
    assert client._last_disconnect is None

    process = _FakeProcess(wait_timeouts=1)
    client._process = _as_popen(process)
    client._reader = current_thread()
    client._stderr_reader = None
    client._stop_process()
    assert process.terminated
    assert process.killed

    no_stdin = _FakeProcess(running=False)
    no_stdin.stdin = None
    client._process = _as_popen(no_stdin)
    client._stop_process()


def test_codex_client_stateless_turn_missing_data_and_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,))

    monkeypatch.setattr(client, "_start_thread", lambda params: {"thread": {"id": "thread"}})
    monkeypatch.setattr(
        client,
        "request",
        lambda method, params, **kwargs: {"turn": {"id": "turn"}} if method == "turn/start" else {},
    )
    state = codex_server_module._TurnState(messages=["{}"])
    state.done.set()
    monkeypatch.setattr(client, "_turn_state", lambda turn_id: state)
    with pytest.raises(CodexAppServerProtocolError, match="without turn data"):
        client.run_stateless_turn_sync(
            thread_params={},
            turn_params={},
            tools={},
            tool_context=contextvars.copy_context(),
            turn_timeout=1,
        )

    failing_client = CodexAppServerClient((sys.executable,))
    monkeypatch.setattr(
        failing_client, "_start_thread", lambda params: {"thread": {"id": "thread"}}
    )
    monkeypatch.setattr(
        failing_client,
        "request",
        lambda method, params, **kwargs: (
            (_ for _ in ()).throw(CodexAppServerProtocolError("turn start failed"))
            if method == "turn/start"
            else {}
        ),
    )
    with pytest.raises(CodexAppServerProtocolError, match="turn start failed"):
        failing_client.run_stateless_turn_sync(
            thread_params={},
            turn_params={},
            tools={},
            tool_context=contextvars.copy_context(),
            turn_timeout=1,
        )


def test_codex_client_native_approval_fallback_and_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient((sys.executable,))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "_send", sent.append)

    command_method = "item/commandExecution/requestApproval"
    permissions_method = "item/permissions/requestApproval"
    client._handle_server_request(1, command_method, "invalid")
    assert sent[-1] == {"id": 1, "result": {"decision": "decline"}}

    client._handle_server_request(2, permissions_method, {"threadId": "missing"})
    assert sent[-1] == {"id": 2, "result": {"permissions": {}, "scope": "turn"}}

    def broken_approval(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("approval failed")

    client._thread_approvals["bound"] = codex_server_module._ApprovalBinding(
        handler=broken_approval,
        context=contextvars.copy_context(),
    )
    client._handle_server_request(3, command_method, {"threadId": "bound"})
    assert sent[-1]["error"]["code"] == -32000
    assert "approval failed" in sent[-1]["error"]["message"]

    def interrupted_approval(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
        raise KeyboardInterrupt

    client._thread_approvals["interrupted"] = codex_server_module._ApprovalBinding(
        handler=interrupted_approval,
        context=contextvars.copy_context(),
    )
    with pytest.raises(KeyboardInterrupt):
        client._handle_server_request(4, command_method, {"threadId": "interrupted"})

    assert codex_server_module._default_native_approval_response(command_method) == {
        "decision": "decline"
    }
    assert codex_server_module._default_native_approval_response(permissions_method) == {
        "permissions": {},
        "scope": "turn",
    }
    assert codex_server_module._validated_native_approval_response(
        permissions_method,
        {
            "permissions": {"network": {"enabled": True}},
            "scope": "session",
            "strictAutoReview": True,
        },
    ) == {
        "permissions": {"network": {"enabled": True}},
        "scope": "session",
        "strictAutoReview": True,
    }

    with pytest.raises(TypeError, match="object response"):
        codex_server_module._validated_native_approval_response(command_method, "invalid")
    with pytest.raises(TypeError, match="permissions field"):
        codex_server_module._validated_native_approval_response(
            permissions_method, {"permissions": "invalid"}
        )
    with pytest.raises(ValueError, match="scope"):
        codex_server_module._validated_native_approval_response(
            permissions_method, {"permissions": {}, "scope": "invalid"}
        )
    with pytest.raises(TypeError, match="strictAutoReview"):
        codex_server_module._validated_native_approval_response(
            permissions_method, {"permissions": {}, "strictAutoReview": "yes"}
        )
    with pytest.raises(ValueError, match="acceptForSession"):
        codex_server_module._validated_native_approval_response(
            command_method, {"decision": "invalid"}
        )


def test_codex_async_bridge_propagates_schedule_and_task_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverScheduled:
        def __await__(self):  # type: ignore[no-untyped-def]
            if False:
                yield None
            return None

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        bridge = codex_server_module._await_result_in_loop(loop)

        async def run_from_thread(awaitable: Any) -> BaseException:
            completed = asyncio.Event()
            errors: list[BaseException] = []

            def invoke() -> None:
                try:
                    bridge(contextvars.copy_context(), awaitable)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    loop.call_soon_threadsafe(completed.set)

            worker = Thread(target=invoke)
            worker.start()
            await completed.wait()
            worker.join()
            assert len(errors) == 1
            return errors[0]

        async def fails() -> None:
            raise ValueError("task failed")

        task_error = await run_from_thread(fails())
        assert isinstance(task_error, ValueError)
        assert str(task_error) == "task failed"

        original_ensure_future = codex_server_module.asyncio.ensure_future

        def reject_sentinel(awaitable: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(awaitable, NeverScheduled):
                raise RuntimeError("schedule failed")
            return original_ensure_future(awaitable, *args, **kwargs)

        monkeypatch.setattr(codex_server_module.asyncio, "ensure_future", reject_sentinel)
        scheduling_error = await run_from_thread(NeverScheduled())
        assert isinstance(scheduling_error, RuntimeError)
        assert str(scheduling_error) == "schedule failed"

    asyncio.run(scenario())


def test_codex_native_approval_classifies_edge_cases_and_declines_unrepresentable_edits() -> None:
    command_method = "item/commandExecution/requestApproval"
    file_method = "item/fileChange/requestApproval"
    permissions_method = "item/permissions/requestApproval"

    with pytest.raises(ValueError, match="Unsupported Codex native approval method"):
        codex_adapter_module._codex_native_approval_details("unknown", {})

    assert codex_adapter_module._codex_command_risk({"path": ".env"}) == "sensitive"
    assert (
        codex_adapter_module._codex_command_risk({"additionalPermissions": {"network": True}})
        == "sensitive"
    )
    assert not codex_adapter_module._is_read_only_command_action("invalid")
    assert (
        codex_adapter_module._codex_command_risk({"commandActions": [{"type": "read"}]})
        == "read_only"
    )
    assert (
        codex_adapter_module._codex_command_risk({"commandActions": [{"type": "unknown"}]})
        == "mutating"
    )
    assert arguments_target_secret_file([{"path": ".env.local"}, 1])
    assert not arguments_target_secret_file(1)

    denied = ApprovalDecision.deny()
    edited = ApprovalDecision.edit({"path": "other.txt"})
    assert codex_adapter_module._codex_native_approval_response(command_method, {}, denied) == {
        "decision": "decline"
    }
    assert codex_adapter_module._codex_native_approval_response(file_method, {}, edited) == {
        "decision": "decline"
    }
    assert codex_adapter_module._codex_native_approval_response(permissions_method, {}, denied) == {
        "permissions": {},
        "scope": "turn",
    }
    assert codex_adapter_module._codex_native_approval_response(
        permissions_method, {"permissions": "invalid"}, ApprovalDecision.allow()
    ) == {"permissions": {}, "scope": "turn"}
