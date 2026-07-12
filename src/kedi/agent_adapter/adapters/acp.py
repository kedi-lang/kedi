from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock, RLock, Thread
from typing import Any, Iterator, cast

from ...agent_profile import AgentProfile
from ...agent_profile.settings import ACP_AGENT_SETTING_KEYS, acp_agent_settings
from ...resolved_types import ResolvedKediType
from ..meta import AdapterCapabilities, AgentAdapterKind

JsonObject = dict[str, Any]


class ACPProtocolError(RuntimeError):
    """Raised when an ACP agent process emits invalid JSON-RPC."""


class ACPDisconnected(RuntimeError):
    """Raised when the ACP agent process disconnects."""


@dataclass
class _SessionState:
    messages: list[str] = field(default_factory=list)


class ACPClient:
    """Small JSON-RPC stdio client for Agent Client Protocol v1."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        request_timeout: float = 30,
    ) -> None:
        if not command:
            raise ValueError("ACP command cannot be empty.")
        self._command = tuple(command)
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        self._request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._stderr_reader: Thread | None = None
        self._start_lock = RLock()
        self._write_lock = Lock()
        self._state_lock = Lock()
        self._pending: dict[int, Future[JsonObject | None]] = {}
        self._sessions: dict[str, _SessionState] = {}
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_request_id = 1
        self._initialized = False
        self._closed = False

    async def run_prompt(
        self,
        *,
        prompt: str,
        cwd: str,
        timeout: float,
    ) -> str:
        return await asyncio.to_thread(
            self.run_prompt_sync, prompt=prompt, cwd=cwd, timeout=timeout
        )

    def run_prompt_sync(self, *, prompt: str, cwd: str, timeout: float) -> str:
        session_id = self._new_session(cwd)
        state = _SessionState()
        with self._state_lock:
            self._sessions[session_id] = state
        try:
            self.request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt}],
                },
                timeout=timeout,
            )
            return "".join(state.messages)
        finally:
            with self._state_lock:
                self._sessions.pop(session_id, None)
            with suppress(Exception):
                self.request("session/close", {"sessionId": session_id}, timeout=1)

    def request(
        self,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
    ) -> JsonObject | None:
        self._ensure_started(initialize=method != "initialize")
        future: Future[JsonObject | None] = Future()
        with self._state_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending[request_id] = future
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            return future.result(timeout=timeout or self._request_timeout)
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    def close(self) -> None:
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            with suppress(Exception):
                process.wait(timeout=1)
            if process.poll() is None:
                process.kill()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _ensure_started(self, *, initialize: bool = True) -> None:
        with self._start_lock:
            if self._closed:
                raise ACPDisconnected("ACP client is closed.")
            process = self._process
            if process is not None and process.poll() is None:
                if initialize and not self._initialized:
                    self._initialize()
                return
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env={**os.environ, **self._env} if self._env is not None else None,
                cwd=self._cwd,
            )
            self._reader = Thread(target=self._read_stdout, daemon=True)
            self._stderr_reader = Thread(target=self._read_stderr, daemon=True)
            self._reader.start()
            self._stderr_reader.start()
            if initialize:
                self._initialize()

    def _initialize(self) -> None:
        response = self.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "kedi",
                    "title": "Kedi",
                    "version": "0",
                },
            },
        )
        if not isinstance(response, dict):
            raise ACPProtocolError("ACP initialize response must be an object.")
        self._initialized = True

    def _new_session(self, cwd: str) -> str:
        response = self.request("session/new", {"cwd": cwd, "mcpServers": []})
        if not isinstance(response, dict) or not isinstance(response.get("sessionId"), str):
            raise ACPProtocolError("ACP session/new response must contain sessionId.")
        return response["sessionId"]

    def _write(self, payload: JsonObject) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ACPDisconnected(self._disconnect_message())
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._write_lock:
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except BrokenPipeError as exc:
                raise ACPDisconnected(self._disconnect_message()) from exc

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._fail_all(ACPProtocolError(f"ACP emitted invalid JSON: {line.strip()}"))
                return
            if not isinstance(message, dict):
                self._fail_all(ACPProtocolError("ACP emitted a non-object JSON-RPC message."))
                return
            self._handle_message(message)
        self._fail_all(ACPDisconnected(self._disconnect_message()))

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip())

    def _handle_message(self, message: JsonObject) -> None:
        if "id" in message:
            request_id = message["id"]
            if not isinstance(request_id, int):
                return
            with self._state_lock:
                future = self._pending.get(request_id)
            if future is None or future.done():
                return
            if "error" in message:
                future.set_exception(ACPProtocolError(str(message["error"])))
            else:
                result = message.get("result")
                future.set_result(cast(JsonObject | None, result))
            return
        if message.get("method") == "session/update":
            self._handle_session_update(cast(JsonObject, message.get("params") or {}))

    def _handle_session_update(self, params: JsonObject) -> None:
        session_id = params.get("sessionId")
        update = params.get("update")
        if not isinstance(session_id, str) or not isinstance(update, dict):
            return
        if update.get("sessionUpdate") != "agent_message_chunk":
            return
        content = update.get("content")
        if not isinstance(content, dict) or content.get("type") != "text":
            return
        text = content.get("text")
        if not isinstance(text, str):
            return
        with self._state_lock:
            state = self._sessions.get(session_id)
        if state is not None:
            state.messages.append(text)

    def _fail_all(self, exc: BaseException) -> None:
        with self._state_lock:
            pending = tuple(self._pending.values())
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    def _disconnect_message(self) -> str:
        details = "\n".join(self._stderr_tail)
        if details:
            return f"ACP agent process disconnected.\n{details}"
        return "ACP agent process disconnected."


class ACPAdapter:
    """Kedi adapter for generic Agent Client Protocol harnesses."""

    kind: AgentAdapterKind = "agent-harness"
    shortname = "acp"
    capabilities = AdapterCapabilities(
        supports_structured_output=False,
        supports_tool_registration=False,
        supports_mcp=False,
        supports_profile_override=True,
        supports_model_override=False,
        supports_effort=False,
        supports_settings=True,
        settings=ACP_AGENT_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | None = None,
        *,
        command: Sequence[str] | str | None = None,
        model_settings: Mapping[str, Any] | None = None,
        request_timeout: float = 30,
        client: ACPClient | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._command = _normalize_optional_command(command)
        self._settings = {
            **acp_agent_settings(model_settings or {}),
            **self._validated_settings(kwargs),
        }
        self._request_timeout = request_timeout
        self._client = client
        self._clients: dict[
            tuple[tuple[str, ...], str, tuple[tuple[str, str], ...]], ACPClient
        ] = {}
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_acp_profile_stack_{id(self)}",
            default=(),
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return null_tool_scope()

    def register_tool(self, tool: Any) -> None:
        return None

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return ()

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            "ACPAdapter does not support structured template output yet. "
            "Use raw invoke or an adapter with structured output support."
        )

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            "ACPAdapter does not support structured template output yet. "
            "Use raw invoke or an adapter with structured output support."
        )

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        client, cwd, timeout = self._client_for_run(kwargs)
        return await client.run_prompt(prompt=prompt, cwd=cwd, timeout=timeout)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        client, cwd, timeout = self._client_for_run(kwargs)
        return client.run_prompt_sync(prompt=prompt, cwd=cwd, timeout=timeout)

    def default_model(self) -> str | None:
        return self._model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        token = self._profile_stack_var.set((*self._profile_stack_var.get(), profile))
        try:
            yield
        finally:
            self._profile_stack_var.reset(token)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _client_for_run(self, kwargs: Mapping[str, Any]) -> tuple[ACPClient, str, float]:
        settings = {
            **self._settings,
            **acp_agent_settings(self.active_profile.settings),
            **self._validated_settings(kwargs),
        }
        cwd = _normalize_cwd(settings.get("cwd"))
        timeout = settings.get("timeout", self._request_timeout)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise TypeError("ACP timeout setting must be a positive number.")
        env = _normalize_env(settings.get("env"))
        if self._client is not None:
            return self._client, cwd, float(timeout)
        command = _normalize_optional_command(self.active_profile.agent_command)
        if command is None:
            command = self._command
        if command is None:
            command = _normalize_optional_command(os.getenv("KEDI_ACP_AGENT_COMMAND"))
        if command is None:
            raise RuntimeError(
                "ACPAdapter requires a command. Set KEDI_ACP_AGENT_COMMAND, pass "
                "--acp-command, or use multiline '> agent:' syntax."
            )
        key = (command, cwd, tuple(sorted(env.items())))
        client = self._clients.get(key)
        if client is None:
            client = ACPClient(command, env=env, cwd=cwd, request_timeout=self._request_timeout)
            self._clients[key] = client
        return client, cwd, float(timeout)

    @staticmethod
    def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(settings).difference(ACP_AGENT_SETTING_KEYS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported ACP setting(s): {names}")
        return dict(settings)


@contextmanager
def null_tool_scope() -> Iterator[None]:
    yield


def _normalize_optional_command(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(shlex.split(value))
    elif isinstance(value, Sequence):
        parts = tuple(value)
    else:
        raise TypeError("ACP command must be a string or a sequence of strings.")
    if not parts:
        return None
    if not all(isinstance(part, str) and part for part in parts):
        raise TypeError("ACP command entries must be non-empty strings.")
    return parts


def _normalize_cwd(value: Any) -> str:
    if value is None:
        return os.getcwd()
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError("ACP cwd setting must be a string or path-like object.")


def _normalize_env(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("ACP env setting must be a mapping of string keys and values.")
    env: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("ACP env setting must be a mapping of string keys and values.")
        env[key] = item
    return env
