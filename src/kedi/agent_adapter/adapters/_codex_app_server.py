from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import Future
from contextlib import suppress
from contextvars import Context
from dataclasses import dataclass, field
from threading import Event, Lock, RLock, Thread, current_thread
from typing import Any, Callable, Mapping, Sequence

JsonObject = dict[str, Any]


class CodexAppServerError(RuntimeError):
    """Base error raised by the Codex App Server transport."""


class CodexAppServerDisconnected(CodexAppServerError):
    """The App Server process or its stdio transport disconnected."""


class CodexAppServerProtocolError(CodexAppServerError):
    """The App Server emitted an invalid or failed JSON-RPC message."""


@dataclass(frozen=True)
class CodexTurnResult:
    text: str
    turn: JsonObject


@dataclass(frozen=True)
class _ToolBinding:
    fn: Callable[..., Any]
    context: Context


@dataclass
class _TurnState:
    done: Event = field(default_factory=Event)
    messages: list[str] = field(default_factory=list)
    turn: JsonObject | None = None
    error: BaseException | None = None


class CodexAppServerClient:
    """Own a persistent Codex App Server process and its JSON-RPC connection."""

    def __init__(
        self,
        command: Sequence[str] = ("codex", "app-server", "--stdio"),
        *,
        env: Mapping[str, str] | None = None,
        request_timeout: float = 30,
    ) -> None:
        if not command:
            raise ValueError("Codex App Server command cannot be empty.")
        self._command = tuple(command)
        self._env = dict(env) if env is not None else None
        self._request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._stderr_reader: Thread | None = None
        self._start_lock = RLock()
        self._write_lock = Lock()
        self._state_lock = Lock()
        self._pending: dict[int, Future[JsonObject]] = {}
        self._turns: dict[str, _TurnState] = {}
        self._thread_tools: dict[str, dict[str, _ToolBinding]] = {}
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_request_id = 1
        self._initialized = False
        self._last_disconnect: CodexAppServerDisconnected | None = None
        self._closed = False

    async def run_stateless_turn(
        self,
        *,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
    ) -> CodexTurnResult:
        return await asyncio.to_thread(
            self.run_stateless_turn_sync,
            thread_params=thread_params,
            turn_params=turn_params,
            tools=tools,
            tool_context=tool_context,
            turn_timeout=turn_timeout,
        )

    def run_stateless_turn_sync(
        self,
        *,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
    ) -> CodexTurnResult:
        thread_response = self._start_thread(thread_params)
        thread = _required_object(thread_response, "thread")
        thread_id = _required_string(thread, "id")
        bindings = {
            name: _ToolBinding(fn=fn, context=tool_context.copy()) for name, fn in tools.items()
        }
        with self._state_lock:
            self._thread_tools[thread_id] = bindings
        turn_id: str | None = None
        try:
            turn_response = self.request("turn/start", {**turn_params, "threadId": thread_id})
            turn_id = _required_string(_required_object(turn_response, "turn"), "id")
            state = self._turn_state(turn_id)
            if not state.done.wait(timeout=turn_timeout):
                self._interrupt_turn(thread_id, turn_id)
                raise TimeoutError(
                    f"Codex turn {turn_id!r} did not complete within {turn_timeout:g}s."
                )
            if state.error is not None:
                raise state.error
            completed_turn = state.turn
            if completed_turn is None:
                raise CodexAppServerProtocolError("Codex turn completed without turn data.")
            status = completed_turn.get("status")
            if status != "completed":
                error = completed_turn.get("error")
                message = error.get("message") if isinstance(error, Mapping) else None
                raise CodexAppServerError(
                    str(message or f"Codex turn finished with status {status!r}.")
                )
            if not state.messages:
                raise CodexAppServerProtocolError(
                    "Codex turn completed without a final agent message."
                )
            return CodexTurnResult(text=state.messages[-1], turn=completed_turn)
        finally:
            with self._state_lock:
                self._thread_tools.pop(thread_id, None)
                if turn_id is not None:
                    self._turns.pop(turn_id, None)
            with suppress(CodexAppServerError, TimeoutError):
                self.request("thread/delete", {"threadId": thread_id})

    def request(
        self,
        method: str,
        params: JsonObject,
        *,
        timeout: float | None = None,
    ) -> JsonObject:
        self._ensure_started()
        return self._request_raw(method, params, timeout=timeout)

    def close(self) -> None:
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            self._fail_waiters(CodexAppServerDisconnected("Codex App Server client closed."))
            self._stop_process()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _start_thread(self, params: JsonObject) -> JsonObject:
        try:
            return self.request("thread/start", params)
        except CodexAppServerDisconnected:
            # Thread creation has no user-visible side effects, so one reconnect is safe.
            with self._start_lock:
                self._stop_process()
            return self.request("thread/start", params)

    def _ensure_started(self) -> None:
        if self._closed:
            raise CodexAppServerDisconnected("Codex App Server client is closed.")
        if self._is_healthy():
            return
        with self._start_lock:
            if self._is_healthy():
                return
            self._stop_process()
            self._start_process()

    def _is_healthy(self) -> bool:
        process = self._process
        return (
            self._initialized
            and process is not None
            and process.poll() is None
            and self._reader is not None
            and self._reader.is_alive()
        )

    def _start_process(self) -> None:
        executable = self._command[0]
        if os.path.sep not in executable and shutil.which(executable) is None:
            raise ModuleNotFoundError(
                "CodexAdapter requires the Codex CLI. Install it and ensure `codex` is on PATH."
            )
        process_env = None if self._env is None else {**os.environ, **self._env}
        try:
            process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=process_env,
            )
        except FileNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Codex App Server executable was not found: {executable}"
            ) from exc
        self._process = process
        self._last_disconnect = None
        self._stderr_tail.clear()
        self._reader = Thread(
            target=self._read_stdout,
            args=(process,),
            name="kedi-codex-app-server-stdout",
            daemon=True,
        )
        self._stderr_reader = Thread(
            target=self._read_stderr,
            args=(process,),
            name="kedi-codex-app-server-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()
        try:
            self._request_raw(
                "initialize",
                {
                    "clientInfo": {"name": "kedi", "title": "Kedi", "version": "0.4.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            self._send({"method": "initialized"})
        except BaseException:
            self._stop_process()
            raise
        self._initialized = True

    def _request_raw(
        self,
        method: str,
        params: JsonObject,
        *,
        timeout: float | None = None,
    ) -> JsonObject:
        with self._state_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            future: Future[JsonObject] = Future()
            self._pending[request_id] = future
        try:
            self._send({"id": request_id, "method": method, "params": params})
            try:
                return future.result(timeout=self._request_timeout if timeout is None else timeout)
            except TimeoutError as exc:
                raise TimeoutError(f"Codex App Server request {method!r} timed out.") from exc
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    def _send(self, message: JsonObject) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise self._disconnect_error("Codex App Server is not running.")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write_lock:
                process.stdin.write(payload + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            error = self._disconnect_error("Codex App Server stdin disconnected.")
            self._mark_disconnected(process, error)
            raise error from exc

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is None:
            return
        for line in stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._mark_disconnected(
                    process,
                    self._disconnect_error("Codex App Server emitted invalid JSON on stdout."),
                )
                return
            if not isinstance(message, dict):
                self._mark_disconnected(
                    process,
                    self._disconnect_error(
                        "Codex App Server emitted a non-object JSON-RPC message."
                    ),
                )
                return
            self._dispatch_message(message)
        self._mark_disconnected(
            process, self._disconnect_error("Codex App Server stdout disconnected.")
        )

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        for line in stderr:
            if process is self._process:
                self._stderr_tail.append(line.rstrip())

    def _dispatch_message(self, message: JsonObject) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if request_id is not None and isinstance(method, str):
            Thread(
                target=self._handle_server_request,
                args=(request_id, method, message.get("params")),
                name="kedi-codex-dynamic-tool",
                daemon=True,
            ).start()
            return
        if isinstance(request_id, int):
            with self._state_lock:
                future = self._pending.get(request_id)
            if future is None or future.done():
                return
            error = message.get("error")
            if isinstance(error, Mapping):
                future.set_exception(
                    CodexAppServerProtocolError(
                        f"Codex App Server request failed ({error.get('code')}): "
                        f"{error.get('message')}"
                    )
                )
                return
            result = message.get("result")
            if not isinstance(result, dict):
                future.set_exception(
                    CodexAppServerProtocolError(
                        "Codex App Server response did not contain an object result."
                    )
                )
                return
            future.set_result(result)
            return
        if isinstance(method, str):
            params = message.get("params")
            self._handle_notification(method, params if isinstance(params, dict) else {})

    def _handle_server_request(self, request_id: Any, method: str, params: Any) -> None:
        if not isinstance(request_id, (int, str)):
            return
        if method != "item/tool/call":
            self._send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported request: {method}",
                    },
                }
            )
            return
        if not isinstance(params, Mapping):
            self._tool_response(request_id, False, "Dynamic tool params must be an object.")
            return
        thread_id = params.get("threadId")
        tool_name = params.get("tool")
        arguments = params.get("arguments", {})
        if not isinstance(thread_id, str) or not isinstance(tool_name, str):
            self._tool_response(request_id, False, "Dynamic tool request is incomplete.")
            return
        if not isinstance(arguments, Mapping):
            self._tool_response(request_id, False, "Dynamic tool arguments must be an object.")
            return
        with self._state_lock:
            binding = self._thread_tools.get(thread_id, {}).get(tool_name)
        if binding is None:
            self._tool_response(request_id, False, f"Unknown Kedi tool: {tool_name}")
            return
        try:
            context = binding.context.copy()
            result = context.run(binding.fn, **dict(arguments))
            if inspect.isawaitable(result):
                result = context.run(asyncio.run, result)
            self._tool_response(request_id, True, _tool_result_text(result))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._tool_response(request_id, False, f"{type(exc).__name__}: {exc}")

    def _tool_response(self, request_id: int | str, success: bool, text: str) -> None:
        self._send(
            {
                "id": request_id,
                "result": {
                    "contentItems": [{"type": "inputText", "text": text}],
                    "success": success,
                },
            }
        )

    def _handle_notification(self, method: str, params: JsonObject) -> None:
        turn_id = params.get("turnId")
        if method == "item/completed" and isinstance(turn_id, str):
            item = params.get("item")
            if isinstance(item, Mapping) and item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str):
                    self._turn_state(turn_id).messages.append(text)
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                state = self._turn_state(turn["id"])
                state.turn = turn
                state.done.set()

    def _turn_state(self, turn_id: str) -> _TurnState:
        with self._state_lock:
            state = self._turns.setdefault(turn_id, _TurnState())
            if self._last_disconnect is not None and not state.done.is_set():
                state.error = self._last_disconnect
                state.done.set()
            return state

    def _interrupt_turn(self, thread_id: str, turn_id: str) -> None:
        with suppress(CodexAppServerError, TimeoutError):
            self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}, timeout=5)

    def _mark_disconnected(
        self, process: subprocess.Popen[str], error: CodexAppServerDisconnected
    ) -> None:
        with self._start_lock:
            if process is not self._process:
                return
            self._initialized = False
            self._last_disconnect = error
            self._fail_waiters(error)

    def _fail_waiters(self, error: CodexAppServerDisconnected) -> None:
        with self._state_lock:
            pending = tuple(self._pending.values())
            turns = tuple(self._turns.values())
        for future in pending:
            if not future.done():
                future.set_exception(error)
        for state in turns:
            if not state.done.is_set():
                state.error = error
                state.done.set()

    def _stop_process(self) -> None:
        process = self._process
        reader = self._reader
        stderr_reader = self._stderr_reader
        self._process = None
        self._reader = None
        self._stderr_reader = None
        self._initialized = False
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2)
        for thread in (reader, stderr_reader):
            if thread is not None and thread is not current_thread():
                thread.join(timeout=2)

    def _disconnect_error(self, message: str) -> CodexAppServerDisconnected:
        stderr = "\n".join(self._stderr_tail)
        if stderr:
            message = f"{message}\nApp Server stderr:\n{stderr}"
        return CodexAppServerDisconnected(message)


def _required_object(value: Mapping[str, Any], key: str) -> JsonObject:
    item = value.get(key)
    if not isinstance(item, dict):
        raise CodexAppServerProtocolError(f"Codex response is missing object field {key!r}.")
    return item


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise CodexAppServerProtocolError(f"Codex response is missing string field {key!r}.")
    return item


def _tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)
