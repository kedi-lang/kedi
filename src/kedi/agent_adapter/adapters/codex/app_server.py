from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext, suppress
from contextvars import Context
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Event, Lock, RLock, Thread, current_thread
from typing import Any, Awaitable, Callable, Mapping, Sequence, cast

from kedi._version import __version__
from kedi.agent_adapter.events import AgentMessagePhase, AgentMessageResolver
from kedi.agent_adapter.execution import (
    _SHARED_LOOP,
    OwnedAsyncLoop,
    _running_loop,
    current_async_owner_loop,
    run_sync,
)
from kedi.agent_adapter.subagents.control import current_live_input, current_live_revision

from .live import TurnBindings, ensure_turn_current, run_cancellable_binding, run_cancellable_turn

JsonObject = dict[str, Any]
AwaitResult = Callable[[Context, Awaitable[Any]], Any]
ServerApprovalHandler = Callable[[str, Mapping[str, Any]], Any]

_NATIVE_APPROVAL_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
    }
)
_TOOL_ITEM_TYPES = frozenset(
    {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
    }
)


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
    thread_id: str = ""
    compacted: bool = False


@dataclass(frozen=True)
class _ToolBinding:
    fn: Callable[..., Any]
    context: Context
    await_result: AwaitResult
    lifetime: TurnBindings | None = None


@dataclass(frozen=True)
class _ApprovalBinding:
    handler: ServerApprovalHandler
    context: Context
    await_result: AwaitResult
    lifetime: TurnBindings | None = None


@dataclass(frozen=True)
class _TurnMessage:
    message_id: str
    text: str
    phase: AgentMessagePhase | None


@dataclass
class _TurnState:
    done: Event = field(default_factory=Event)
    messages: list[_TurnMessage] = field(default_factory=list)
    turn: JsonObject | None = None
    error: BaseException | None = None
    compacted: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    usage_total: int | None = None
    model_requests: int = 0
    tool_calls: int = 0


class CodexAppServerClient:
    """Own a persistent Codex App Server process and its JSON-RPC connection."""

    def __init__(
        self,
        command: Sequence[str] = ("codex", "app-server", "--stdio"),
        *,
        env: Mapping[str, str] | None = None,
        request_timeout: float = 30,
        dynamic_request_workers: int = 4,
        max_dynamic_requests: int = 32,
    ) -> None:
        if not command:
            raise ValueError("Codex App Server command cannot be empty.")
        if dynamic_request_workers < 1:
            raise ValueError("Codex dynamic request workers must be at least 1.")
        if max_dynamic_requests < dynamic_request_workers:
            raise ValueError(
                "Codex max dynamic requests cannot be smaller than dynamic request workers."
            )
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
        self._thread_approvals: dict[str, _ApprovalBinding] = {}
        self._thread_event_resolvers: dict[str, AgentMessageResolver] = {}
        self._retained_threads: set[str] = set()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_request_id = 1
        self._initialized = False
        self._last_disconnect: CodexAppServerDisconnected | None = None
        self._closed = False
        self._tool_lock = Lock()
        self._tool_loop: asyncio.AbstractEventLoop | None = None
        self._private_tool_loop: OwnedAsyncLoop | None = None
        self._dynamic_slots = BoundedSemaphore(max_dynamic_requests)
        self._dynamic_executor = ThreadPoolExecutor(
            max_workers=dynamic_request_workers,
            thread_name_prefix="kedi-codex-dynamic",
        )

    async def run_stateless_turn(
        self,
        *,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
        approval_handler: ServerApprovalHandler | None = None,
        approval_context: Context | None = None,
        event_resolver: AgentMessageResolver | None = None,
    ) -> CodexTurnResult:
        loop = asyncio.get_running_loop()
        return await run_cancellable_turn(
            lambda stop: self.run_stateless_turn_sync(
                thread_params=thread_params,
                turn_params=turn_params,
                tools=tools,
                tool_context=tool_context,
                turn_timeout=turn_timeout,
                approval_handler=approval_handler,
                approval_context=approval_context,
                event_resolver=event_resolver,
                await_result=_await_result_in_loop(loop),
                stop_event=stop,
            )
        )

    async def run_conversation_turn(
        self,
        *,
        thread_id: str | None,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
        approval_handler: ServerApprovalHandler | None = None,
        approval_context: Context | None = None,
        event_resolver: AgentMessageResolver | None = None,
    ) -> CodexTurnResult:
        loop = asyncio.get_running_loop()
        return await run_cancellable_turn(
            lambda stop: self.run_conversation_turn_sync(
                thread_id=thread_id,
                thread_params=thread_params,
                turn_params=turn_params,
                tools=tools,
                tool_context=tool_context,
                turn_timeout=turn_timeout,
                approval_handler=approval_handler,
                approval_context=approval_context,
                event_resolver=event_resolver,
                await_result=_await_result_in_loop(loop),
                stop_event=stop,
            )
        )

    def run_stateless_turn_sync(
        self,
        *,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
        approval_handler: ServerApprovalHandler | None = None,
        approval_context: Context | None = None,
        await_result: AwaitResult | None = None,
        event_resolver: AgentMessageResolver | None = None,
        stop_event: Event | None = None,
    ) -> CodexTurnResult:
        return self._run_turn_sync(
            thread_id=None,
            retain_thread=False,
            thread_params=thread_params,
            turn_params=turn_params,
            tools=tools,
            tool_context=tool_context,
            turn_timeout=turn_timeout,
            approval_handler=approval_handler,
            approval_context=approval_context,
            await_result=await_result,
            event_resolver=event_resolver,
            stop_event=stop_event,
        )

    def run_conversation_turn_sync(
        self,
        *,
        thread_id: str | None,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
        approval_handler: ServerApprovalHandler | None = None,
        approval_context: Context | None = None,
        await_result: AwaitResult | None = None,
        event_resolver: AgentMessageResolver | None = None,
        stop_event: Event | None = None,
    ) -> CodexTurnResult:
        return self._run_turn_sync(
            thread_id=thread_id,
            retain_thread=True,
            thread_params=thread_params,
            turn_params=turn_params,
            tools=tools,
            tool_context=tool_context,
            turn_timeout=turn_timeout,
            approval_handler=approval_handler,
            approval_context=approval_context,
            await_result=await_result,
            event_resolver=event_resolver,
            stop_event=stop_event,
        )

    def _run_turn_sync(
        self,
        *,
        thread_id: str | None,
        retain_thread: bool,
        thread_params: JsonObject,
        turn_params: JsonObject,
        tools: Mapping[str, Callable[..., Any]],
        tool_context: Context,
        turn_timeout: float,
        approval_handler: ServerApprovalHandler | None,
        approval_context: Context | None,
        await_result: AwaitResult | None,
        event_resolver: AgentMessageResolver | None,
        stop_event: Event | None,
    ) -> CodexTurnResult:
        control = current_live_input()
        revision = current_live_revision(control) if control is not None else 0
        ensure_turn_current(control, revision, stop_event)
        created_thread = thread_id is None
        thread_response = (
            self._start_thread(thread_params)
            if thread_id is None
            else self._resume_thread(thread_id)
        )
        thread = _required_object(thread_response, "thread")
        resolved_thread_id = _required_string(thread, "id")
        # A sync turn called inline on a running loop blocks that loop until the turn ends, so
        # its tools run on a loop of the client's own.
        resolve_awaitable = (
            await_result
            if await_result is not None
            else self._await_private_result
            if _running_loop() is not None
            else self._await_sync_result
        )
        lifetime = TurnBindings()
        bindings = {
            name: _ToolBinding(
                fn=fn,
                context=tool_context.copy(),
                await_result=resolve_awaitable,
                lifetime=lifetime,
            )
            for name, fn in tools.items()
        }
        with self._state_lock:
            self._thread_tools[resolved_thread_id] = bindings
            if approval_handler is not None:
                self._thread_approvals[resolved_thread_id] = _ApprovalBinding(
                    handler=approval_handler,
                    context=(
                        approval_context if approval_context is not None else tool_context
                    ).copy(),
                    await_result=resolve_awaitable,
                    lifetime=lifetime,
                )
            if event_resolver is not None:
                self._thread_event_resolvers[resolved_thread_id] = event_resolver
        turn_id: str | None = None
        completed = False
        try:
            ensure_turn_current(control, revision, stop_event)
            turn_response = self.request(
                "turn/start", {**turn_params, "threadId": resolved_thread_id}
            )
            turn_id = _required_string(_required_object(turn_response, "turn"), "id")
            if control is not None:
                control.confirm_contents(
                    tuple(
                        part["text"]
                        for part in turn_params.get("input", ())
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    )
                )
            state = self._turn_state(turn_id)
            deadline = time.monotonic() + turn_timeout
            interrupted = False
            while not state.done.wait(timeout=0.02):
                if (stop_event is not None and stop_event.is_set()) or (
                    control is not None and control.revision != revision
                ):
                    self._request_turn_control(
                        "turn/interrupt",
                        {"threadId": resolved_thread_id, "turnId": turn_id},
                        state,
                    )
                    if not state.done.wait(timeout=5.0):
                        raise TimeoutError("Codex interrupt did not reach terminal completion")
                    interrupted = True
                    break
                if control is not None and control.pending:
                    items = control.take_pending(revision=revision)
                    if items:
                        delivered = self._request_turn_control(
                            "turn/steer",
                            {
                                "threadId": resolved_thread_id,
                                "expectedTurnId": turn_id,
                                "input": [
                                    {"type": "text", "text": item.render()} for item in items
                                ],
                            },
                            state,
                        )
                        if delivered:
                            control.mark_delivered(items)
                        else:
                            control.restore_pending(items)
                if time.monotonic() >= deadline:
                    self._interrupt_turn(resolved_thread_id, turn_id)
                    raise TimeoutError(
                        f"Codex turn {turn_id!r} did not complete within {turn_timeout:g}s."
                    )
            error, completed_turn, messages, compacted = self._turn_snapshot(turn_id)
            if error is not None:
                raise error
            if completed_turn is None:
                raise CodexAppServerProtocolError("Codex turn completed without turn data.")
            status = completed_turn.get("status")
            if status != "completed" and not (interrupted and status == "interrupted"):
                error = completed_turn.get("error")
                message = error.get("message") if isinstance(error, Mapping) else None
                raise CodexAppServerError(
                    str(message or f"Codex turn finished with status {status!r}.")
                )
            final_message = _final_turn_message(messages)
            if final_message is None and not interrupted:
                raise CodexAppServerProtocolError(
                    "Codex turn completed without a final agent message."
                )
            completed = True
            with self._state_lock:
                if state.usage:
                    completed_turn = {
                        **completed_turn,
                        "usage": {**state.usage, "requests": state.model_requests},
                    }
                completed_turn = {**completed_turn, "toolCalls": state.tool_calls}
            if retain_thread:
                with self._state_lock:
                    self._retained_threads.add(resolved_thread_id)
            return CodexTurnResult(
                text=final_message.text if final_message is not None else "",
                turn=completed_turn,
                thread_id=resolved_thread_id,
                compacted=compacted,
            )
        finally:
            with self._state_lock:
                self._thread_tools.pop(resolved_thread_id, None)
                self._thread_approvals.pop(resolved_thread_id, None)
                self._thread_event_resolvers.pop(resolved_thread_id, None)
                if turn_id is not None:
                    self._turns.pop(turn_id, None)
            try:
                lifetime.close()
            finally:
                if not retain_thread or (created_thread and not completed):
                    with suppress(CodexAppServerError, TimeoutError):
                        self.delete_thread(resolved_thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.request("thread/delete", {"threadId": thread_id})
        with self._state_lock:
            self._retained_threads.discard(thread_id)

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
            with self._state_lock:
                retained_threads = tuple(self._retained_threads)
            for thread_id in retained_threads:
                with suppress(CodexAppServerError, TimeoutError):
                    self.delete_thread(thread_id)
            self._closed = True
            self._fail_waiters(CodexAppServerDisconnected("Codex App Server client closed."))
            self._stop_process()
            self._dynamic_executor.shutdown(wait=False, cancel_futures=True)
            private_tool_loop = self._private_tool_loop
        with self._tool_lock:
            tool_loop, self._tool_loop = self._tool_loop, None
        if tool_loop is not None:
            _SHARED_LOOP.release(tool_loop)
        if private_tool_loop is not None:
            private_tool_loop.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _await_sync_result(self, context: Context, awaitable: Awaitable[Any]) -> Any:
        with self._start_lock:
            self._ensure_open(awaitable)
        if context.run(current_async_owner_loop) is None:
            with self._tool_lock:  # not the start lock: holding can wait for a closing loop
                if self._tool_loop is None and not self._closed:
                    # A background delegation must outlive the dynamic tool that started it.
                    self._tool_loop = _SHARED_LOOP.hold()
        return context.run(run_sync, run_cancellable_binding(awaitable))

    def _await_private_result(self, context: Context, awaitable: Awaitable[Any]) -> Any:
        with self._start_lock:
            self._ensure_open(awaitable)
            if self._private_tool_loop is None:
                self._private_tool_loop = OwnedAsyncLoop()
            result = context.run(self._private_tool_loop.submit, run_cancellable_binding(awaitable))
        return result.result()

    def _ensure_open(self, awaitable: Awaitable[Any]) -> None:
        if self._closed:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise CodexAppServerDisconnected("Codex App Server client is closed.")

    def _start_thread(self, params: JsonObject) -> JsonObject:
        try:
            return self.request("thread/start", params)
        except CodexAppServerDisconnected:
            # Thread creation has no user-visible side effects, so one reconnect is safe.
            with self._start_lock:
                self._stop_process()
            return self.request("thread/start", params)

    def _resume_thread(self, thread_id: str) -> JsonObject:
        params = {"threadId": thread_id, "excludeTurns": True}
        try:
            return self.request("thread/resume", params)
        except CodexAppServerDisconnected:
            with self._start_lock:
                self._stop_process()
            return self.request("thread/resume", params)

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
        with self._state_lock:
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
                    "clientInfo": {"name": "kedi", "title": "Kedi", "version": __version__},
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
            except FutureTimeoutError as exc:
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
                with self._state_lock:
                    self._stderr_tail.append(line.rstrip())

    def _dispatch_message(self, message: JsonObject) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if request_id is not None and isinstance(method, str):
            self._submit_server_request(request_id, method, message.get("params"))
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

    def _submit_server_request(self, request_id: Any, method: str, params: Any) -> None:
        if not self._dynamic_slots.acquire(blocking=False):
            self._send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "Kedi dynamic request capacity is exhausted.",
                    },
                }
            )
            return
        try:
            future = self._dynamic_executor.submit(
                self._handle_server_request,
                request_id,
                method,
                params,
            )
        except RuntimeError:
            self._dynamic_slots.release()
            self._send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "Kedi Codex dynamic request executor is closed.",
                    },
                }
            )
            return
        future.add_done_callback(lambda _completed: self._dynamic_slots.release())

    def _handle_server_request(self, request_id: Any, method: str, params: Any) -> None:
        if not isinstance(request_id, (int, str)):
            return
        if method == "item/tool/call":
            self._handle_dynamic_tool_request(request_id, params)
            return
        if method in _NATIVE_APPROVAL_METHODS:
            self._handle_native_approval_request(request_id, method, params)
            return
        self._send(
            {
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unsupported request: {method}",
                },
            }
        )

    def _handle_dynamic_tool_request(self, request_id: int | str, params: Any) -> None:
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
        turn_id = params.get("turnId")
        if isinstance(turn_id, str):
            with self._state_lock:
                self._turns.setdefault(turn_id, _TurnState()).tool_calls += 1
        try:
            result = _invoke_binding(binding, **dict(arguments))
            self._tool_response(request_id, True, _tool_result_text(result))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._tool_response(request_id, False, f"{type(exc).__name__}: {exc}")

    def _handle_native_approval_request(
        self,
        request_id: int | str,
        method: str,
        params: Any,
    ) -> None:
        if not isinstance(params, Mapping):
            self._send(
                {
                    "id": request_id,
                    "result": _default_native_approval_response(method),
                }
            )
            return
        thread_id = params.get("threadId")
        with self._state_lock:
            binding = self._thread_approvals.get(thread_id) if isinstance(thread_id, str) else None
        if binding is None:
            self._send(
                {
                    "id": request_id,
                    "result": _default_native_approval_response(method),
                }
            )
            return
        try:
            result = _invoke_binding(binding, method, dict(params))
            response = _validated_native_approval_response(method, result)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": f"Kedi approval handler failed: {type(exc).__name__}: {exc}",
                    },
                }
            )
            return
        self._send({"id": request_id, "result": response})

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
        thread_id = params.get("threadId")
        resolver = self._event_resolver(thread_id)
        item = params.get("item")
        if method == "thread/tokenUsage/updated" and isinstance(turn_id, str):
            token_usage = params.get("tokenUsage")
            if isinstance(token_usage, Mapping):
                last, total = token_usage.get("last"), token_usage.get("total")
                if isinstance(last, Mapping) and isinstance(total, Mapping):
                    total_tokens = total.get("totalTokens")
                    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
                        with self._state_lock:
                            state = self._turns.setdefault(turn_id, _TurnState())
                            if state.usage_total != total_tokens:
                                state.usage_total = total_tokens
                                state.model_requests += 1
                                for key, value in last.items():
                                    if isinstance(value, int) and not isinstance(value, bool):
                                        state.usage[key] = state.usage.get(key, 0) + value
            return
        if (
            method in {"item/started", "item/completed"}
            and isinstance(turn_id, str)
            and isinstance(item, Mapping)
            and item.get("type") == "contextCompaction"
        ):
            with self._state_lock:
                self._turns.setdefault(turn_id, _TurnState()).compacted = True
        if method == "item/agentMessage/delta" and resolver is not None:
            item_id = params.get("itemId")
            delta = params.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                resolver.append_delta(item_id, delta)
            return
        if method == "item/started" and resolver is not None:
            if isinstance(item, Mapping) and item.get("type") in _TOOL_ITEM_TYPES:
                resolver.mark_tool_boundary()
            return
        if method == "item/completed" and isinstance(turn_id, str):
            if isinstance(item, Mapping) and item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str):
                    message_id = item.get("id")
                    resolved_message_id = (
                        message_id
                        if isinstance(message_id, str)
                        else f"{turn_id}:message:{len(self._turn_state(turn_id).messages) + 1}"
                    )
                    phase = _message_phase(item.get("phase"))
                    self._append_turn_message(
                        turn_id,
                        _TurnMessage(
                            message_id=resolved_message_id,
                            text=text,
                            phase=phase,
                        ),
                    )
                    if resolver is not None:
                        resolver.complete_message(
                            resolved_message_id,
                            text,
                            explicit_phase=phase,
                        )
            elif (
                resolver is not None
                and isinstance(item, Mapping)
                and item.get("type") in _TOOL_ITEM_TYPES
            ):
                resolver.mark_tool_boundary()
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                self._complete_turn(turn)

    def _turn_state(self, turn_id: str) -> _TurnState:
        with self._state_lock:
            state = self._turns.setdefault(turn_id, _TurnState())
            if self._last_disconnect is not None and not state.done.is_set():
                state.error = self._last_disconnect
                state.done.set()
            return state

    def _event_resolver(self, thread_id: object) -> AgentMessageResolver | None:
        if not isinstance(thread_id, str):
            return None
        with self._state_lock:
            return self._thread_event_resolvers.get(thread_id)

    def _append_turn_message(self, turn_id: str, message: _TurnMessage) -> None:
        with self._state_lock:
            self._turns.setdefault(turn_id, _TurnState()).messages.append(message)

    def _complete_turn(self, turn: JsonObject) -> None:
        turn_id = cast(str, turn["id"])
        with self._state_lock:
            state = self._turns.setdefault(turn_id, _TurnState())
            state.turn = turn
            state.done.set()

    def _turn_snapshot(
        self,
        turn_id: str,
    ) -> tuple[BaseException | None, JsonObject | None, tuple[_TurnMessage, ...], bool]:
        with self._state_lock:
            state = self._turns.setdefault(turn_id, _TurnState())
            return state.error, state.turn, tuple(state.messages), state.compacted

    def _request_turn_control(self, method: str, params: JsonObject, state: _TurnState) -> bool:
        try:
            self.request(method, params)
        except CodexAppServerProtocolError:
            with self._state_lock:
                if state.turn is not None and state.turn.get("status") == "completed":
                    # Completion won the race; input still belongs to the next turn.
                    return False
            raise
        return True

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
            for state in turns:
                if not state.done.is_set():
                    state.error = error
                    state.done.set()
        for future in pending:
            if not future.done():
                future.set_exception(error)

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
        with self._state_lock:
            stderr = "\n".join(self._stderr_tail)
        if stderr:
            message = f"{message}\nApp Server stderr:\n{stderr}"
        return CodexAppServerDisconnected(message)


def _invoke_binding(
    binding: _ToolBinding | _ApprovalBinding,
    *args: Any,
    **kwargs: Any,
) -> Any:
    context = binding.context.copy()
    callable_ = binding.fn if isinstance(binding, _ToolBinding) else binding.handler
    with binding.lifetime.binding() if binding.lifetime is not None else nullcontext():
        result = context.run(callable_, *args, **kwargs)
        if not inspect.isawaitable(result):
            return result
        awaitable = cast(Awaitable[Any], result)
        if binding.lifetime is not None:
            awaitable = binding.lifetime.resolve(awaitable)
        return binding.await_result(context, awaitable)


def _await_result_in_loop(loop: asyncio.AbstractEventLoop) -> AwaitResult:
    """Return a blocking bridge that schedules an awaitable on ``loop`` safely."""

    def await_result(context: Context, awaitable: Awaitable[Any]) -> Any:
        result: Future[Any] = Future()

        def schedule() -> None:
            try:
                task = context.run(asyncio.ensure_future, awaitable)
                if context.run(current_live_input) is not None:
                    task = context.run(asyncio.ensure_future, run_cancellable_binding(task))
            except BaseException as exc:
                result.set_exception(exc)
                return

            def complete(completed: asyncio.Future[Any]) -> None:
                try:
                    result.set_result(completed.result())
                except BaseException as exc:
                    result.set_exception(exc)

            task.add_done_callback(complete)

        loop.call_soon_threadsafe(schedule)
        return result.result()

    return await_result


def _default_native_approval_response(method: str) -> JsonObject:
    if method == "item/permissions/requestApproval":
        return {"permissions": {}, "scope": "turn"}
    return {"decision": "decline"}


def _validated_native_approval_response(method: str, value: Any) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError("Codex native approval handler must return an object response.")
    if method == "item/permissions/requestApproval":
        permissions = value.get("permissions")
        scope = value.get("scope", "turn")
        if not isinstance(permissions, Mapping):
            raise TypeError(
                "Codex permissions approval response requires an object permissions field."
            )
        if scope not in {"turn", "session"}:
            raise ValueError("Codex permissions approval scope must be 'turn' or 'session'.")
        response: JsonObject = {"permissions": dict(permissions), "scope": scope}
        strict_auto_review = value.get("strictAutoReview")
        if strict_auto_review is not None:
            if not isinstance(strict_auto_review, bool):
                raise TypeError("Codex strictAutoReview must be bool when provided.")
            response["strictAutoReview"] = strict_auto_review
        return response

    decision = value.get("decision")
    if decision not in {"accept", "acceptForSession", "decline", "cancel"}:
        raise ValueError(
            "Codex command/file approval response must use accept, acceptForSession, decline, or cancel."
        )
    return {"decision": decision}


def _message_phase(value: object) -> AgentMessagePhase | None:
    if value == "commentary":
        return "commentary"
    if value == "final_answer":
        return "final"
    return None


def _final_turn_message(messages: Sequence[_TurnMessage]) -> _TurnMessage | None:
    for message in reversed(messages):
        if message.phase == "final":
            return message
    for message in reversed(messages):
        if message.phase is None:
            return message
    return None


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
