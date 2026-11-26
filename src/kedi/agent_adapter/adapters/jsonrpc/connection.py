"""One bridge process and its JSON-RPC stream over stdio.

A reader thread owns stdout for the life of the connection, so responses resolve while
handlers wait on callbacks, and no event loop's lifetime bounds the connection. Requests
from the bridge go to the adapter's handler, which answers later with :meth:`respond`;
a bounded number may be in flight. A connection is one generation: once it disconnects,
every waiter fails and nothing on it completes again.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from threading import BoundedSemaphore, Lock, Thread
from typing import IO, Any

from .protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    MAX_FRAME_BYTES,
    OVERLOADED,
    JsonObject,
    JsonRpcDisconnected,
    JsonRpcError,
)

RequestHandler = Callable[[int | str, str, JsonObject], None]
"""Handles one request from the bridge; must eventually call ``Connection.respond``."""

_STDERR_LINES = 50


class Connection:
    """A launched bridge process: Kedi owns it and stops it on :meth:`close`."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        handler: RequestHandler,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        max_inbound_requests: int = 64,
    ) -> None:
        if not command:
            raise ValueError("JSON-RPC bridge command cannot be empty.")
        self._handler = handler
        self._write_lock = Lock()
        self._state_lock = Lock()
        self._pending: dict[int, Future[JsonObject]] = {}
        self._inbound: set[int | str] = set()
        self._slots = BoundedSemaphore(max_inbound_requests)
        self._next_id = 1
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        self._disconnected: JsonRpcDisconnected | None = None
        try:
            self._process = subprocess.Popen(
                tuple(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=None if env is None else {**os.environ, **env},
                cwd=cwd,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"JSON-RPC bridge executable not found: {command[0]}") from exc
        assert self._process.stdout is not None and self._process.stderr is not None
        self._reader = Thread(
            target=self._read, args=(self._process.stdout,), name="kedi-jsonrpc", daemon=True
        )
        self._stderr_reader = Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            name="kedi-jsonrpc-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()

    @property
    def disconnected(self) -> JsonRpcDisconnected | None:
        """Why the connection ended, or ``None`` while it is open."""

        return self._disconnected

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        """The bridge's last stderr lines, for diagnostics."""

        with self._state_lock:
            return tuple(self._stderr)

    def start_request(self, method: str, params: JsonObject) -> tuple[int, Future[JsonObject]]:
        """Send a request; its future resolves with the result or fails with the error."""

        future: Future[JsonObject] = Future()
        with self._state_lock:
            if self._disconnected is not None:
                raise self._disconnected
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = future
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        except BaseException:
            self.forget(request_id)
            raise
        return request_id, future

    def forget(self, request_id: int) -> None:
        """Stop waiting for a response; a late one is ignored."""

        with self._state_lock:
            self._pending.pop(request_id, None)

    def request_sync(self, method: str, params: JsonObject, *, timeout: float | None) -> JsonObject:
        request_id, future = self.start_request(method, params)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"JSON-RPC request {method!r} timed out after {timeout}s") from exc
        finally:
            self.forget(request_id)

    async def request(
        self, method: str, params: JsonObject, *, timeout: float | None = None
    ) -> JsonObject:
        request_id, future = self.start_request(method, params)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"JSON-RPC request {method!r} timed out after {timeout}s") from exc
        finally:
            self.forget(request_id)

    def notify(self, method: str, params: JsonObject) -> None:
        """Send a notification; it has no answer, so a closed connection drops it."""

        with suppress(JsonRpcDisconnected):
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def respond(
        self, request_id: int | str, *, result: Any = None, error: JsonRpcError | None = None
    ) -> None:
        """Answer a request from the bridge and free its slot."""

        with self._state_lock:
            if request_id not in self._inbound:
                return  # answered already
            self._inbound.discard(request_id)
        message: JsonObject = {"jsonrpc": "2.0", "id": request_id}
        if error is None:
            message["result"] = result
        else:
            message["error"] = error.to_json()
        try:
            self._send(message)
        except JsonRpcDisconnected:
            pass  # the bridge is gone; nobody waits for this answer
        except JsonRpcError as exc:  # the answer itself is over the frame limit
            self._send_error(request_id, exc.code, str(exc))
        finally:
            self._slots.release()

    def close(self, *, timeout: float = 5) -> None:
        """Ask the bridge to shut down, then stop the process; bounded by ``timeout``."""

        process = self._process
        if self._disconnected is None and process.poll() is None:
            try:
                self.request_sync("shutdown", {}, timeout=timeout)
            except (JsonRpcError, TimeoutError):
                pass  # still stopped below
        self._disconnect("The JSON-RPC bridge connection is closed.")
        assert process.stdin is not None
        with suppress(OSError):
            process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            process.wait(timeout=timeout)
        self._reader.join(timeout=timeout)
        self._stderr_reader.join(timeout=timeout)

    def _signal(self, number: int) -> None:
        if os.name == "posix":
            with suppress(ProcessLookupError):  # it exited meanwhile
                os.killpg(self._process.pid, number)
        else:  # pragma: no cover - Windows has no process groups
            self._process.kill()

    def _send(self, message: JsonObject) -> None:
        frame = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        if len(frame) > MAX_FRAME_BYTES:
            raise JsonRpcError(
                INVALID_REQUEST,
                f"JSON-RPC message of {len(frame)} bytes exceeds the {MAX_FRAME_BYTES}-byte frame limit",
            )
        stdin = self._process.stdin
        assert stdin is not None
        with self._write_lock:
            if self._disconnected is not None:
                raise self._disconnected
            try:
                stdin.write(frame)
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                error = self._disconnect("The JSON-RPC bridge closed its stdin.")
                raise error from None

    def _read(self, stdout: IO[bytes]) -> None:
        while True:
            line = stdout.readline(MAX_FRAME_BYTES + 1)
            if not line:
                # Its last diagnostics explain why it exited.
                self._stderr_reader.join(timeout=1)
                self._disconnect("The JSON-RPC bridge exited or closed its stdout.")
                return
            if len(line) > MAX_FRAME_BYTES:
                self._disconnect(
                    f"The JSON-RPC bridge sent a frame over the {MAX_FRAME_BYTES}-byte limit."
                )
                return
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self._disconnect(
                    "The JSON-RPC bridge wrote a line that is not JSON to stdout; stdout carries "
                    f"only protocol frames: {line[:200].decode(errors='replace')!r}"
                )
                return
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                self._disconnect(
                    f"The JSON-RPC bridge sent a message that is not JSON-RPC 2.0: {line[:200]!r}"
                )
                return
            try:
                self._dispatch(message)
            except Exception as exc:  # noqa: BLE001 - the reader must not die silently
                self._disconnect(f"Kedi could not handle a bridge message: {exc!r}")
                return

    def _dispatch(self, message: JsonObject) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if isinstance(method, str):
            if request_id is None:
                return  # protocol version 1 defines no notifications to Kedi
            self._admit(request_id, method, message.get("params"))
            return
        with self._state_lock:
            future = self._pending.pop(request_id, None) if isinstance(request_id, int) else None
        if future is None:
            return  # a late response to a request Kedi stopped waiting for
        error, result = message.get("error"), message.get("result")
        with suppress(InvalidStateError):  # the waiter gave up (timeout or cancellation)
            if isinstance(error, dict):
                code = error.get("code")
                future.set_exception(
                    JsonRpcError(
                        code if isinstance(code, int) else INTERNAL_ERROR,
                        str(error.get("message", "")),
                        error.get("data"),
                    )
                )
            elif isinstance(result, dict):
                future.set_result(result)
            else:
                future.set_exception(
                    JsonRpcError(
                        INVALID_REQUEST, "The JSON-RPC bridge answered without an object result."
                    )
                )

    def _admit(self, request_id: Any, method: str, params: Any) -> None:
        if not isinstance(request_id, (int, str)) or not isinstance(params, dict):
            self._send_error(
                request_id, INVALID_REQUEST, "A request needs an id and object params."
            )
            return
        if not self._slots.acquire(blocking=False):
            self._send_error(request_id, OVERLOADED, "Kedi has no capacity for another request.")
            return
        with self._state_lock:
            self._inbound.add(request_id)
        self._handler(request_id, method, params)

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        error = {"code": code, "message": message}
        with suppress(JsonRpcDisconnected):
            self._send({"jsonrpc": "2.0", "id": request_id, "error": error})

    def _drain_stderr(self, stderr: IO[bytes]) -> None:
        # Drained to the end, so diagnostics cannot fill the pipe and stall the bridge.
        while line := stderr.readline(64 * 1024):
            with self._state_lock:
                self._stderr.append(line.decode(errors="replace").rstrip())

    def _disconnect(self, reason: str) -> JsonRpcDisconnected:
        with self._state_lock:
            if self._disconnected is None:
                tail = "\n".join(self._stderr)
                self._disconnected = JsonRpcDisconnected(
                    f"{reason}\nBridge stderr:\n{tail}" if tail else reason
                )
            error = self._disconnected
            pending, self._pending = self._pending, {}
        for future in pending.values():
            with suppress(InvalidStateError):  # cancelled by a waiter that gave up
                future.set_exception(error)
        return error


__all__ = ["Connection", "RequestHandler"]
