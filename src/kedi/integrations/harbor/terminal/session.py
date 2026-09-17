from __future__ import annotations

import base64
import math
import os
import shlex
import subprocess
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Sequence

from kedi import ArtifactStream, tool
from kedi.agent_adapter.hooks import HookEvent, PostToolUseEvent
from kedi.integrations.harbor.process_groups import finish_process_group
from kedi.integrations.harbor.records import VerificationState, redact_text, utc_now
from kedi.integrations.harbor.terminal import audit as _audit
from kedi.integrations.harbor.terminal import processes as _processes
from kedi.integrations.harbor.terminal.output import (
    _WRITE_TOOLS,
    OutputStream,
    _BytePreview,
    _empty_binary_file,
    _foreground_result,
    _preview_file,
)
from kedi.integrations.harbor.terminal.processes import (
    _close_process_input,
    _is_read_only_command,
    _is_verification_command,
    _ManagedProcess,
    _sanitized_subprocess_env,
    _validated_argv,
)


def _terminal_tool(*, risk: Literal["read_only", "mutating"]) -> Any:
    def decorate(fn: Any) -> Any:
        @wraps(fn)
        def invoke(self: TerminalSession, *args: Any, **kwargs: Any) -> Any:
            result = fn(self, *args, **kwargs)
            if isinstance(result, dict):
                self._add_time_notice(result)
            return result

        wrapped = tool(risk=risk)(invoke)
        setattr(wrapped, "__kedi_tool_kind__", "terminal")
        return wrapped

    return decorate


class TerminalSession:
    """Tracked terminal tools with explicit session and execution process lifetimes."""

    def __init__(
        self,
        *,
        root: Path,
        records_dir: Path,
        cwd_root: Path | None = None,
        command_timeout_seconds: float = 600.0,
        preview_bytes: int = 32_768,
        output_limit_bytes: int = 512_000_000,
        deadline_monotonic: float | None = None,
        agent_deadline_monotonic: float | None = None,
        finalization_reserve_seconds: float = 0.0,
        artifacts_enabled: bool = True,
        retained_process_timeout_seconds: float = 3_600.0,
    ) -> None:
        self.root = root.resolve()
        self.cwd_root = (cwd_root or self.root).resolve()
        if not self.root.is_relative_to(self.cwd_root):
            raise ValueError("root must be inside cwd_root")
        self.records_dir = records_dir.resolve()
        self.output_dir = self.records_dir / "terminal"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if not math.isfinite(finalization_reserve_seconds) or finalization_reserve_seconds < 0:
            raise ValueError("finalization_reserve_seconds must not be negative")
        self.command_timeout_seconds = command_timeout_seconds
        self.preview_bytes = preview_bytes
        self.output_limit_bytes = output_limit_bytes
        self._execution_deadline_monotonic = deadline_monotonic
        deadlines = [
            value for value in (deadline_monotonic, agent_deadline_monotonic) if value is not None
        ]
        self.deadline_monotonic = min(deadlines) if deadlines else None
        self.finalization_reserve_seconds = finalization_reserve_seconds
        self._time_notice_threshold = (
            min(120.0, max(0.0, self.deadline_monotonic - time.monotonic()) * 0.2)
            if self.deadline_monotonic is not None
            else 0.0
        )
        self._time_notice_sent = False
        self.artifacts_enabled = artifacts_enabled
        if (
            not math.isfinite(retained_process_timeout_seconds)
            or retained_process_timeout_seconds <= 0
        ):
            raise ValueError("retained_process_timeout_seconds must be positive")
        self.retained_process_timeout_seconds = retained_process_timeout_seconds
        self._lock = threading.RLock()
        self._combined_lock = threading.Lock()
        self._processes: dict[str, _ManagedProcess] = {}
        self._next_process_id = 1
        self._closed = False
        self._destroyed = False
        self._verification_state: VerificationState = "not_attempted"
        self._verification_command: str | None = None
        self._verification_exit_code: int | None = None
        self._verification_updated_at: str | None = None
        self._commands: list[dict[str, Any]] = []
        self._events_path = self.records_dir / "terminal-events.jsonl"
        self._tool_events_path = self.records_dir / "tool-events.jsonl"
        self._tool_event_count = 0
        self._tool_failure_count = 0
        self._tool_audit_errors = 0

    def _add_time_notice(self, result: dict[str, Any]) -> None:
        if self.deadline_monotonic is None:
            return
        with self._lock:
            remaining = self.deadline_monotonic - time.monotonic()
            if self._time_notice_sent or remaining > self._time_notice_threshold:
                return
            result["execution_budget"] = {
                "remaining_seconds": max(0, math.floor(remaining)),
                "finalization_reserve_seconds": self.finalization_reserve_seconds,
                "notice": "Execution time is nearly exhausted; the deadline has not changed.",
            }
            self._time_notice_sent = True

    @property
    def tools(self) -> tuple[Any, ...]:
        tools = (
            self.run_command,
            self.run_shell,
            self.start_process,
            self.start_shell,
            self.retain_process,
            self.process_status,
            self.wait_process,
            self.read_process_output,
            self.write_process_input,
            self.stop_process,
            self.list_processes,
            self.verification_status,
        )
        if self.artifacts_enabled:
            return (*tools[:8], self.stream_process_output, *tools[8:])
        return tools

    def hook_handler(self, event: HookEvent) -> None:
        self._record_tool_event(event)
        if isinstance(event, PostToolUseEvent) and event.tool_name in _WRITE_TOOLS:
            self.invalidate_verification(f"{event.tool_name} changed the workspace")

    @_terminal_tool(risk="mutating")
    def run_command(
        self,
        argv: list[str],
        cwd: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one foreground command without shell expansion.

        Use ``start_process`` instead for servers, emulators, daemons, and other
        commands that are expected to keep running after startup.

        Args:
            argv: Exact executable and arguments, such as ["pytest", "-q"].
            cwd: Working directory inside the configured execution scope.
            timeout_seconds: Optional shorter timeout. Omit it for long builds and test
                suites so the configured command ceiling and trial deadline apply.
        """
        normalized = _validated_argv(argv)
        managed = self._start(normalized, cwd=cwd, shell=False, background=False)
        if isinstance(managed, dict):
            return _foreground_result(managed)
        return self._wait(managed, timeout_seconds=timeout_seconds)

    @_terminal_tool(risk="mutating")
    def run_shell(
        self,
        command: str,
        cwd: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run a foreground Bash command when pipes, redirects, or expansion are required.

        Use ``start_shell`` instead for servers, emulators, daemons, and other
        commands that are expected to keep running after startup.
        Omit ``timeout_seconds`` for long builds and test suites so the configured
        command ceiling and trial deadline apply.
        """
        if not command.strip():
            raise ValueError("shell command cannot be empty")
        managed = self._start(command, cwd=cwd, shell=True, background=False)
        if isinstance(managed, dict):
            return _foreground_result(managed)
        return self._wait(managed, timeout_seconds=timeout_seconds)

    @_terminal_tool(risk="mutating")
    def start_process(self, argv: list[str], cwd: str = ".") -> dict[str, Any]:
        """Start a background command without shell expansion.

        Use this for services or finite jobs that will be followed with ``wait_process``.
        """
        managed = self._start(
            _validated_argv(argv),
            cwd=cwd,
            shell=False,
            background=True,
        )
        if isinstance(managed, dict):
            return managed
        self.invalidate_verification("a background process was started")
        return self._snapshot(managed)

    @_terminal_tool(risk="mutating")
    def start_shell(self, command: str, cwd: str = ".") -> dict[str, Any]:
        """Start a background Bash command when shell syntax is required.

        Use this for services or finite jobs that will be followed with ``wait_process``.
        """
        if not command.strip():
            raise ValueError("shell command cannot be empty")
        managed = self._start(command, cwd=cwd, shell=True, background=True)
        if isinstance(managed, dict):
            return managed
        self.invalidate_verification("a background process was started")
        return self._snapshot(managed)

    @_terminal_tool(risk="mutating")
    def retain_process(self, process_id: str) -> dict[str, Any]:
        """Keep a verified background service alive until the outer execution is destroyed.

        Call this only after checking that a service required by later verification is healthy.
        Session processes remain the default and are stopped when the agent finishes.
        """
        managed = self._process(process_id)
        if not managed.background:
            raise ValueError("only background processes can be retained")
        if self._snapshot(managed)["status"] != "running":
            raise RuntimeError(f"process {process_id!r} is no longer running")
        managed.lifetime = "execution"
        self._write_supervisor_control(managed, retained=True)
        self._write_event(
            {
                "event": "process_retention_changed",
                "process_id": process_id,
                "lifetime": managed.lifetime,
            }
        )
        return self._snapshot(managed)

    @_terminal_tool(risk="read_only")
    def process_status(self, process_id: str) -> dict[str, Any]:
        """Return current status, exit code, output sizes, and previews for a process."""
        return self._snapshot(self._process(process_id))

    @_terminal_tool(risk="read_only")
    def wait_process(
        self,
        process_id: str,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Wait boundedly for a background process without stopping it on timeout.

        Use this for builds and other finite jobs started with ``start_process`` or
        ``start_shell``. The result returns immediately when the process finishes;
        otherwise it reports compact progress after the requested wait. Use
        ``process_status`` separately only when partial output must be inspected.
        """
        managed = self._process(process_id)
        if not managed.background:
            raise ValueError("only background processes can be waited on")
        timeout = self._bounded_timeout(timeout_seconds)
        wait_started = time.monotonic()
        try:
            managed.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        else:
            finish_process_group(managed.target_pgid or managed.process.pid)
            self._join_readers(managed)
            self._record_process(managed)
            self._update_verification(managed)
        snapshot = self._snapshot(managed)
        waited_seconds = round(time.monotonic() - wait_started, 3)
        if snapshot["status"] == "running":
            return {
                "process_id": process_id,
                "status": "running",
                "wait_completed": False,
                "waited_seconds": waited_seconds,
                "duration_seconds": round(float(snapshot["duration_seconds"]), 3),
                "stdout_bytes": snapshot["stdout_bytes"],
                "stderr_bytes": snapshot["stderr_bytes"],
            }
        result = _foreground_result(snapshot)
        result["status"] = snapshot["status"]
        result["wait_completed"] = True
        result["waited_seconds"] = waited_seconds
        return result

    @_terminal_tool(risk="read_only")
    def read_process_output(
        self,
        process_id: str,
        stream: OutputStream = "combined",
        offset_bytes: int = 0,
        max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        """Read a bounded byte range from a process output log.

        Binary data is returned as base64 while text data is returned as UTF-8.
        When a terminal result reports a truncated stream, continue from the exact
        process, stream, and offset arguments returned with that result instead of
        rerunning the command.
        """
        if offset_bytes < 0:
            raise ValueError("offset_bytes must not be negative")
        if max_bytes <= 0 or max_bytes > 1_000_000:
            raise ValueError("max_bytes must be between 1 and 1,000,000")
        path = self._output_path(self._process(process_id), stream)
        size = path.stat().st_size if path.exists() else 0
        source = path.open("rb") if path.exists() else _empty_binary_file()
        with source as handle:
            handle.seek(offset_bytes)
            payload = handle.read(max_bytes)
        next_offset = offset_bytes + len(payload)
        complete = next_offset >= size
        try:
            text = payload.decode("utf-8")
            encoding = "utf-8"
            content = text
        except UnicodeDecodeError:
            encoding = "base64"
            content = base64.b64encode(payload).decode("ascii")
        return {
            "process_id": process_id,
            "stream": stream,
            "encoding": encoding,
            "content": content,
            "offset_bytes": offset_bytes,
            "next_offset_bytes": next_offset,
            "total_bytes": size,
            "complete": complete,
        }

    @_terminal_tool(risk="read_only")
    def stream_process_output(
        self,
        process_id: str,
        stream: OutputStream = "combined",
    ) -> ArtifactStream[str]:
        """Load a complete process log through Kedi's artifact transport."""
        path = self._output_path(self._process(process_id), stream)

        def chunks() -> Any:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                while chunk := handle.read(65_536):
                    yield chunk

        return ArtifactStream.text(
            chunks(),
            content_length=path.stat().st_size,
            metadata={"process_id": process_id, "stream": stream},
        )

    @_terminal_tool(risk="mutating")
    def write_process_input(
        self,
        process_id: str,
        content: str,
        append_newline: bool = False,
    ) -> dict[str, Any]:
        """Write UTF-8 input to a running background process."""
        managed = self._process(process_id)
        if managed.process.poll() is not None or managed.process.stdin is None:
            raise RuntimeError(f"process {process_id!r} is not accepting input")
        payload = (content + ("\n" if append_newline else "")).encode("utf-8")
        managed.process.stdin.write(payload)
        managed.process.stdin.flush()
        self.invalidate_verification("input was sent to a background process")
        self._write_event(
            {"event": "process_input", "process_id": process_id, "size_bytes": len(payload)}
        )
        return {"process_id": process_id, "written_bytes": len(payload)}

    @_terminal_tool(risk="mutating")
    def stop_process(self, process_id: str, grace_seconds: float = 2.0) -> dict[str, Any]:
        """Terminate a tracked process tree, escalating to SIGKILL if needed."""
        if grace_seconds < 0 or grace_seconds > 30:
            raise ValueError("grace_seconds must be between 0 and 30")
        managed = self._process(process_id)
        self._terminate(managed, grace_seconds=grace_seconds)
        self.invalidate_verification("a background process was stopped")
        return self._snapshot(managed)

    @_terminal_tool(risk="read_only")
    def list_processes(self) -> list[dict[str, Any]]:
        """List all foreground and background processes tracked by this execution."""
        with self._lock:
            return [self._snapshot(item) for item in self._processes.values()]

    @_terminal_tool(risk="read_only")
    def verification_status(self) -> dict[str, Any]:
        """Return the latest evidence-backed verification state."""
        with self._lock:
            return {
                "state": self._verification_state,
                "command": self._verification_command,
                "exit_code": self._verification_exit_code,
                "updated_at": self._verification_updated_at,
            }

    def invalidate_verification(self, reason: str) -> None:
        with self._lock:
            if self._verification_state in {"passed", "failed", "unavailable"}:
                self._verification_state = "stale"
                self._verification_updated_at = utc_now()
                self._write_event({"event": "verification_invalidated", "reason": reason})

    def summary(self) -> dict[str, Any]:
        with self._lock:
            snapshots = [self._snapshot(item) for item in self._processes.values()]
            running = sum(item["status"] == "running" for item in snapshots)
            return {
                "commands": len(self._commands),
                "processes": len(self._processes),
                "running_processes": running,
                "retained_processes": sum(
                    item["status"] == "running" and item["lifetime"] == "execution"
                    for item in snapshots
                ),
                "verification": self.verification_status(),
                "tool_events": self._tool_event_count,
                "tool_failures": self._tool_failure_count,
                "tool_audit_errors": self._tool_audit_errors,
                "tool_events_path": str(self._tool_events_path),
                "records": list(self._commands),
            }

    def finish_session(self) -> None:
        """Close agent-owned resources while handing retained services to the execution."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            processes = tuple(self._processes.values())
        for managed in processes:
            snapshot = self._snapshot(managed)
            if snapshot["status"] != "running":
                self._join_readers(managed)
                self._record_process(managed)
                continue
            if managed.background and managed.lifetime == "execution":
                managed.handed_off = True
                _close_process_input(managed.process)
                self._record_retained_process(managed)
                continue
            self._terminate(managed, grace_seconds=1.0)
        self._flush_tool_events()

    def close(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            self._closed = True
            self._destroyed = True
            processes = tuple(self._processes.values())
        for managed in processes:
            self._write_supervisor_control(managed, retained=False)
            if self._snapshot(managed)["status"] == "running":
                self._terminate(managed, grace_seconds=1.0)
            else:
                self._join_readers(managed)
                self._record_process(managed)
        self._flush_tool_events()

    def _start(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str,
        shell: bool,
        background: bool,
    ) -> _ManagedProcess | dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("terminal session is closed")
            process_id = f"process_{self._next_process_id}"
            self._next_process_id += 1
        resolved_cwd = self._safe_cwd(cwd)
        display = command if isinstance(command, str) else shlex.join(command)
        process_dir = self.output_dir / process_id
        process_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = process_dir / "stdout.bin"
        stderr_path = process_dir / "stderr.bin"
        combined_path = process_dir / "combined.bin"
        stdout_path.touch()
        stderr_path.touch()
        combined_path.touch()
        popen_command: Sequence[str]
        if shell:
            popen_command = ("/bin/bash", "-c", str(command))
        else:
            popen_command = command
        if background:
            return self._start_supervised(
                process_id=process_id,
                command=redact_text(str(display)),
                popen_command=popen_command,
                shell=shell,
                cwd=resolved_cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                combined_path=combined_path,
            )
        try:
            process = subprocess.Popen(
                popen_command,
                cwd=resolved_cwd,
                env=_sanitized_subprocess_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return self._record_start_failure(
                process_id=process_id,
                command=redact_text(str(display)),
                shell=shell,
                cwd=resolved_cwd,
                started_at=utc_now(),
                started_monotonic=time.monotonic(),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                combined_path=combined_path,
                error=exc,
            )
        managed = _ManagedProcess(
            process_id=process_id,
            command=redact_text(str(display)),
            shell=shell,
            cwd=resolved_cwd,
            process=process,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_path=combined_path,
            stdout_preview=_BytePreview(self.preview_bytes),
            stderr_preview=_BytePreview(self.preview_bytes),
            background=False,
        )
        with self._lock:
            self._processes[process_id] = managed
        assert process.stdout is not None and process.stderr is not None
        managed.threads = [
            self._reader_thread(managed, "stdout", process.stdout),
            self._reader_thread(managed, "stderr", process.stderr),
        ]
        for thread in managed.threads:
            thread.start()
        self._write_event(
            {
                "event": "process_started",
                "process_id": process_id,
                "pid": process.pid,
                "command": managed.command,
                "cwd": str(resolved_cwd),
                "lifetime": managed.lifetime,
            }
        )
        return managed

    def _start_supervised(
        self,
        *,
        process_id: str,
        command: str,
        popen_command: Sequence[str],
        shell: bool,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        combined_path: Path,
    ) -> _ManagedProcess | dict[str, Any]:
        return _processes._start_supervised(
            self,
            process_id=process_id,
            command=command,
            popen_command=popen_command,
            shell=shell,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_path=combined_path,
        )

    def _await_supervisor_start(
        self,
        process: subprocess.Popen[bytes],
        state_path: Path,
    ) -> dict[str, Any]:
        return _processes._await_supervisor_start(self, process, state_path)

    def _record_start_failure(
        self,
        *,
        process_id: str,
        command: str,
        shell: bool,
        cwd: Path,
        started_at: str,
        started_monotonic: float,
        stdout_path: Path,
        stderr_path: Path,
        combined_path: Path,
        error: OSError,
    ) -> dict[str, Any]:
        error_message = redact_text(str(error))
        stderr = f"could not start command: {error_message}\n"
        payload = stderr.encode("utf-8")
        for path in (stderr_path, combined_path):
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        record = {
            "process_id": process_id,
            "pid": None,
            "supervisor_pid": None,
            "command": command,
            "shell": shell,
            "cwd": str(cwd),
            "status": "failed_to_start",
            "exit_code": 127 if isinstance(error, FileNotFoundError) else 126,
            "timed_out": False,
            "stopped": False,
            "output_limited": False,
            "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
            "lifetime": "session",
            "retained": False,
            "handed_off": False,
            "stdout": "",
            "stderr": stderr,
            "stdout_bytes": 0,
            "stderr_bytes": len(payload),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_invalid_utf8": False,
            "stderr_invalid_utf8": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "combined_path": str(combined_path),
            "started_at": started_at,
            "start_error_type": type(error).__name__,
        }
        with self._lock:
            self._commands.append(record)
            if _is_verification_command(command, shell=shell):
                self._verification_state = "failed"
                self._verification_command = command
                self._verification_exit_code = int(record["exit_code"])
                self._verification_updated_at = utc_now()
        self._write_event({"event": "process_start_failed", **record})
        if _is_verification_command(command, shell=shell):
            self._write_event({"event": "verification_updated", **self.verification_status()})
        return record

    def _reader_thread(
        self,
        managed: _ManagedProcess,
        stream: Literal["stdout", "stderr"],
        source: Any,
    ) -> threading.Thread:
        return _processes._reader_thread(self, managed, stream, source)

    def _wait(
        self,
        managed: _ManagedProcess,
        *,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        return _processes._wait(self, managed, timeout_seconds=timeout_seconds)

    def _terminate(self, managed: _ManagedProcess, *, grace_seconds: float) -> None:
        return _processes._terminate(self, managed, grace_seconds=grace_seconds)

    def _terminate_for_output_limit(self, managed: _ManagedProcess) -> None:
        return _processes._terminate_for_output_limit(self, managed)

    def _join_readers(self, managed: _ManagedProcess) -> None:
        return _processes._join_readers(self, managed)

    def _record_process(self, managed: _ManagedProcess) -> None:
        with self._lock:
            if managed.recorded:
                return
            managed.recorded = True
            record = self._snapshot(managed)
            self._commands.append(record)
        self._write_event({"event": "process_finished", **record})

    def _record_retained_process(self, managed: _ManagedProcess) -> None:
        with self._lock:
            if managed.recorded:
                return
            managed.recorded = True
            record = self._snapshot(managed)
            self._commands.append(record)
        self._write_event({"event": "process_handed_off", **record})

    def _snapshot(self, managed: _ManagedProcess) -> dict[str, Any]:
        state = self._refresh_supervisor_state(managed)
        state_phase = state.get("phase")
        state_returncode = state.get("returncode")
        exit_code = managed.process.poll()
        if state_phase in {"finished", "failed_to_start", "supervisor_failure"} and isinstance(
            state_returncode, int
        ):
            exit_code = state_returncode
        if exit_code is not None and managed.finished_monotonic is None:
            finished = state.get("finished_monotonic")
            managed.finished_monotonic = (
                float(finished) if isinstance(finished, (int, float)) else time.monotonic()
            )
        if managed.background:
            stdout, stdout_truncated, stdout_invalid, stdout_bytes = _preview_file(
                managed.stdout_path,
                self.preview_bytes,
            )
            stderr, stderr_truncated, stderr_invalid, stderr_bytes = _preview_file(
                managed.stderr_path,
                self.preview_bytes,
            )
        else:
            stdout, stdout_truncated, stdout_invalid = managed.stdout_preview.render()
            stderr, stderr_truncated, stderr_invalid = managed.stderr_preview.render()
            stdout_bytes = managed.stdout_preview.total_bytes
            stderr_bytes = managed.stderr_preview.total_bytes
        snapshot: dict[str, Any] = {
            "process_id": managed.process_id,
            "pid": managed.target_pid or managed.process.pid,
            "supervisor_pid": managed.process.pid if managed.background else None,
            "command": managed.command,
            "shell": managed.shell,
            "cwd": str(managed.cwd),
            "status": "running" if exit_code is None else "finished",
            "exit_code": exit_code,
            "timed_out": managed.timed_out,
            "stopped": managed.stopped,
            "output_limited": managed.output_limited,
            "duration_seconds": max(
                0.0,
                (managed.finished_monotonic or time.monotonic()) - managed.started_monotonic,
            ),
            "lifetime": managed.lifetime,
            "retained": managed.lifetime == "execution",
            "handed_off": managed.handed_off,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_invalid_utf8": stdout_invalid,
            "stderr_invalid_utf8": stderr_invalid,
            "stdout_path": str(managed.stdout_path),
            "stderr_path": str(managed.stderr_path),
            "combined_path": str(managed.combined_path),
            "started_at": managed.started_at,
        }
        for stream in ("stdout", "stderr"):
            if not snapshot[f"{stream}_truncated"]:
                continue
            # Metadata sorts before stdout/stderr in the artifact JSON preview.
            snapshot["output_excerpt"] = "head_tail"
            snapshot.setdefault("output_continuations", {})[stream] = {
                "tool": "read_process_output",
                "arguments": {
                    "process_id": managed.process_id,
                    "stream": stream,
                    "offset_bytes": 0,
                    "max_bytes": 65_536,
                },
            }
        return snapshot

    def _refresh_supervisor_state(self, managed: _ManagedProcess) -> dict[str, Any]:
        return _processes._refresh_supervisor_state(self, managed)

    def _write_supervisor_control(self, managed: _ManagedProcess, *, retained: bool) -> None:
        return _processes._write_supervisor_control(self, managed, retained=retained)

    def _update_verification(self, managed: _ManagedProcess) -> None:
        with self._lock:
            if managed.verification_recorded:
                return
            snapshot = self._snapshot(managed)
            if snapshot["status"] == "running":
                return
            managed.verification_recorded = True
            if not _is_verification_command(managed.command, shell=managed.shell):
                if not _is_read_only_command(managed.command, shell=managed.shell):
                    self.invalidate_verification("a command may have changed the workspace")
                return
            self._verification_state = (
                "passed"
                if snapshot["exit_code"] == 0
                and not managed.timed_out
                and not managed.output_limited
                else "failed"
            )
            self._verification_command = managed.command
            self._verification_exit_code = snapshot["exit_code"]
            self._verification_updated_at = utc_now()
        self._write_event({"event": "verification_updated", **self.verification_status()})

    def _safe_cwd(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.cwd_root):
            raise PermissionError(f"working directory escapes allowed cwd root: {raw}")
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        return resolved

    @staticmethod
    def _signal_process_group(managed: _ManagedProcess, signum: int) -> None:
        groups = [getattr(managed, "target_pgid", None), managed.process.pid]
        for process_group in dict.fromkeys(groups):
            if not isinstance(process_group, int):
                continue
            TerminalSession._signal_pid_group(process_group, signum)

    @staticmethod
    def _signal_pid_group(process_group: int, signum: int) -> None:
        try:
            os.killpg(process_group, signum)
        except (PermissionError, ProcessLookupError):
            pass

    def _bounded_timeout(self, requested: float | None) -> float:
        if requested is not None and (not math.isfinite(requested) or requested <= 0):
            raise ValueError("timeout_seconds must be positive")
        timeout = (
            self.command_timeout_seconds
            if requested is None
            else min(self.command_timeout_seconds, requested)
        )
        if self.deadline_monotonic is not None:
            remaining = self.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("trial deadline has elapsed")
            command_window = remaining - self.finalization_reserve_seconds
            if command_window <= 0:
                raise TimeoutError("trial finalization reserve has begun")
            timeout = min(timeout, command_window)
        return timeout

    def _process(self, process_id: str) -> _ManagedProcess:
        with self._lock:
            try:
                return self._processes[process_id]
            except KeyError as exc:
                raise KeyError(f"unknown session process {process_id!r}") from exc

    @staticmethod
    def _output_path(managed: _ManagedProcess, stream: OutputStream) -> Path:
        if stream == "stdout":
            return managed.stdout_path
        if stream == "stderr":
            return managed.stderr_path
        if stream == "combined":
            return managed.combined_path
        raise ValueError("stream must be stdout, stderr, or combined")

    def _write_event(self, event: dict[str, Any]) -> None:
        return _audit._write_event(self, event)

    def _record_tool_event(self, event: HookEvent) -> None:
        return _audit._record_tool_event(self, event)

    def _flush_tool_events(self) -> None:
        return _audit._flush_tool_events(self)
