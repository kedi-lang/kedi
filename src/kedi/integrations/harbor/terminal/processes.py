from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from kedi.integrations.harbor.process_groups import finish_process_group
from kedi.integrations.harbor.records import atomic_write_json, is_secret_name, utc_now
from kedi.integrations.harbor.terminal.output import (
    _NON_EXECUTING_VERIFICATION_FLAGS,
    _READ_ONLY_COMMANDS,
    _READ_ONLY_GIT_COMMANDS,
    _VERIFICATION_COMMANDS,
    ProcessLifetime,
    _BytePreview,
    _foreground_result,
)

if TYPE_CHECKING:
    from kedi.integrations.harbor.terminal.session import TerminalSession


@dataclass(slots=True)
class _ManagedProcess:
    process_id: str
    command: str
    shell: bool
    cwd: Path
    process: subprocess.Popen[bytes]
    started_at: str
    started_monotonic: float
    stdout_path: Path
    stderr_path: Path
    combined_path: Path
    stdout_preview: _BytePreview
    stderr_preview: _BytePreview
    background: bool = False
    lifetime: ProcessLifetime = "session"
    supervisor_state_path: Path | None = None
    supervisor_control_path: Path | None = None
    target_pid: int | None = None
    target_pgid: int | None = None
    retention_deadline_monotonic: float | None = None
    handed_off: bool = False
    threads: list[threading.Thread] = field(default_factory=list)
    finished_monotonic: float | None = None
    timed_out: bool = False
    stopped: bool = False
    output_limited: bool = False
    limit_termination_started: bool = False
    recorded: bool = False
    verification_recorded: bool = False


def _start_supervised(
    self: TerminalSession,
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
    started_at = utc_now()
    started_monotonic = time.monotonic()
    state_path = stdout_path.parent / "state.json"
    control_path = stdout_path.parent / "control.json"
    retention_deadline = started_monotonic + self.retained_process_timeout_seconds
    if self._execution_deadline_monotonic is not None:
        retention_deadline = min(retention_deadline, self._execution_deadline_monotonic)
    atomic_write_json(control_path, {"retained": False})
    config = {
        "command": list(popen_command),
        "cwd": str(cwd),
        "state_path": str(state_path),
        "control_path": str(control_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "combined_path": str(combined_path),
        "output_limit_bytes": self.output_limit_bytes,
        "deadline_monotonic": retention_deadline,
    }
    try:
        process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "kedi.integrations.harbor.process_supervisor",
            ),
            cwd=cwd,
            env=_sanitized_subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return self._record_start_failure(
            process_id=process_id,
            command=command,
            shell=shell,
            cwd=cwd,
            started_at=started_at,
            started_monotonic=started_monotonic,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_path=combined_path,
            error=exc,
        )
    assert process.stdin is not None
    try:
        process.stdin.write(
            (json.dumps(config, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
        process.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    state = self._await_supervisor_start(process, state_path)
    phase = state.get("phase")
    if phase in {"failed_to_start", "supervisor_failure"}:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._signal_pid_group(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        error = _supervisor_start_error(state)
        return self._record_start_failure(
            process_id=process_id,
            command=command,
            shell=shell,
            cwd=cwd,
            started_at=started_at,
            started_monotonic=started_monotonic,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_path=combined_path,
            error=error,
        )
    target_pid = state.get("target_pid")
    target_pgid = state.get("target_pgid")
    if not isinstance(target_pid, int) or not isinstance(target_pgid, int):
        self._signal_pid_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._signal_pid_group(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        return self._record_start_failure(
            process_id=process_id,
            command=command,
            shell=shell,
            cwd=cwd,
            started_at=started_at,
            started_monotonic=started_monotonic,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            combined_path=combined_path,
            error=OSError("background process supervisor did not initialize"),
        )
    managed = _ManagedProcess(
        process_id=process_id,
        command=command,
        shell=shell,
        cwd=cwd,
        process=process,
        started_at=started_at,
        started_monotonic=started_monotonic,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        combined_path=combined_path,
        stdout_preview=_BytePreview(self.preview_bytes),
        stderr_preview=_BytePreview(self.preview_bytes),
        background=True,
        supervisor_state_path=state_path,
        supervisor_control_path=control_path,
        target_pid=target_pid,
        target_pgid=target_pgid,
        retention_deadline_monotonic=retention_deadline,
    )
    with self._lock:
        self._processes[process_id] = managed
    self._write_event(
        {
            "event": "process_started",
            "process_id": process_id,
            "pid": target_pid,
            "supervisor_pid": process.pid,
            "command": command,
            "cwd": str(cwd),
            "lifetime": managed.lifetime,
        }
    )
    return managed


def _await_supervisor_start(
    self: TerminalSession,
    process: subprocess.Popen[bytes],
    state_path: Path,
) -> dict[str, Any]:
    wait_deadline = time.monotonic() + 5.0
    if self.deadline_monotonic is not None:
        wait_deadline = min(wait_deadline, self.deadline_monotonic)
    while time.monotonic() < wait_deadline:
        state = _read_json_object(state_path)
        if state.get("phase") in {
            "running",
            "finished",
            "failed_to_start",
            "supervisor_failure",
        }:
            return state
        if process.poll() is not None:
            return state
        time.sleep(0.01)
    return _read_json_object(state_path)


def _reader_thread(
    self: TerminalSession,
    managed: _ManagedProcess,
    stream: Literal["stdout", "stderr"],
    source: Any,
) -> threading.Thread:
    path = managed.stdout_path if stream == "stdout" else managed.stderr_path
    preview = managed.stdout_preview if stream == "stdout" else managed.stderr_preview

    def read() -> None:
        written = 0
        with path.open("ab", buffering=0) as output:
            read_chunk = getattr(source, "read1", source.read)
            while chunk := read_chunk(65_536):
                allowed = max(0, self.output_limit_bytes - written)
                kept = chunk[:allowed]
                if kept:
                    output.write(kept)
                    preview.append(kept)
                    with self._combined_lock:
                        with managed.combined_path.open("ab", buffering=0) as combined:
                            combined.write(kept)
                    written += len(kept)
                if len(kept) < len(chunk):
                    self._write_event(
                        {
                            "event": "output_limit_reached",
                            "process_id": managed.process_id,
                            "stream": stream,
                            "limit_bytes": self.output_limit_bytes,
                        }
                    )
                    self._terminate_for_output_limit(managed)
                    break
            os.fsync(output.fileno())
        with managed.combined_path.open("ab", buffering=0) as combined:
            os.fsync(combined.fileno())
        source.close()

    return threading.Thread(
        target=read,
        name=f"kedi-{managed.process_id}-{stream}",
        daemon=True,
    )


def _wait(
    self: TerminalSession,
    managed: _ManagedProcess,
    *,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    timeout = self._bounded_timeout(timeout_seconds)
    try:
        managed.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        managed.timed_out = True
        self._terminate(managed, grace_seconds=1.0)
    else:
        # A departed shell can leave descendants holding its output pipes open.
        finish_process_group(managed.process.pid)
    self._join_readers(managed)
    self._record_process(managed)
    self._update_verification(managed)
    return _foreground_result(self._snapshot(managed))


def _terminate(self: TerminalSession, managed: _ManagedProcess, *, grace_seconds: float) -> None:
    self._write_supervisor_control(managed, retained=False)
    if self._snapshot(managed)["status"] == "running":
        self._signal_process_group(managed, signal.SIGTERM)
        try:
            managed.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self._signal_process_group(managed, signal.SIGKILL)
            managed.process.wait(timeout=5)
        finish_process_group(managed.target_pgid or managed.process.pid)
    if managed.finished_monotonic is None:
        managed.finished_monotonic = time.monotonic()
    managed.stopped = True
    _close_process_input(managed.process)
    self._join_readers(managed)
    self._record_process(managed)


def _terminate_for_output_limit(self: TerminalSession, managed: _ManagedProcess) -> None:
    with self._lock:
        managed.output_limited = True
        if managed.limit_termination_started:
            return
        managed.limit_termination_started = True
    if managed.process.poll() is None:
        self._signal_process_group(managed, signal.SIGTERM)
        try:
            managed.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._signal_process_group(managed, signal.SIGKILL)
            managed.process.wait(timeout=5.0)
    managed.stopped = True
    if managed.finished_monotonic is None:
        managed.finished_monotonic = time.monotonic()


def _join_readers(self: TerminalSession, managed: _ManagedProcess) -> None:
    for thread in managed.threads:
        if thread is threading.current_thread():
            continue
        thread.join(timeout=5)


def _refresh_supervisor_state(self: TerminalSession, managed: _ManagedProcess) -> dict[str, Any]:
    if managed.supervisor_state_path is None:
        return {}
    state = _read_json_object(managed.supervisor_state_path)
    target_pid = state.get("target_pid")
    target_pgid = state.get("target_pgid")
    if isinstance(target_pid, int):
        managed.target_pid = target_pid
    if isinstance(target_pgid, int):
        managed.target_pgid = target_pgid
    if state.get("timed_out") is True:
        managed.timed_out = True
    if state.get("output_limited") is True:
        managed.output_limited = True
    return state


def _write_supervisor_control(
    self: TerminalSession, managed: _ManagedProcess, *, retained: bool
) -> None:
    if managed.supervisor_control_path is None:
        return
    atomic_write_json(managed.supervisor_control_path, {"retained": retained})


def _validated_argv(argv: list[str]) -> tuple[str, ...]:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("argv must contain at least one non-empty string")
    return tuple(argv)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _supervisor_start_error(state: dict[str, Any]) -> OSError:
    error_type = state.get("start_error_type")
    message = str(state.get("start_error_message") or "background process failed to start")
    if error_type == "FileNotFoundError":
        return FileNotFoundError(message)
    if error_type == "PermissionError":
        return PermissionError(message)
    return OSError(message)


def _close_process_input(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.close()
    except OSError:
        pass


def _sanitized_subprocess_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not is_secret_name(name)}


def _is_verification_command(command: str, *, shell: bool) -> bool:
    if shell and re.search(r"[;&|<>`]|\$\(|\n", command):
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    if not arguments:
        return False
    executable = Path(arguments[0]).name.lower()
    if re.fullmatch(r"python(?:3(?:\.\d+)*)?", executable):
        executable = "python"
    arguments = [executable, *(argument.lower() for argument in arguments[1:])]
    normalized = " ".join(arguments)
    if _NON_EXECUTING_VERIFICATION_FLAGS.intersection(arguments):
        return False
    return any(
        normalized == candidate or normalized.startswith(candidate + " ")
        for candidate in _VERIFICATION_COMMANDS
    )


def _is_read_only_command(command: str, *, shell: bool) -> bool:
    if shell:
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    if executable in _READ_ONLY_COMMANDS:
        return True
    return executable == "git" and len(arguments) >= 2 and arguments[1] in _READ_ONLY_GIT_COMMANDS
