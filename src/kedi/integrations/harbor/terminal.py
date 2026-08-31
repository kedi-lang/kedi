from __future__ import annotations

import base64
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from ... import ArtifactStream, tool
from ...agent_adapter.hooks import HookEvent, PostToolUseEvent
from .records import VerificationState, is_secret_name, redact_text, utc_now

OutputStream = Literal["stdout", "stderr", "combined"]
_WRITE_TOOLS = frozenset(
    {
        "write_text_file",
        "apply_patch",
        "create_directory",
        "remove_file",
        "remove_directory",
    }
)
_VERIFICATION_COMMANDS = frozenset(
    {
        "cargo test",
        "go test",
        "make check",
        "make prod",
        "make test",
        "make tests",
        "mypy",
        "npm test",
        "pnpm test",
        "pre-commit",
        "python -m pytest",
        "python3 -m pytest",
        "pytest",
        "ruff check",
        "tox",
        "ty check",
        "uv run pytest",
        "yarn test",
    }
)
_NON_EXECUTING_VERIFICATION_FLAGS = frozenset(
    {"--collect-only", "--help", "--list", "--version", "-h"}
)
_READ_ONLY_COMMANDS = frozenset({"cat", "file", "head", "ls", "pwd", "stat", "tail", "wc"})
_READ_ONLY_GIT_COMMANDS = frozenset(
    {"diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
)


@dataclass(slots=True)
class _BytePreview:
    limit: int
    total_bytes: int = 0
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if len(self.head) < self.limit:
            needed = self.limit - len(self.head)
            self.head.extend(chunk[:needed])
        self.tail.extend(chunk)
        if len(self.tail) > self.limit:
            del self.tail[: len(self.tail) - self.limit]

    def render(self) -> tuple[str, bool, bool]:
        truncated = self.total_bytes > len(self.head)
        payload = bytes(self.head)
        if self.total_bytes > len(self.head) + len(self.tail):
            omitted = self.total_bytes - len(self.head) - len(self.tail)
            payload += f"\n... {omitted} bytes omitted ...\n".encode()
            payload += bytes(self.tail)
        elif truncated:
            overlap = max(0, len(self.head) + len(self.tail) - self.total_bytes)
            payload += bytes(self.tail[overlap:])
        invalid_utf8 = False
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            invalid_utf8 = True
        return text, truncated, invalid_utf8


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
    threads: list[threading.Thread] = field(default_factory=list)
    finished_monotonic: float | None = None
    timed_out: bool = False
    stopped: bool = False
    output_limited: bool = False
    limit_termination_started: bool = False
    recorded: bool = False


class TerminalSession:
    """Session-owned terminal tools with bounded previews and process cleanup."""

    def __init__(
        self,
        *,
        root: Path,
        records_dir: Path,
        command_timeout_seconds: float = 120.0,
        preview_bytes: int = 32_768,
        output_limit_bytes: int = 512_000_000,
        deadline_monotonic: float | None = None,
        artifacts_enabled: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.records_dir = records_dir.resolve()
        self.output_dir = self.records_dir / "terminal"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.command_timeout_seconds = command_timeout_seconds
        self.preview_bytes = preview_bytes
        self.output_limit_bytes = output_limit_bytes
        self.deadline_monotonic = deadline_monotonic
        self.artifacts_enabled = artifacts_enabled
        self._lock = threading.RLock()
        self._combined_lock = threading.Lock()
        self._processes: dict[str, _ManagedProcess] = {}
        self._next_process_id = 1
        self._closed = False
        self._verification_state: VerificationState = "not_attempted"
        self._verification_command: str | None = None
        self._verification_exit_code: int | None = None
        self._verification_updated_at: str | None = None
        self._commands: list[dict[str, Any]] = []
        self._events_path = self.records_dir / "terminal-events.jsonl"

    @property
    def tools(self) -> tuple[Any, ...]:
        tools = (
            self.run_command,
            self.run_shell,
            self.start_process,
            self.start_shell,
            self.process_status,
            self.read_process_output,
            self.write_process_input,
            self.stop_process,
            self.list_processes,
            self.verification_status,
        )
        if self.artifacts_enabled:
            return (*tools[:6], self.stream_process_output, *tools[6:])
        return tools

    def hook_handler(self, event: HookEvent) -> None:
        if isinstance(event, PostToolUseEvent) and event.tool_name in _WRITE_TOOLS:
            self.invalidate_verification(f"{event.tool_name} changed the workspace")

    @tool(risk="mutating")
    def run_command(
        self,
        argv: list[str],
        cwd: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one foreground command without shell expansion.

        Args:
            argv: Exact executable and arguments, such as ["pytest", "-q"].
            cwd: Working directory inside the assigned task root.
            timeout_seconds: Optional command timeout bounded by the trial deadline.
        """
        normalized = _validated_argv(argv)
        managed = self._start(normalized, cwd=cwd, shell=False)
        return self._wait(managed, timeout_seconds=timeout_seconds)

    @tool(risk="mutating")
    def run_shell(
        self,
        command: str,
        cwd: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run a foreground Bash command when pipes, redirects, or expansion are required."""
        if not command.strip():
            raise ValueError("shell command cannot be empty")
        managed = self._start(command, cwd=cwd, shell=True)
        return self._wait(managed, timeout_seconds=timeout_seconds)

    @tool(risk="mutating")
    def start_process(self, argv: list[str], cwd: str = ".") -> dict[str, Any]:
        """Start a background process without shell expansion and return its process ID."""
        managed = self._start(_validated_argv(argv), cwd=cwd, shell=False)
        self.invalidate_verification("a background process was started")
        return self._snapshot(managed)

    @tool(risk="mutating")
    def start_shell(self, command: str, cwd: str = ".") -> dict[str, Any]:
        """Start a background Bash command when shell syntax is required."""
        if not command.strip():
            raise ValueError("shell command cannot be empty")
        managed = self._start(command, cwd=cwd, shell=True)
        self.invalidate_verification("a background process was started")
        return self._snapshot(managed)

    @tool(risk="read_only")
    def process_status(self, process_id: str) -> dict[str, Any]:
        """Return current status, exit code, output sizes, and previews for a process."""
        return self._snapshot(self._process(process_id))

    @tool(risk="read_only")
    def read_process_output(
        self,
        process_id: str,
        stream: OutputStream = "combined",
        offset_bytes: int = 0,
        max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        """Read a bounded byte range from a process output log.

        Binary data is returned as base64 while text data is returned as UTF-8.
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

    @tool(risk="read_only")
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

    @tool(risk="mutating")
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

    @tool(risk="mutating")
    def stop_process(self, process_id: str, grace_seconds: float = 2.0) -> dict[str, Any]:
        """Terminate a session-owned process tree, escalating to SIGKILL if needed."""
        if grace_seconds < 0 or grace_seconds > 30:
            raise ValueError("grace_seconds must be between 0 and 30")
        managed = self._process(process_id)
        self._terminate(managed, grace_seconds=grace_seconds)
        self.invalidate_verification("a background process was stopped")
        return self._snapshot(managed)

    @tool(risk="read_only")
    def list_processes(self) -> list[dict[str, Any]]:
        """List all foreground and background processes owned by this Kedi session."""
        with self._lock:
            return [self._snapshot(item) for item in self._processes.values()]

    @tool(risk="read_only")
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
            running = sum(item.process.poll() is None for item in self._processes.values())
            return {
                "commands": len(self._commands),
                "processes": len(self._processes),
                "running_processes": running,
                "verification": self.verification_status(),
                "records": list(self._commands),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            processes = tuple(self._processes.values())
        for managed in processes:
            if managed.process.poll() is None:
                self._terminate(managed, grace_seconds=1.0)
            else:
                self._join_readers(managed)
                self._record_process(managed)

    def _start(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str,
        shell: bool,
    ) -> _ManagedProcess:
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
            self._write_event(
                {
                    "event": "process_start_failed",
                    "process_id": process_id,
                    "command": redact_text(str(display)),
                    "cwd": str(resolved_cwd),
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc)),
                }
            )
            raise RuntimeError(f"could not start command: {exc}") from exc
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
            }
        )
        return managed

    def _reader_thread(
        self,
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
        self,
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
        self._join_readers(managed)
        self._record_process(managed)
        self._update_verification(managed)
        return self._snapshot(managed)

    def _terminate(self, managed: _ManagedProcess, *, grace_seconds: float) -> None:
        if managed.process.poll() is None:
            self._signal_process_group(managed, signal.SIGTERM)
            try:
                managed.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self._signal_process_group(managed, signal.SIGKILL)
                managed.process.wait(timeout=5)
        if managed.finished_monotonic is None:
            managed.finished_monotonic = time.monotonic()
        managed.stopped = True
        if managed.process.stdin is not None:
            managed.process.stdin.close()
        self._join_readers(managed)
        self._record_process(managed)

    def _terminate_for_output_limit(self, managed: _ManagedProcess) -> None:
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

    def _join_readers(self, managed: _ManagedProcess) -> None:
        for thread in managed.threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=5)

    def _record_process(self, managed: _ManagedProcess) -> None:
        with self._lock:
            if managed.recorded:
                return
            managed.recorded = True
            record = self._snapshot(managed)
            self._commands.append(record)
        self._write_event({"event": "process_finished", **record})

    def _snapshot(self, managed: _ManagedProcess) -> dict[str, Any]:
        exit_code = managed.process.poll()
        if exit_code is not None and managed.finished_monotonic is None:
            managed.finished_monotonic = time.monotonic()
        stdout, stdout_truncated, stdout_invalid = managed.stdout_preview.render()
        stderr, stderr_truncated, stderr_invalid = managed.stderr_preview.render()
        return {
            "process_id": managed.process_id,
            "pid": managed.process.pid,
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
            "stdout": stdout,
            "stderr": stderr,
            "stdout_bytes": managed.stdout_preview.total_bytes,
            "stderr_bytes": managed.stderr_preview.total_bytes,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_invalid_utf8": stdout_invalid,
            "stderr_invalid_utf8": stderr_invalid,
            "stdout_path": str(managed.stdout_path),
            "stderr_path": str(managed.stderr_path),
            "combined_path": str(managed.combined_path),
            "started_at": managed.started_at,
        }

    def _update_verification(self, managed: _ManagedProcess) -> None:
        if not _is_verification_command(managed.command, shell=managed.shell):
            if not _is_read_only_command(managed.command, shell=managed.shell):
                self.invalidate_verification("a command may have changed the workspace")
            return
        with self._lock:
            self._verification_state = (
                "passed"
                if managed.process.returncode == 0
                and not managed.timed_out
                and not managed.output_limited
                else "failed"
            )
            self._verification_command = managed.command
            self._verification_exit_code = managed.process.returncode
            self._verification_updated_at = utc_now()
        self._write_event({"event": "verification_updated", **self.verification_status()})

    def _safe_cwd(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError(f"working directory escapes task root: {raw}")
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        return resolved

    @staticmethod
    def _signal_process_group(managed: _ManagedProcess, signum: int) -> None:
        try:
            os.killpg(managed.process.pid, signum)
        except ProcessLookupError:
            pass

    def _bounded_timeout(self, requested: float | None) -> float:
        timeout = self.command_timeout_seconds if requested is None else requested
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.deadline_monotonic is not None:
            remaining = self.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("trial deadline has elapsed")
            timeout = min(timeout, remaining)
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
        payload = {"timestamp": utc_now(), **event}
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        with self._lock:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


def _validated_argv(argv: list[str]) -> tuple[str, ...]:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("argv must contain at least one non-empty string")
    return tuple(argv)


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


class _empty_binary_file:
    def __enter__(self) -> _empty_binary_file:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def seek(self, _: int) -> None:
        return None

    def read(self, _: int) -> bytes:
        return b""


__all__ = ["TerminalSession"]
