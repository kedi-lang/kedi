from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence

from .process_groups import finish_process_group
from .records import atomic_write_json, redact_text


class _ProcessSupervisor:
    def __init__(self, config: Mapping[str, Any]) -> None:
        command = config.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError("supervisor command must be a non-empty string list")
        self.command: Sequence[str] = tuple(command)
        self.cwd = Path(str(config["cwd"]))
        self.state_path = Path(str(config["state_path"]))
        self.control_path = Path(str(config["control_path"]))
        self.stdout_path = Path(str(config["stdout_path"]))
        self.stderr_path = Path(str(config["stderr_path"]))
        self.combined_path = Path(str(config["combined_path"]))
        self.output_limit_bytes = int(config["output_limit_bytes"])
        self.deadline_monotonic = float(config["deadline_monotonic"])
        self._child: subprocess.Popen[bytes] | None = None
        self._state_lock = threading.RLock()
        self._combined_lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._termination_deadline: float | None = None
        self._timed_out = False
        self._output_limited = False
        self._requested_signal: int | None = None

    def run(self) -> int:
        self._install_signal_handlers()
        try:
            child = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=dict(os.environ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return self._record_start_failure(exc)

        self._child = child
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        self._write_state(
            phase="running",
            target_pid=child.pid,
            target_pgid=child.pid,
            returncode=None,
        )
        readers = (
            self._reader("stdout", child.stdout, self.stdout_path),
            self._reader("stderr", child.stderr, self.stderr_path),
        )
        for reader in readers:
            reader.start()
        input_forwarder = threading.Thread(
            target=self._forward_input,
            name="kedi-process-input",
            daemon=True,
        )
        input_forwarder.start()

        while child.poll() is None:
            now = time.monotonic()
            if now >= self.deadline_monotonic:
                self._timed_out = True
                self._request_termination(signal.SIGTERM)
            if self._termination_deadline is not None and now >= self._termination_deadline:
                self._signal_child(signal.SIGKILL)
                self._termination_deadline = None
            time.sleep(0.05)

        returncode = child.wait()
        finish_process_group(child.pid)
        self._close_child_input()
        for reader in readers:
            reader.join(timeout=5)
        self._write_state(
            phase="finished",
            target_pid=child.pid,
            target_pgid=child.pid,
            returncode=returncode,
            finished_monotonic=time.monotonic(),
        )
        if returncode >= 0:
            return min(returncode, 125)
        return min(128 + abs(returncode), 255)

    def _forward_input(self) -> None:
        child = self._child
        if child is None or child.stdin is None:
            return
        source = sys.stdin.buffer
        read_chunk = getattr(source, "read1", source.read)
        try:
            while chunk := read_chunk(65_536):
                child.stdin.write(chunk)
                child.stdin.flush()
        except (BrokenPipeError, OSError):
            return
        if not self._is_retained():
            self._request_termination(signal.SIGTERM)

    def _close_child_input(self) -> None:
        child = self._child
        if child is None or child.stdin is None:
            return
        try:
            child.stdin.close()
        except BrokenPipeError:
            pass

    def _reader(self, stream: str, source: Any, path: Path) -> threading.Thread:
        def read() -> None:
            written = 0
            with path.open("ab", buffering=0) as output:
                read_chunk = getattr(source, "read1", source.read)
                while chunk := read_chunk(65_536):
                    allowed = max(0, self.output_limit_bytes - written)
                    kept = chunk[:allowed]
                    if kept:
                        output.write(kept)
                        with self._combined_lock:
                            with self.combined_path.open("ab", buffering=0) as combined:
                                combined.write(kept)
                        written += len(kept)
                    if len(kept) < len(chunk):
                        self._output_limited = True
                        self._write_state(
                            phase="running",
                            target_pid=self._child.pid if self._child is not None else None,
                            target_pgid=self._child.pid if self._child is not None else None,
                            returncode=None,
                            limited_stream=stream,
                        )
                        self._request_termination(signal.SIGTERM)
                        break
                output.flush()
                os.fsync(output.fileno())
            with self._combined_lock:
                with self.combined_path.open("ab", buffering=0) as combined:
                    os.fsync(combined.fileno())
            source.close()

        return threading.Thread(target=read, name=f"kedi-process-{stream}", daemon=True)

    def _record_start_failure(self, exc: OSError) -> int:
        exit_code = 127 if isinstance(exc, FileNotFoundError) else 126
        message = f"could not start command: {redact_text(str(exc))}\n"
        payload = message.encode("utf-8")[: self.output_limit_bytes]
        for path in (self.stderr_path, self.combined_path):
            with path.open("ab", buffering=0) as handle:
                handle.write(payload)
                os.fsync(handle.fileno())
        self._write_state(
            phase="failed_to_start",
            target_pid=None,
            target_pgid=None,
            returncode=exit_code,
            start_error_type=type(exc).__name__,
            start_error_message=redact_text(str(exc)),
            finished_monotonic=time.monotonic(),
        )
        return exit_code

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._handle_signal)

    def _handle_signal(self, signum: int, _: FrameType | None) -> None:
        self._requested_signal = signum
        self._request_termination(signum)

    def _request_termination(self, signum: int) -> None:
        with self._termination_lock:
            self._signal_child(signum)
            if signum != signal.SIGKILL and self._termination_deadline is None:
                self._termination_deadline = time.monotonic() + 1.0

    def _signal_child(self, signum: int) -> None:
        child = self._child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signum)
        except (PermissionError, ProcessLookupError):
            pass

    def _is_retained(self) -> bool:
        try:
            value = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return isinstance(value, dict) and value.get("retained") is True

    def _write_state(self, **updates: Any) -> None:
        with self._state_lock:
            state: dict[str, Any] = {
                "schema_version": 1,
                "supervisor_pid": os.getpid(),
                "timed_out": self._timed_out,
                "output_limited": self._output_limited,
                "requested_signal": self._requested_signal,
            }
            try:
                current = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                current = None
            if isinstance(current, dict):
                state.update(current)
            state.update(updates)
            state["timed_out"] = self._timed_out
            state["output_limited"] = self._output_limited
            state["requested_signal"] = self._requested_signal
            atomic_write_json(self.state_path, state)


def main() -> int:
    raw_config = sys.stdin.buffer.readline()
    if not raw_config:
        return 126
    try:
        config = json.loads(raw_config)
    except (ValueError, TypeError):
        return 126
    if not isinstance(config, dict):
        return 126
    try:
        return _ProcessSupervisor(config).run()
    except BaseException as exc:
        state_path = config.get("state_path")
        if isinstance(state_path, str):
            atomic_write_json(
                Path(state_path),
                {
                    "schema_version": 1,
                    "phase": "supervisor_failure",
                    "supervisor_pid": os.getpid(),
                    "returncode": 126,
                    "start_error_type": type(exc).__name__,
                    "start_error_message": redact_text(str(exc)),
                    "finished_monotonic": time.monotonic(),
                },
            )
        return 126


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
