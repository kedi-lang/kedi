"""Harbor signal-handler safety (audit #214, items 6.1 and 6.3).

6.1: the process supervisor's signal handler must not block on the
termination lock - it only records the request and the run loop drains it.
6.3: the Harbor runner's signal handler must not call blocking
terminal.close() - teardown belongs in run_trial's finally block, with the
previous handlers restored before resource cleanup.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi.integrations.harbor import runner
from kedi.integrations.harbor.runner import _SignalState, _TrialTimedOut
from kedi.integrations.harbor.terminal import TerminalSession
from tests.test_harbor_integration import requires_posix_processes

_REPO_ROOT = Path(__file__).resolve().parent.parent


@requires_posix_processes
def test_supervisor_signal_handler_returns_while_termination_lock_held(
    tmp_path: Path,
) -> None:
    """A SIGTERM arriving while the main thread holds _termination_lock must
    not deadlock the handler (6.1)."""
    child = tmp_path / "signal_child.py"
    child.write_text(
        "\n".join(
            [
                "import os, signal, sys, time",
                "sys.path.insert(0, sys.argv[1])",
                "from kedi.integrations.harbor.process_supervisor import _ProcessSupervisor",
                "config = {",
                '    "command": [sys.executable, "-c", "pass"],',
                '    "cwd": ".",',
                '    "state_path": sys.argv[2],',
                '    "control_path": sys.argv[3],',
                '    "stdout_path": sys.argv[4],',
                '    "stderr_path": sys.argv[5],',
                '    "combined_path": sys.argv[6],',
                '    "output_limit_bytes": 1000,',
                '    "deadline_monotonic": time.monotonic() + 60,',
                "}",
                "sup = _ProcessSupervisor(config)",
                "sup._install_signal_handlers()",
                "sup._termination_lock.acquire()",
                "os.kill(os.getpid(), signal.SIGTERM)",
                "time.sleep(0.2)",
                "# The handler must have returned by now despite the held lock.",
                "assert sup._requested_signal == signal.SIGTERM",
                "assert sup._signal_pending == signal.SIGTERM",
                "sup._termination_lock.release()",
                "print('handler-returned')",
                ""
            ]
        ),
        encoding="utf-8",
    )
    paths = [tmp_path / name for name in ("state.json", "control.json", "stdout.bin", "stderr.bin", "combined.bin")]
    proc = subprocess.run(
        [
            sys.executable,
            str(child),
            str(_REPO_ROOT / "src"),
            *(str(path) for path in paths),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "handler-returned" in proc.stdout


@requires_posix_processes
def test_supervisor_run_terminates_sleeping_child_on_sigterm(tmp_path: Path) -> None:
    """End-to-end: a deferred SIGTERM still terminates the child promptly."""
    root = tmp_path / "run"
    root.mkdir()
    for name in ("stdout.bin", "stderr.bin", "combined.bin"):
        (root / name).touch()
    config = {
        "command": [sys.executable, "-c", "import time; time.sleep(30)"],
        "cwd": str(root),
        "state_path": str(root / "state.json"),
        "control_path": str(root / "control.json"),
        "stdout_path": str(root / "stdout.bin"),
        "stderr_path": str(root / "stderr.bin"),
        "combined_path": str(root / "combined.bin"),
        "output_limit_bytes": 1_000_000,
        "deadline_monotonic": time.monotonic() + 60,
    }
    (root / "control.json").write_text(json.dumps({"retained": False}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "kedi.integrations.harbor.process_supervisor"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(config) + "\n")
    proc.stdin.flush()
    # Wait until the supervisor has installed its handlers and is polling the
    # child; a fixed sleep is racy with interpreter/package import time.
    state_path = root / "state.json"
    deadline = time.monotonic() + 30
    while True:
        try:
            phase = json.loads(state_path.read_text(encoding="utf-8")).get("phase")
        except (OSError, ValueError):
            phase = None
        if phase == "running":
            break
        assert time.monotonic() < deadline, "supervisor never reached running phase"
        time.sleep(0.1)
    assert proc.poll() is None
    os.kill(proc.pid, signal.SIGTERM)
    # Closing stdin lets the input-forwarder thread exit instead of blocking
    # on the pipe while the interpreter shuts down.
    proc.stdin.close()
    try:
        returncode = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("supervisor did not terminate after SIGTERM") from None
    assert returncode == 128 + signal.SIGTERM
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "finished"
    assert state["requested_signal"] == signal.SIGTERM


@requires_posix_processes
def test_runner_signal_handler_does_not_close_terminal() -> None:
    """The runner's handler raises without doing blocking terminal teardown."""
    close_calls: list[None] = []
    terminal = cast(
        TerminalSession,
        SimpleNamespace(close=lambda: close_calls.append(None)),
    )
    state = _SignalState(terminal)
    state.install(timeout_seconds=None)
    try:
        with pytest.raises(_TrialTimedOut):
            os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    finally:
        state.restore()
    assert close_calls == []


def test_runner_restores_signals_before_terminal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_trial's finally restores handlers before conversation/terminal
    cleanup, so a second signal during teardown cannot re-enter the trial
    handler (6.3)."""
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    order: list[str] = []

    def failing_solve(instruction: str) -> str:
        raise RuntimeError("boom")

    def bind(**kwargs: Any) -> Any:
        return lambda _func: failing_solve

    monkeypatch.setattr(runner.kedi, "bind", bind)
    original_restore = _SignalState.restore

    def recording_restore(self: _SignalState) -> None:
        order.append("signals")
        original_restore(self)

    monkeypatch.setattr(_SignalState, "restore", recording_restore)
    original_close = TerminalSession.close
    original_finish = TerminalSession.finish_session

    def recording_close(self: TerminalSession) -> None:
        order.append("terminal")
        original_close(self)

    def recording_finish(self: TerminalSession) -> None:
        order.append("terminal")
        original_finish(self)

    monkeypatch.setattr(TerminalSession, "close", recording_close)
    monkeypatch.setattr(TerminalSession, "finish_session", recording_finish)
    record = runner.run_trial(
        instruction="test",
        model="test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "result.json",
        trial_id="test",
    )
    assert record.state == "agent_failure"
    assert order.index("signals") < order.index("terminal")
    assert order[0] == "signals"
