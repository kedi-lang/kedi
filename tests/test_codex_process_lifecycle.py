from __future__ import annotations

import os
import signal
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any

import pytest

from kedi.agent_adapter.adapters.codex import app_server as server_module
from kedi.agent_adapter.adapters.codex.app_server import (
    CodexAppServerClient,
    CodexAppServerDisconnected,
)
from tests.test_codex_adapter import FAKE_APP_SERVER, _as_popen, _FakeProcess


def test_close_marks_closed_before_rpc_without_blocking_the_reader_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient(("unused",))
    client._retained_threads.update(("one", "two"))
    monkeypatch.setattr(client, "_is_healthy", lambda: True)
    deleted = []

    def request(method: str, params: dict[str, Any], *, timeout: float | None = None) -> dict:
        acquired = []

        def reader() -> None:
            if client._start_lock.acquire(timeout=0.2):
                acquired.append(True)
                client._start_lock.release()

        thread = Thread(target=reader)
        thread.start()
        thread.join(timeout=1)
        assert acquired, "close held the reader's disconnect lock during RPC"
        assert client._closed
        assert timeout is not None and 0 < timeout <= 2
        assert method == "thread/delete"
        deleted.append(params["threadId"])
        return {}

    monkeypatch.setattr(client, "_request_raw", request)
    try:
        client.close()
        assert set(deleted) == {"one", "two"}
        assert not client._retained_threads
        with pytest.raises(CodexAppServerDisconnected, match="closed"):
            client.request("thread/start", {})
    finally:
        client._retained_threads.clear()
        client.close()


def test_start_rechecks_closed_after_acquiring_its_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CodexAppServerClient(("unused",))

    def unhealthy() -> bool:
        client._closed = True
        return False

    monkeypatch.setattr(client, "_is_healthy", unhealthy)
    monkeypatch.setattr(client, "_start_process", lambda: pytest.fail("restarted a closed client"))
    try:
        with pytest.raises(CodexAppServerDisconnected, match="closed"):
            client._ensure_started()
    finally:
        client.close()


def test_close_uses_one_total_budget_for_retained_thread_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient(("unused",))
    client._retained_threads.update(("one", "two", "three"))
    clock = iter((0.0, 0.5, 2.1))
    timeouts = []

    def timeout(*_args: Any, **kwargs: Any) -> dict:
        timeouts.append(kwargs["timeout"])
        raise TimeoutError("backend stalled")

    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(client, "_request_raw", timeout)
    client.close()
    assert timeouts == [1.5]
    assert client._closed and not client._retained_threads


@pytest.mark.parametrize("running, wait_timeouts", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("with_pipes", [False, True])
def test_non_posix_stop_closes_pipes_and_escalates_a_stalled_process(
    monkeypatch: pytest.MonkeyPatch, running: bool, wait_timeouts: int, with_pipes: bool
) -> None:
    client = CodexAppServerClient(("unused",))
    process = _FakeProcess(running=running, wait_timeouts=wait_timeouts)
    if not with_pipes:
        process.stdout = process.stderr = None
    if wait_timeouts:
        monkeypatch.setattr(process, "kill", lambda: setattr(process, "killed", True))
    client._process = _as_popen(process)
    with monkeypatch.context() as context:
        context.setattr(server_module, "os", SimpleNamespace(name="nt"))
        client._stop_process()
    assert process.terminated is running
    assert process.killed is bool(wait_timeouts)
    assert process.stdin.closed
    if with_pipes:
        assert process.stdout.closed and process.stderr.closed
    client.close()


def test_disconnect_rechecks_process_identity_after_acquiring_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CodexAppServerClient(("unused",))
    process = _as_popen(_FakeProcess())
    client._process = process

    @contextmanager
    def detach():
        client._process = None
        yield

    with monkeypatch.context() as context:
        context.setattr(client, "_start_lock", detach())
        client._mark_disconnected(process, CodexAppServerDisconnected("old process"))
    assert client._last_disconnect is None
    client.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership")
@pytest.mark.parametrize("root_exits_first", [False, True])
def test_close_reaps_owned_descendants_and_closes_all_pipes(
    tmp_path: Path, root_exits_first: bool
) -> None:
    pid_path = tmp_path / "child.pid"
    client = CodexAppServerClient(
        (sys.executable, str(FAKE_APP_SERVER)),
        env={"KEDI_CODEX_FAKE_CHILD_PID": str(pid_path)},
        request_timeout=2,
    )
    process = None
    child_pid = None
    try:
        client.request("thread/start", {})
        process = client._process
        assert process is not None
        deadline = time.monotonic() + 2
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(pid_path.read_text())
        assert os.getpgid(process.pid) == process.pid
        assert os.getpgid(child_pid) == process.pid
        readers = (client._reader, client._stderr_reader)
        if root_exits_first:
            process.terminate()
            process.wait(timeout=2)
        client.close()
        assert process.poll() is not None
        assert all(reader is not None and not reader.is_alive() for reader in readers)
        assert all(
            pipe is not None and pipe.closed
            for pipe in (process.stdin, process.stdout, process.stderr)
        )
    finally:
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        client.close()
        if process is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
