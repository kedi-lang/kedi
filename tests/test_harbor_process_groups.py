from __future__ import annotations

import signal

import pytest

from kedi.integrations.harbor import process_groups


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
@pytest.mark.parametrize("at_signal", [signal.SIGTERM, 0, signal.SIGKILL])
def test_group_cleanup_handles_exit_races(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError], at_signal: int
) -> None:
    calls: list[int] = []

    def killpg(group: int, signum: int) -> None:
        assert group == 4321
        calls.append(signum)
        if signum == at_signal:
            raise error()

    monkeypatch.setattr(process_groups.os, "killpg", killpg)
    process_groups.finish_process_group(4321, grace_seconds=0 if at_signal == signal.SIGKILL else 1)
    assert calls[-1] == at_signal


def test_group_cleanup_escalates_only_the_owned_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(process_groups.os, "killpg", lambda group, sig: calls.append((group, sig)))
    process_groups.finish_process_group(4321, grace_seconds=0)
    assert calls == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
