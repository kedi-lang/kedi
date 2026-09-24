from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from kedi.integrations.harbor import runner
from kedi.integrations.harbor.terminal import TerminalSession
from tests.test_harbor_integration import _install_fake_harbor


@pytest.mark.parametrize(
    "task,config,expected",
    [
        ("steps = [{}]", {}, None),
        ("", {}, None),
        ("[agent]\ntimeout_sec=1", {"timeout_multiplier": 0}, "invalid"),
    ],
)
def test_harbor_deadlines_reject_invalid_or_ambiguous_budgets(
    tmp_path, monkeypatch, task, config, expected
):
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor.agent import KediAgent

    logs = tmp_path / "logs" / "agent"
    logs.mkdir(parents=True)
    (logs.parent / "config.json").write_text(json.dumps(config))
    envdir = tmp_path / "task" / "environment"
    envdir.mkdir(parents=True)
    (envdir.parent / "task.toml").write_text(task)
    agent = KediAgent(logs_dir=logs, model_name="test")
    if expected == "invalid":
        with pytest.raises(ValueError, match="positive and finite"):
            agent._execution_timeout(SimpleNamespace(environment_dir=envdir))
    else:
        assert agent._execution_timeout(SimpleNamespace(environment_dir=envdir)) is None


@pytest.mark.parametrize("option", ["codex_auth_helper_distribution", "kedi_otel_distribution"])
def test_harbor_missing_local_distributions_fail_early(tmp_path, monkeypatch, option):
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor.agent import KediAgent

    with pytest.raises(FileNotFoundError):
        KediAgent(logs_dir=tmp_path, model_name="test", **{option: tmp_path / "missing.whl"})


@pytest.mark.parametrize(
    "options,message",
    [
        ({"trial_deadline_unix": float("nan")}, "must be finite"),
        (
            {"adapter": "claude", "history_archive_threshold_tokens": 1000},
            "history archival requires",
        ),
        ({"adapter": "langchain", "request_observation": True}, "request observation currently"),
    ],
)
def test_runner_rejects_invalid_setup_before_model_io(tmp_path, monkeypatch, options, message):
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    record = runner.run_trial(
        instruction="test",
        model="test",
        workspace=tmp_path,
        result_path=tmp_path / "result.json",
        trial_id="test",
        **{"adapter": "pydantic", **options},
    )
    assert record.failure_phase == "setup"
    assert message in record.error_message


@pytest.mark.parametrize(
    "adapter,observation", [("pydantic", True), ("langchain", False), ("claude", False)]
)
def test_runner_selects_adapter_and_observer_without_model_io(
    tmp_path, monkeypatch, adapter, observation
):
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    captured = []

    def bind(**kwargs):
        captured.append(kwargs)
        return lambda _: lambda instruction: "done"

    monkeypatch.setattr(runner.kedi, "bind", bind)
    record = runner.run_trial(
        instruction="test",
        model="test",
        workspace=tmp_path,
        result_path=tmp_path / "result.json",
        trial_id="test",
        adapter=adapter,
        request_observation=observation,
    )
    assert record.state == "completed"
    configured = captured[0]["adapter"]
    assert (configured if isinstance(configured, str) else configured.shortname) == adapter


def test_runner_deadline_can_expire_during_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    clock = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original = runner.ModelRequestBudgetLimits

    def limits(**kwargs):
        clock[0] = 2.0
        return original(**kwargs)

    monkeypatch.setattr(runner, "ModelRequestBudgetLimits", limits)
    record = runner.run_trial(
        instruction="test",
        model="test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "result.json",
        trial_id="test",
        trial_timeout_seconds=1,
    )
    assert record.state == "timed_out"
    assert "during runner setup" in record.error_message


@pytest.mark.skipif(os.name != "posix", reason="Harbor sandbox process lifecycle requires POSIX")
def test_terminal_rejects_outside_root_and_foreground_wait(tmp_path):
    with pytest.raises(ValueError, match="inside cwd_root"):
        TerminalSession(root=tmp_path, cwd_root=tmp_path / "other", records_dir=tmp_path / "logs")
    terminal = TerminalSession(root=tmp_path, records_dir=tmp_path / "logs")
    try:
        result = terminal.run_command([sys.executable, "-c", "pass"])
        with pytest.raises(ValueError, match="only background"):
            terminal.wait_process(result["process_id"])
        running = terminal.start_process([sys.executable, "-c", "import time; time.sleep(10)"])
        managed = terminal._process(running["process_id"])
        terminal._update_verification(managed)
        assert not managed.verification_recorded
    finally:
        terminal.close()
