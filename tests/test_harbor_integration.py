from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from pathlib import PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, cast

import pytest
from click import ClickException, UsageError
from click.testing import CliRunner

from kedi.agent_adapter.approval import ApprovalRequest
from kedi.agent_adapter.hooks import PostToolUseEvent
from kedi.agent_adapter.invocation import InvocationUsage, report_invocation
from kedi.integrations.harbor import host, runner
from kedi.integrations.harbor.records import (
    HarborRunPolicy,
    HarnessPolicy,
    RunManifest,
    TrialRecord,
    atomic_write_json,
    current_gitlink_revision,
    current_git_state,
    is_secret_name,
    invocation_usage_dict,
    load_manifest,
    redact_text,
    sha256_file,
)
from kedi.integrations.harbor import terminal as harbor_terminal
from kedi.integrations.harbor.terminal import TerminalSession
from kedi.lang.parser import parse_program


@pytest.fixture
def terminal(tmp_path: Path) -> Iterator[TerminalSession]:
    value = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        command_timeout_seconds=10,
        preview_bytes=32,
        output_limit_bytes=1_000_000,
    )
    yield value
    value.close()


def test_manifest_is_immutable_and_digest_validated(tmp_path: Path) -> None:
    manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/gpt-test",
        task_names=("build-task", "service-task"),
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    manifest.write(path)
    assert load_manifest(path).material_digest == manifest.material_digest

    payload = json.loads(path.read_text())
    payload["model"] = "openai/changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        load_manifest(path)

    other = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/other",
        task_names=("build-task",),
    )
    with pytest.raises((ValueError, FileExistsError)):
        other.write(path)

    immutable_path = tmp_path / "immutable.json"
    manifest.write(immutable_path)
    with pytest.raises(FileExistsError, match="different run manifest"):
        other.write(immutable_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"compaction_threshold": 1}, "requires native"),
        ({"compaction_mode": "native", "compaction_threshold": 0}, "positive"),
        ({"command_timeout_seconds": 0}, "positive"),
        ({"command_timeout_seconds": float("nan")}, "positive"),
        ({"terminal_preview_bytes": 0}, "positive"),
        ({"terminal_output_limit_bytes": 0}, "positive"),
        ({"history": False}, "prefix_cache requires history"),
        ({"prefix_cache": False}, "stateful Kedi history"),
        ({"subagents": True}, "does not define subagents"),
        ({"dynamic_workflows": True}, "does not define subagents"),
        ({"artifact_threshold_bytes": 0}, "positive"),
        ({"artifact_quota_bytes": 0}, "positive"),
    ],
)
def test_harness_policy_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HarnessPolicy(**kwargs)


def test_harness_policy_rejects_unknown_compaction_and_stateless_compaction() -> None:
    with pytest.raises(ValueError, match="disabled or native"):
        HarnessPolicy(compaction_mode=cast(Any, "unknown"))
    with pytest.raises(ValueError, match="requires history"):
        HarnessPolicy(history=False, prefix_cache=False, compaction_mode="native")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_multiplier": 0}, "positive"),
        ({"timeout_multiplier": float("inf")}, "positive"),
        ({"agent_timeout_multiplier": 0}, "positive"),
        ({"verifier_timeout_multiplier": -1}, "positive"),
        ({"agent_setup_timeout_multiplier": 0}, "positive"),
        ({"environment_build_timeout_multiplier": 0}, "positive"),
        ({"max_retries": -1}, "must not be negative"),
        ({"retry_include_exceptions": ("",)}, "empty"),
        ({"retry_exclude_exceptions": ("Error", "Error")}, "duplicates"),
        (
            {
                "retry_include_exceptions": ("TransientError",),
                "retry_exclude_exceptions": ("TransientError",),
            },
            "overlap",
        ),
    ],
)
def test_harbor_run_policy_rejects_ambiguous_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HarborRunPolicy(**kwargs)


def test_records_are_atomic_and_secrets_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    atomic_write_json(path, {"status": "ok"})
    assert json.loads(path.read_text()) == {"status": "ok"}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret-value")
    assert redact_text("token=sk-very-secret-value") == "token=[REDACTED]"
    assert "[REDACTED]" in redact_text("Bearer abcdefghijklmnopqrstuvwxyz")
    with pytest.raises(ValueError, match="must not contain credentials"):
        RunManifest(
            harbor_revision="harbor-commit",
            kedi_commit="kedi-commit",
            kedi_dirty=False,
            model="openai/gpt-test",
            model_settings={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema_version": 2}, "unsupported"),
        ({"harbor_revision": ""}, "revisions must be pinned"),
        ({"kedi_commit": ""}, "revisions must be pinned"),
        ({"kedi_package": " "}, "must not be empty"),
        (
            {"kedi_package": "kedi", "kedi_wheel_sha256": "0" * 64},
            "mutually exclusive",
        ),
        ({"kedi_wheel_sha256": "ABC"}, "lowercase SHA-256"),
        ({"tree_sitter_kedi_revision": "ABC"}, "lowercase Git revision"),
        ({"model": " "}, "model must not be empty"),
        ({"effort": "impossible"}, "unsupported effort"),
        ({"model_settings": {"temperature": float("nan")}}, "JSON object"),
        ({"model_settings": ["not", "an", "object"]}, "JSON object"),
        ({"attempts": 0}, "must be positive"),
        ({"concurrency": 0}, "must be positive"),
        ({"task_names": ("same", "same")}, "duplicates"),
    ],
)
def test_manifest_rejects_non_reproducible_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    values: dict[str, Any] = {
        "harbor_revision": "harbor-commit",
        "kedi_commit": "kedi-commit",
        "kedi_dirty": False,
        "model": "openai/test",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RunManifest(**values)


def test_record_helpers_capture_git_usage_hashes_and_nested_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(("abc123\n", " M changed.py\n"))

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert current_git_state(tmp_path) == ("abc123", True)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="160000 commit " + "a" * 40 + "\ttree-sitter-kedi\n"
        ),
    )
    assert current_gitlink_revision(tmp_path, "tree-sitter-kedi") == "a" * 40
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="100644 blob abc\tfile\n"),
    )
    with pytest.raises(ValueError, match="not a pinned Git submodule"):
        current_gitlink_revision(tmp_path, "tree-sitter-kedi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="160000 commit " + "z" * 40 + "\ttree-sitter-kedi\n"
        ),
    )
    with pytest.raises(ValueError, match="invalid Git revision"):
        current_gitlink_revision(tmp_path, "tree-sitter-kedi")

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    assert (
        sha256_file(payload) == "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
    )

    usage = InvocationUsage(
        requests=1,
        tool_calls=2,
        input_tokens=10,
        cache_read_tokens=4,
        cache_write_tokens=3,
        output_tokens=5,
        cost_usd=Decimal("0.125"),
    )
    assert invocation_usage_dict(usage)["cost_usd"] == 0.125
    assert invocation_usage_dict(InvocationUsage())["cost_usd"] is None

    with pytest.raises(ValueError, match="secret"):
        RunManifest(
            harbor_revision="harbor-commit",
            kedi_commit="kedi-commit",
            kedi_dirty=False,
            model="openai/test",
            model_settings={"nested": [{"password": "fixture"}]},
        )


def test_atomic_write_removes_temporary_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace")))
    with pytest.raises(OSError, match="replace"):
        atomic_write_json(tmp_path / "record.json", {"state": "pending"})
    assert not list(tmp_path.glob(".record.json.*.tmp"))


def test_trial_record_serializes_all_fields() -> None:
    record = TrialRecord(
        schema_version=1,
        trial_id="trial",
        state="completed",
        started_at="start",
        finished_at="finish",
        duration_seconds=1.0,
        instruction_digest="digest",
        workspace="/workspace",
        model="openai/test",
        adapter="pydantic",
        manifest_digest=None,
        usage={},
        verification={},
        terminal={},
        policy={},
    )
    assert record.to_dict()["trial_id"] == "trial"


def test_terminal_runs_argv_shell_binary_and_bounded_output(terminal: TerminalSession) -> None:
    argv = terminal.run_command([sys.executable, "-c", "print('hello')"])
    assert argv["exit_code"] == 0
    assert argv["stdout"] == "hello\n"

    shell = terminal.run_shell("printf 'left\\n'; printf 'right\\n' >&2")
    assert shell["exit_code"] == 0
    assert shell["stdout"] == "left\n"
    assert shell["stderr"] == "right\n"

    binary = terminal.run_command(
        [sys.executable, "-c", "import os; os.write(1, bytes([255, 0, 65]))"]
    )
    page = terminal.read_process_output(binary["process_id"], stream="stdout")
    assert page["encoding"] == "base64"
    assert page["complete"] is True

    long_output = terminal.run_command([sys.executable, "-c", "print('a' * 100)"])
    assert long_output["stdout_truncated"] is True
    assert "omitted" in long_output["stdout"]
    stream = terminal.stream_process_output(long_output["process_id"], "stdout")
    assert "".join(stream.iter_chunks()).strip() == "a" * 100


def test_terminal_background_lifecycle_and_verification(terminal: TerminalSession) -> None:
    process = terminal.start_process(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; print('ready', flush=True); "
                "line=sys.stdin.readline(); print(line.strip(), flush=True); time.sleep(30)"
            ),
        ]
    )
    process_id = cast(str, process["process_id"])
    time.sleep(0.1)
    assert terminal.process_status(process_id)["stdout"] == "ready\n"
    assert (
        terminal.write_process_input(process_id, "continue", append_newline=True)["written_bytes"]
        == 9
    )
    time.sleep(0.1)
    assert "continue" in terminal.read_process_output(process_id)["content"]
    assert terminal.stop_process(process_id)["status"] == "finished"
    assert terminal.list_processes()[0]["process_id"] == process_id

    (terminal.root / "test_fixture.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    verification = terminal.run_command([sys.executable, "-m", "pytest", "-q", "test_fixture.py"])
    assert verification["exit_code"] == 0
    assert terminal.verification_status()["state"] == "passed"
    terminal.run_command(["git", "status", "--short"])
    assert terminal.verification_status()["state"] == "passed"
    terminal.hook_handler(
        PostToolUseEvent(
            run_id="run",
            sequence=1,
            adapter_shortname="pydantic",
            tool_call_id="call",
            tool_name="apply_patch",
            origin="kedi",
            arguments={},
            result="ok",
        )
    )
    assert terminal.verification_status()["state"] == "stale"


def test_shell_compound_command_cannot_claim_verification(
    terminal: TerminalSession,
) -> None:
    terminal.run_shell("pytest --version; true")
    assert terminal.verification_status()["state"] == "not_attempted"
    terminal.run_command(["pytest", "--version"])
    assert terminal.verification_status()["state"] == "not_attempted"


def test_terminal_timeout_path_bounds_and_missing_process(
    terminal: TerminalSession,
    tmp_path: Path,
) -> None:
    timed_out = terminal.run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds=0.05,
    )
    assert timed_out["timed_out"] is True
    assert timed_out["status"] == "finished"
    with pytest.raises(PermissionError, match="escapes"):
        terminal.run_command(["true"], cwd=str(tmp_path.parent))
    with pytest.raises(KeyError, match="unknown"):
        terminal.process_status("process_999")
    with pytest.raises(ValueError, match="argv"):
        terminal.run_command([])


def test_terminal_output_limit_kills_a_process_that_ignores_sigterm(tmp_path: Path) -> None:
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        output_limit_bytes=1_024,
        preview_bytes=128,
    )
    try:
        result = terminal.run_command(
            [
                sys.executable,
                "-c",
                (
                    "import signal,sys,time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.write('x' * 100_000); sys.stdout.flush(); time.sleep(30)"
                ),
            ],
            timeout_seconds=5,
        )
        assert result["status"] == "finished"
        assert result["output_limited"] is True
        assert result["stdout_bytes"] == 1_024
        assert result["exit_code"] != 0
    finally:
        terminal.close()


def test_terminal_rejects_invalid_inputs_and_closed_sessions(
    terminal: TerminalSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="shell command"):
        terminal.run_shell("  ")
    with pytest.raises(ValueError, match="shell command"):
        terminal.start_shell("\n")
    with pytest.raises(ValueError, match="offset_bytes"):
        terminal.read_process_output("missing", offset_bytes=-1)
    with pytest.raises(ValueError, match="max_bytes"):
        terminal.read_process_output("missing", max_bytes=0)
    with pytest.raises(ValueError, match="max_bytes"):
        terminal.read_process_output("missing", max_bytes=1_000_001)
    with pytest.raises(ValueError, match="grace_seconds"):
        terminal.stop_process("missing", grace_seconds=-1)
    with pytest.raises(NotADirectoryError):
        terminal.run_command(["true"], cwd="missing-directory")
    missing = terminal.run_command([str(tmp_path / "missing-executable")])
    assert missing["status"] == "failed_to_start"
    assert missing["exit_code"] == 127
    assert missing["pid"] is None
    assert missing["start_error_type"] == "FileNotFoundError"
    assert "could not start command" in missing["stderr"]
    assert terminal.summary()["commands"] == 1

    missing_verification = terminal.run_command([str(tmp_path / "pytest"), "-q"])
    assert missing_verification["exit_code"] == 127
    assert terminal.verification_status()["state"] == "failed"

    missing_background = terminal.start_process([str(tmp_path / "missing-background")])
    assert missing_background["status"] == "failed_to_start"
    assert terminal.list_processes() == []

    with monkeypatch.context() as context:
        context.setattr(
            harbor_terminal.subprocess,
            "Popen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("blocked")),
        )
        blocked_shell = terminal.run_shell("printf nope")
        blocked_background_shell = terminal.start_shell("printf nope")
    assert blocked_shell["exit_code"] == 126
    assert blocked_shell["start_error_type"] == "PermissionError"
    assert blocked_background_shell["status"] == "failed_to_start"

    terminal.close()
    terminal.close()
    with pytest.raises(RuntimeError, match="closed"):
        terminal.run_command(["true"])


def test_terminal_output_boundaries_deadline_and_environment(
    terminal: TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = harbor_terminal._BytePreview(limit=4)
    preview.append(b"abcdef")
    assert preview.render() == ("abcdef", True, False)

    result = terminal.run_command([sys.executable, "-c", "print('done')"])
    process_id = cast(str, result["process_id"])
    with pytest.raises(RuntimeError, match="not accepting input"):
        terminal.write_process_input(process_id, "late")
    with pytest.raises(ValueError, match="stream must be"):
        terminal.read_process_output(process_id, stream=cast(Any, "invalid"))

    combined_path = Path(cast(str, result["combined_path"]))
    combined_path.unlink()
    assert terminal.read_process_output(process_id)["content"] == ""

    with pytest.raises(ValueError, match="positive"):
        terminal._bounded_timeout(0)
    terminal.deadline_monotonic = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="deadline"):
        terminal._bounded_timeout(None)
    terminal.deadline_monotonic = time.monotonic() + 1
    assert 0 < terminal._bounded_timeout(10) <= 1

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/credentials.json")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("VISIBLE_FIXTURE", "yes")
    child_env = harbor_terminal._sanitized_subprocess_env()
    assert "OPENAI_API_KEY" not in child_env
    assert "AWS_ACCESS_KEY_ID" not in child_env
    assert "AWS_SESSION_TOKEN" not in child_env
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in child_env
    assert "HF_TOKEN" not in child_env
    assert child_env["VISIBLE_FIXTURE"] == "yes"
    assert is_secret_name("AUTH_TOKEN")
    assert is_secret_name("token")
    assert not is_secret_name("TOKENIZERS_PARALLELISM")


def test_terminal_command_classification_and_process_group_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert harbor_terminal._is_verification_command("python3.14 -m pytest -q", shell=False)
    assert not harbor_terminal._is_verification_command("'", shell=False)
    assert not harbor_terminal._is_verification_command("", shell=False)
    assert not harbor_terminal._is_read_only_command("'", shell=False)
    assert not harbor_terminal._is_read_only_command("", shell=False)
    assert not harbor_terminal._is_read_only_command("echo changed", shell=False)
    assert harbor_terminal._is_read_only_command("cat file.txt", shell=False)

    fake = SimpleNamespace(process=SimpleNamespace(pid=999999))
    monkeypatch.setattr(
        harbor_terminal.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )
    TerminalSession._signal_process_group(cast(Any, fake), signal.SIGTERM)

    empty = harbor_terminal._empty_binary_file()
    with empty as handle:
        assert handle.seek(0) is None
        assert handle.read(1) == b""


def test_terminal_shell_close_and_termination_races(tmp_path: Path) -> None:
    session = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        preview_bytes=16,
        output_limit_bytes=1_000_000,
    )
    running = session.start_shell("sleep 30")
    session.hook_handler(
        PostToolUseEvent(
            run_id="run",
            sequence=1,
            adapter_shortname="pydantic",
            tool_call_id="read",
            tool_name="read_text_file",
            origin="kedi",
            arguments={},
            result="ok",
        )
    )
    session.close()
    assert session.process_status(cast(str, running["process_id"]))["status"] == "finished"

    kill_session = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "kill-records",
        preview_bytes=16,
        output_limit_bytes=1_000_000,
    )
    try:
        stubborn = kill_session.start_process(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            ]
        )
        time.sleep(0.1)
        stopped = kill_session.stop_process(cast(str, stubborn["process_id"]), grace_seconds=0)
        assert stopped["status"] == "finished"
        kill_session.stop_process(cast(str, stubborn["process_id"]), grace_seconds=0)
        managed = kill_session._process(cast(str, stubborn["process_id"]))
        managed.threads.append(threading.current_thread())
        kill_session._join_readers(managed)
        kill_session._terminate_for_output_limit(managed)
        kill_session._terminate_for_output_limit(managed)
        managed.process.stdin = None
        kill_session._terminate(managed, grace_seconds=0)
        assert (
            kill_session.read_process_output(cast(str, stubborn["process_id"]), stream="stderr")[
                "stream"
            ]
            == "stderr"
        )
    finally:
        kill_session.close()


def test_terminal_zero_output_budget_records_limit_without_payload(tmp_path: Path) -> None:
    session = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        preview_bytes=16,
        output_limit_bytes=0,
    )
    try:
        result = session.run_command([sys.executable, "-c", "print('discarded')"])
        assert result["output_limited"] is True
        assert result["stdout_bytes"] == 0
    finally:
        session.close()


def test_benchmark_policy_is_non_interactive_and_allowlisted() -> None:
    read = ApprovalRequest.create(
        tool_name="read_text_file",
        arguments={},
        risk="read_only",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(read).decision == "allow"
    write = ApprovalRequest.create(
        tool_name="apply_patch",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(write).decision == "allow"
    remove = ApprovalRequest.create(
        tool_name="remove_file",
        arguments={"file_path": "generated.tmp"},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(remove).decision == "allow"
    unknown = ApprovalRequest.create(
        tool_name="host_escape",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(unknown).decision == "deny"
    sensitive = ApprovalRequest.create(
        tool_name="read_text_file",
        arguments={"secret_files": True},
        risk="sensitive",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(sensitive).decision == "deny"


def test_agent_source_variants_parse() -> None:
    for mode, threshold in (("disabled", None), ("native", None), ("native", 100_000)):
        parse_program(
            runner.render_agent_source(
                history_enabled=True,
                compaction_mode=mode,
                compaction_threshold=threshold,
            ),
            source_path="<harbor-agent>",
        )
    with pytest.raises(ValueError, match="compaction_mode"):
        runner.render_agent_source(compaction_mode="other", compaction_threshold=None)
    with pytest.raises(ValueError, match="positive"):
        runner.render_agent_source(compaction_mode="native", compaction_threshold=0)
    with pytest.raises(ValueError, match="requires native"):
        runner.render_agent_source(compaction_mode="disabled", compaction_threshold=1)
    disabled = runner.render_agent_source(
        history_enabled=False,
        compaction_mode="disabled",
        compaction_threshold=None,
    )
    assert "> history: disabled" in disabled
    parse_program(disabled, source_path="<harbor-agent-stateless>")
    with pytest.raises(ValueError, match="requires history"):
        runner.render_agent_source(
            history_enabled=False,
            compaction_mode="native",
            compaction_threshold=None,
        )


def test_runner_executes_the_real_benchmark_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.models.test import TestModel

    from kedi.agent_adapter import PydanticAdapter

    real_bind = runner.kedi.bind

    def bind_with_test_model(**kwargs: Any) -> Any:
        kwargs["adapter"] = PydanticAdapter(TestModel(call_tools=[]))
        kwargs.pop("model", None)
        kwargs.pop("effort", None)
        return real_bind(**kwargs)

    monkeypatch.setattr(runner.kedi, "bind", bind_with_test_model)
    result_path = tmp_path / "logs" / "kedi-result.json"
    record = runner.run_trial(
        instruction="Inspect the workspace and report completion.",
        model="test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=result_path,
        trial_id="real-runtime-smoke",
    )

    assert record.state == "completed"
    assert record.usage["requests"] == 1
    assert record.usage["tool_calls"] == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "completed"


def test_runner_configures_logfire_only_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    fake_logfire = SimpleNamespace(
        configure=lambda **kwargs: calls.append(("configure", kwargs)),
        instrument_pydantic_ai=lambda: calls.append(("pydantic_ai", None)),
    )
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    assert runner._configure_logfire() is False
    assert calls == []

    monkeypatch.setenv("KEDI_HARBOR_LOGFIRE", "1")
    assert runner._configure_logfire() is True
    assert calls == [
        ("configure", {"service_name": "kedi", "send_to_logfire": "if-token-present"}),
        ("pydantic_ai", None),
    ]


def test_runner_writes_durable_completed_and_failed_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_bind(**kwargs: Any) -> Any:
        captured.update(kwargs)

        def decorate(_: Any) -> Any:
            def invoke(instruction: str) -> str:
                assert instruction == "fix it"
                run_command = next(
                    item for item in kwargs["tools"] if item.__name__ == "run_command"
                )
                assert run_command([sys.executable, "-c", "print('checked')"])["exit_code"] == 0
                return "done"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    result_path = tmp_path / "logs" / "result.json"
    record = runner.run_trial(
        instruction="fix it",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=result_path,
        trial_id="trial-1",
        manifest_digest="manifest",
    )
    assert record.state == "completed"
    assert record.result_summary == "done"
    assert record.manifest_digest == "manifest"
    assert record.failure_phase is None
    assert record.policy["history"] is True
    assert json.loads(result_path.read_text())["state"] == "completed"
    assert captured["skills"] is True
    assert captured["conversation"]._closed is True

    def failing_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                raise RuntimeError("sk-do-not-leak-this-secret")

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", failing_bind)
    failed = runner.run_trial(
        instruction="fail",
        model="openai/test",
        adapter="langchain",
        workspace=tmp_path,
        result_path=tmp_path / "failed" / "result.json",
        trial_id="trial-2",
    )
    assert failed.state == "agent_failure"
    assert failed.failure_phase == "agent"
    assert failed.error_type == "RuntimeError"
    assert "sk-do-not" not in cast(str, failed.error_message)


def test_runner_classifies_setup_failure_and_enforces_local_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_workspace = tmp_path / "missing"
    setup_failure = runner.run_trial(
        instruction="irrelevant",
        model="openai/test",
        adapter="pydantic",
        workspace=missing_workspace,
        result_path=tmp_path / "setup" / "result.json",
        trial_id="setup",
    )
    assert setup_failure.state == "integration_failure"
    assert setup_failure.failure_phase == "setup"

    def slow_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                time.sleep(5)
                return "too late"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", slow_bind)
    timed_out = runner.run_trial(
        instruction="wait",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "timeout" / "result.json",
        trial_id="timeout",
        trial_timeout_seconds=0.05,
    )
    assert timed_out.state == "timed_out"
    assert timed_out.failure_phase == "agent"


def test_runner_can_disable_history_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_bind(**kwargs: Any) -> Any:
        captured.update(kwargs)

        def decorate(_: Any) -> Any:
            return lambda _: "done"

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    record = runner.run_trial(
        instruction="stateless",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "stateless" / "result.json",
        trial_id="stateless",
        history_enabled=False,
        artifacts_enabled=False,
    )
    assert record.state == "completed"
    assert captured["conversation"] is None
    assert captured["artifacts"] is False
    assert "stream_process_output" not in {item.__name__ for item in captured["tools"]}


def test_runner_records_usage_cancellation_and_invalid_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reporting_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                report_invocation(
                    InvocationUsage(
                        requests=1,
                        tool_calls=2,
                        input_tokens=10,
                        cache_read_tokens=6,
                        output_tokens=3,
                    )
                )
                return "done"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", reporting_bind)
    completed = runner.run_trial(
        instruction="measure",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "usage" / "result.json",
        trial_id="usage",
    )
    assert completed.usage["requests"] == 1
    assert completed.usage["cache_read_tokens"] == 6

    def cancelled_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                raise KeyboardInterrupt

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", cancelled_bind)
    cancelled = runner.run_trial(
        instruction="cancel",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "cancelled" / "result.json",
        trial_id="cancelled",
    )
    assert cancelled.state == "cancelled"
    assert cancelled.error_type == "KeyboardInterrupt"

    invalid = runner.run_trial(
        instruction="invalid",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "invalid-timeout" / "result.json",
        trial_id="invalid-timeout",
        trial_timeout_seconds=0,
    )
    assert invalid.state == "integration_failure"
    assert invalid.failure_phase == "setup"


def test_runner_cleanup_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_terminal = runner.TerminalSession

    class CleanupFailureTerminal(original_terminal):
        def close(self) -> None:
            super().close()
            raise RuntimeError("cleanup failed")

    def fake_bind(**_: Any) -> Any:
        return lambda _: lambda _instruction: "done"

    monkeypatch.setattr(runner, "TerminalSession", CleanupFailureTerminal)
    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    record = runner.run_trial(
        instruction="cleanup",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "cleanup" / "result.json",
        trial_id="cleanup",
    )
    assert record.state == "integration_failure"
    assert record.failure_phase == "teardown"
    assert record.error_type == "CleanupError"
    assert "terminal: RuntimeError" in (tmp_path / "cleanup" / "cleanup-errors.txt").read_text()

    def failing_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                raise RuntimeError("agent failed")

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", failing_bind)
    failed = runner.run_trial(
        instruction="fail and cleanup",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "failed-cleanup" / "result.json",
        trial_id="failed-cleanup",
    )
    assert failed.state == "agent_failure"
    assert failed.failure_phase == "agent"
    assert (tmp_path / "failed-cleanup" / "cleanup-errors.txt").is_file()


def test_runner_helpers_signal_state_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner._mcp_servers(None) == ()
    assert runner._settings(None) == {}
    assert runner._effort(None) is None
    assert runner._effort("high") == "high"
    with pytest.raises(ValueError, match="effort must be"):
        runner._effort("impossible")

    mcp = tmp_path / "mcp.json"
    mcp.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON list"):
        runner._mcp_servers(mcp)
    mcp.write_text("[1]", encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        runner._mcp_servers(mcp)

    settings = tmp_path / "settings.json"
    settings.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        runner._settings(settings)
    settings.write_text('{"temperature": 0}', encoding="utf-8")
    assert runner._settings(settings) == {"temperature": 0}

    closed: list[bool] = []
    signal_state = runner._SignalState(
        cast(TerminalSession, SimpleNamespace(close=lambda: closed.append(True)))
    )
    with pytest.raises(runner._TrialTimedOut):
        signal_state._handle(signal.SIGTERM, None)
    with pytest.raises(KeyboardInterrupt):
        signal_state._handle(signal.SIGINT, None)
    assert closed == [True, True]

    inactive_state = runner._SignalState(cast(TerminalSession, SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(runner, "threading_is_main", lambda: False)
    inactive_state.install(timeout_seconds=1)
    inactive_state.restore()

    with monkeypatch.context() as context:
        context.setattr(runner, "threading_is_main", lambda: True)
        context.delattr(runner.signal, "SIGALRM")
        no_alarm_state = runner._SignalState(
            cast(TerminalSession, SimpleNamespace(close=lambda: None))
        )
        no_alarm_state.install(timeout_seconds=None)
        no_alarm_state.restore()

    signals: list[tuple[int, Any]] = []
    timers: list[tuple[Any, ...]] = []
    monkeypatch.setattr(runner, "threading_is_main", lambda: True)
    monkeypatch.setattr(runner.signal, "getsignal", lambda signum: f"handler-{signum}")
    monkeypatch.setattr(
        runner.signal, "signal", lambda signum, handler: signals.append((signum, handler))
    )

    def fake_setitimer(*args: Any) -> tuple[float, float]:
        timers.append(args)
        return (2.0, 0.5)

    monkeypatch.setattr(runner.signal, "setitimer", fake_setitimer)
    active_state = runner._SignalState(cast(TerminalSession, SimpleNamespace(close=lambda: None)))
    active_state.install(timeout_seconds=1)
    active_state.restore()
    assert timers[-1] == (signal.ITIMER_REAL, 2.0, 0.5)
    assert signals

    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")),
    )
    runner._write_error(tmp_path, RuntimeError("ignored"))
    monkeypatch.undo()

    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    result = tmp_path / "main-result.json"
    captured: dict[str, Any] = {}

    def fake_trial(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(state="completed")

    monkeypatch.setattr(runner, "run_trial", fake_trial)
    assert (
        runner.main(
            [
                "--instruction-file",
                str(instruction),
                "--model",
                "openai/test",
                "--result-file",
                str(result),
                "--trial-id",
                "trial",
            ]
        )
        == 0
    )
    assert captured["instruction"] == "solve"
    monkeypatch.setattr(runner, "run_trial", lambda **_: SimpleNamespace(state="agent_failure"))
    assert (
        runner.main(
            [
                "--instruction-file",
                str(instruction),
                "--model",
                "openai/test",
                "--result-file",
                str(result),
                "--trial-id",
                "trial",
            ]
        )
        == 1
    )


def test_host_builds_exact_harbor_command(tmp_path: Path) -> None:
    manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/gpt-test",
        environment="daytona",
        task_names=("task-a", "terminal-bench/task-b"),
        tree_sitter_kedi_revision="a" * 40,
        concurrency=2,
        harbor_policy=HarborRunPolicy(
            timeout_multiplier=1.25,
            agent_timeout_multiplier=1.5,
            verifier_timeout_multiplier=1.75,
            agent_setup_timeout_multiplier=2.0,
            environment_build_timeout_multiplier=2.25,
            max_retries=2,
            retry_include_exceptions=("ApiRateLimitError",),
            retry_exclude_exceptions=("AgentTimeoutError",),
        ),
        policy=HarnessPolicy(compaction_mode="native", compaction_threshold=100_000),
    )
    wheel = tmp_path / "kedi.whl"
    wheel.touch()
    command = host._harbor_run_command(
        manifest,
        harbor_command="harbor",
        kedi_wheel=wheel,
        kedi_package=None,
        jobs_dir=tmp_path / "jobs",
        job_name="pilot",
    )
    assert command[:3] == ["harbor", "run", "--dataset"]
    assert command[3] == "terminal-bench/terminal-bench-2-1@6"
    assert "kedi.integrations.harbor:KediAgent" in command
    assert f"tree_sitter_kedi_revision={'a' * 40}" in command
    assert command.count("--include-task-name") == 2
    task_filters = [
        command[index + 1] for index, value in enumerate(command) if value == "--include-task-name"
    ]
    assert task_filters == ["terminal-bench/task-a", "terminal-bench/task-b"]
    assert f"manifest_digest={manifest.material_digest}" in command
    assert "history_enabled=true" in command
    assert "artifacts_enabled=true" in command
    assert command[command.index("--timeout-multiplier") + 1] == "1.25"
    assert command[command.index("--agent-timeout-multiplier") + 1] == "1.5"
    assert command[command.index("--verifier-timeout-multiplier") + 1] == "1.75"
    assert command[command.index("--agent-setup-timeout-multiplier") + 1] == "2.0"
    assert command[command.index("--environment-build-timeout-multiplier") + 1] == "2.25"
    assert command[command.index("--max-retries") + 1] == "2"
    assert command[command.index("--retry-include") + 1] == "ApiRateLimitError"
    assert command[command.index("--retry-exclude") + 1] == "AgentTimeoutError"

    package_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_package="kedi==fixture",
        model="openai/test",
        effort="high",
        model_settings={"temperature": 0},
    )
    minimal = host._harbor_run_command(
        package_manifest,
        harbor_command="harbor",
        kedi_wheel=None,
        kedi_package="kedi==fixture",
        jobs_dir=None,
        job_name=None,
    )
    assert "effort=high" in minimal
    assert 'model_settings={"temperature":0}' in minimal
    assert "kedi_package=kedi==fixture" in minimal
    assert "--jobs-dir" not in minimal
    assert "--job-name" not in minimal
    no_distribution = host._harbor_run_command(
        package_manifest,
        harbor_command="harbor",
        kedi_wheel=None,
        kedi_package=None,
        jobs_dir=None,
        job_name=None,
    )
    assert not any(value.startswith("kedi_package=") for value in no_distribution)


def test_host_persists_manifest_next_to_harbor_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/gpt-test",
        task_names=("task-a",),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> Any:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(host, "_resolve_executable", lambda *_args, **_kwargs: "harbor")
    monkeypatch.setattr(host, "_validate_harbor_version", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(host.subprocess, "run", fake_run)
    result = CliRunner().invoke(
        host.cli,
        [
            "run",
            str(manifest_path),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "pilot",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls and calls[0][:2] == ["harbor", "run"]
    persisted = load_manifest(tmp_path / "jobs" / "pilot" / "kedi-manifest.json")
    assert persisted.material_digest == manifest.material_digest


def test_host_rejects_a_different_harbor_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0.21.0\n"),
    )
    with pytest.raises(ClickException, match="does not match"):
        host._validate_harbor_version("harbor", expected="0.22.0")


def test_resume_requires_and_validates_the_original_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    monkeypatch.setattr(host, "_resolve_executable", lambda *_args, **_kwargs: "harbor")
    missing = CliRunner().invoke(host.cli, ["resume", str(job_dir)])
    assert missing.exit_code != 0
    assert "manifest is missing" in missing.output

    RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/test",
    ).write(job_dir / "kedi-manifest.json")
    validated: list[tuple[str, str]] = []
    calls: list[list[str]] = []
    monkeypatch.setattr(
        host,
        "_validate_harbor_version",
        lambda executable, *, expected: validated.append((executable, expected)),
    )
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    resumed = CliRunner().invoke(host.cli, ["resume", str(job_dir)])
    assert resumed.exit_code == 0, resumed.output
    assert validated == [("harbor", "0.22.0")]
    assert calls == [["harbor", "job", "resume", "-p", str(job_dir.resolve())]]


def test_manifest_cli_reports_invalid_model_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(host, "current_git_state", lambda _: ("commit", False))
    result = CliRunner().invoke(
        host.cli,
        [
            "manifest",
            "--output",
            str(tmp_path / "manifest.json"),
            "--harbor-revision",
            "harbor-commit",
            "--model",
            "openai/test",
            "--task",
            "task-a",
            "--model-settings",
            str(settings),
        ],
    )
    assert result.exit_code != 0
    assert "could not read model settings" in result.output


def test_manifest_cli_writes_complete_policy_and_rejects_dirty_or_conflicting_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"wheel")
    settings = tmp_path / "settings.json"
    settings.write_text('{"temperature": 0}', encoding="utf-8")
    monkeypatch.setattr(host, "current_git_state", lambda _: ("commit", False))
    output = tmp_path / "manifest.json"
    created = CliRunner().invoke(
        host.cli,
        [
            "manifest",
            "--output",
            str(output),
            "--harbor-revision",
            "harbor-commit",
            "--model",
            "openai/test",
            "--task",
            "task-a",
            "--kedi-wheel",
            str(wheel),
            "--model-settings",
            str(settings),
            "--retry-all-exceptions",
            "--max-retries",
            "2",
            "--compaction-mode",
            "native",
            "--compaction-threshold",
            "1000",
        ],
    )
    assert created.exit_code == 0, created.output
    manifest = load_manifest(output)
    assert manifest.kedi_wheel_sha256 == sha256_file(wheel)
    assert manifest.tree_sitter_kedi_revision == current_gitlink_revision(
        Path.cwd(), "tree-sitter-kedi"
    )
    assert manifest.harbor_policy.retry_exclude_exceptions == ()
    assert manifest.model_settings == {"temperature": 0}

    unused_wheel = tmp_path / "unused.whl"
    unused_wheel.write_bytes(b"unused")
    conflict = CliRunner().invoke(
        host.cli,
        [
            "manifest",
            "--output",
            str(tmp_path / "conflict.json"),
            "--harbor-revision",
            "harbor-commit",
            "--model",
            "openai/test",
            "--task",
            "task-a",
            "--kedi-wheel",
            str(wheel),
            "--kedi-package",
            "kedi",
        ],
    )
    assert conflict.exit_code != 0
    assert "mutually exclusive" in conflict.output

    monkeypatch.setattr(host, "current_git_state", lambda _: ("commit", True))
    dirty = CliRunner().invoke(
        host.cli,
        [
            "manifest",
            "--output",
            str(tmp_path / "dirty.json"),
            "--harbor-revision",
            "harbor-commit",
            "--model",
            "openai/test",
            "--task",
            "task-a",
        ],
    )
    assert dirty.exit_code != 0
    assert "worktree is dirty" in dirty.output

    monkeypatch.setattr(host, "current_git_state", lambda _: ("commit", False))
    package_output = tmp_path / "package-manifest.json"
    package_created = CliRunner().invoke(
        host.cli,
        [
            "manifest",
            "--output",
            str(package_output),
            "--harbor-revision",
            "harbor-commit",
            "--model",
            "openai/test",
            "--task",
            "task-a",
            "--kedi-package",
            "kedi==fixture",
        ],
    )
    assert package_created.exit_code == 0, package_created.output
    assert load_manifest(package_output).model_settings == {}


def test_host_dry_run_and_subprocess_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_package="kedi==0.4.0",
        model="openai/test",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    dry = CliRunner().invoke(
        host.cli,
        ["run", str(manifest_path), "--dry-run", "--harbor-command", "missing-harbor"],
    )
    assert dry.exit_code == 0, dry.output
    assert dry.output.startswith("missing-harbor run")

    monkeypatch.setattr(host, "_resolve_executable", lambda *_args, **_kwargs: "harbor")
    monkeypatch.setattr(host, "_validate_harbor_version", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )
    failed = CliRunner().invoke(
        host.cli,
        [
            "run",
            str(manifest_path),
            "--jobs-dir",
            str(tmp_path / "failed-jobs"),
            "--job-name",
            "failed-job",
        ],
    )
    assert failed.exit_code != 0
    assert "Harbor exited with status 7" in failed.output

    job = tmp_path / "job"
    job.mkdir()
    manifest.write(job / "kedi-manifest.json")
    resumed = CliRunner().invoke(host.cli, ["resume", str(job)])
    assert resumed.exit_code != 0
    assert "Harbor resume exited with status 7" in resumed.output

    unused_wheel = tmp_path / "unused.whl"
    unused_wheel.write_bytes(b"unused")
    conflict = CliRunner().invoke(
        host.cli,
        [
            "run",
            str(manifest_path),
            "--kedi-wheel",
            str(unused_wheel),
            "--kedi-package",
            "kedi==0.4.0",
        ],
    )
    assert conflict.exit_code != 0


def test_host_executable_version_and_distribution_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absolute = tmp_path / "harbor"
    assert host._resolve_executable(str(absolute), allow_missing=True) == str(absolute)
    with pytest.raises(ClickException, match="not found"):
        host._resolve_executable(str(absolute))
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    assert host._resolve_executable("harbor", allow_missing=True) == "harbor"
    with pytest.raises(ClickException, match="not installed"):
        host._resolve_executable("harbor")
    monkeypatch.setattr(host.shutil, "which", lambda _: "/bin/harbor")
    assert host._resolve_executable("harbor") == "/bin/harbor"

    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )
    with pytest.raises(ClickException, match="could not determine"):
        host._validate_harbor_version("harbor", expected="0.22.0")
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0.22.0\n"),
    )
    host._validate_harbor_version("harbor", expected="0.22.0")

    package_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_package="kedi==fixture",
        model="openai/test",
    )
    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(UsageError, match="does not record a wheel"):
        host._resolve_kedi_distribution(
            package_manifest,
            kedi_wheel=wheel,
            kedi_package=None,
        )
    with pytest.raises(ClickException, match="does not match"):
        host._resolve_kedi_distribution(
            package_manifest,
            kedi_wheel=None,
            kedi_package="kedi==other",
        )
    assert host._resolve_kedi_distribution(
        package_manifest,
        kedi_wheel=None,
        kedi_package="kedi==fixture",
    ) == (None, "kedi==fixture")

    wheel_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_wheel_sha256=sha256_file(wheel),
        model="openai/test",
    )
    with pytest.raises(UsageError, match="recorded Kedi wheel"):
        host._resolve_kedi_distribution(
            wheel_manifest,
            kedi_wheel=wheel,
            kedi_package="kedi",
        )


def test_host_main_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str] | None, str]] = []
    dotenv_calls = 0

    def fake_main(*, args: list[str] | None, prog_name: str) -> None:
        calls.append((args, prog_name))

    def fake_load_dotenv() -> None:
        nonlocal dotenv_calls
        dotenv_calls += 1

    monkeypatch.setattr(host.cli, "main", fake_main)
    monkeypatch.setattr(host.dotenv, "load_dotenv", fake_load_dotenv)
    assert host.main(["--help"]) == 0
    assert calls == [(["--help"], "kedi-terminal-bench")]
    assert dotenv_calls == 1


def test_runner_preserves_mcp_transport_configuration(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            [
                {
                    "transport": "stdio",
                    "command": "server",
                    "args": ["--mode", "test"],
                    "env": {"SERVER_MODE": "fixture"},
                },
                {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                    "headers": {"X-Test": "fixture"},
                },
            ]
        ),
        encoding="utf-8",
    )
    stdio, http = runner._mcp_servers(config)
    assert stdio.command == "server"
    assert stdio.args == ("--mode", "test")
    assert stdio.env == {"SERVER_MODE": "fixture"}
    assert http.transport == "http"
    assert http.url == "https://example.test/mcp"
    assert http.headers == {"X-Test": "fixture"}


def test_host_requires_the_distribution_recorded_by_manifest(tmp_path: Path) -> None:
    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"exact-wheel")
    manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_wheel_sha256=sha256_file(wheel),
        model="openai/gpt-test",
    )
    resolved, package = host._resolve_kedi_distribution(
        manifest,
        kedi_wheel=wheel,
        kedi_package=None,
    )
    assert resolved == wheel.resolve()
    assert package is None

    changed = tmp_path / "changed.whl"
    changed.write_bytes(b"different-wheel")
    with pytest.raises(ClickException, match="SHA-256"):
        host._resolve_kedi_distribution(
            manifest,
            kedi_wheel=changed,
            kedi_package=None,
        )
    with pytest.raises(UsageError, match="requires --kedi-wheel"):
        host._resolve_kedi_distribution(
            manifest,
            kedi_wheel=None,
            kedi_package=None,
        )


def _install_fake_harbor(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    modules = {
        name: ModuleType(name)
        for name in (
            "harbor",
            "harbor.agents",
            "harbor.agents.installed",
            "harbor.agents.installed.base",
            "harbor.agents.model_connection",
            "harbor.environments",
            "harbor.environments.base",
            "harbor.models",
            "harbor.models.agent",
            "harbor.models.agent.context",
        )
    }

    class ModelConnectionSpec:
        def __init__(self, *, passthrough: bool = False) -> None:
            self.passthrough = passthrough

    class BaseEnvironment:
        pass

    class AgentContext:
        pass

    class BaseInstalledAgent:
        def __init__(
            self,
            *,
            logs_dir: Path,
            model_name: str | None = None,
            environment_logs_dir: PurePosixPath | None = None,
            skills_dir: str | None = None,
            mcp_servers: list[Any] | None = None,
            **_: Any,
        ) -> None:
            self.logs_dir = logs_dir
            self.model_name = model_name
            self.environment_logs_dir = environment_logs_dir or PurePosixPath("/logs/agent")
            self.skills_dir = skills_dir
            self.mcp_servers = mcp_servers or []
            self.context_id = None
            self.session_id = "session"
            self.model_connection = SimpleNamespace(
                env={"OPENROUTER_API_KEY": "sk-provider-secret-value"}
            )

        async def ensure_system_dependencies(
            self, environment: Any, dependencies: tuple[str, ...]
        ) -> None:
            environment.dependencies = dependencies

        async def exec_as_root(self, environment: Any, **kwargs: Any) -> Any:
            environment.root_calls.append(kwargs)
            return SimpleNamespace(return_code=0)

        async def exec_as_agent(self, environment: Any, **kwargs: Any) -> Any:
            environment.agent_calls.append(kwargs)
            return SimpleNamespace(return_code=0)

        async def _upload_config_text(self, environment: Any, **kwargs: Any) -> None:
            environment.configs.append(kwargs)

        def render_instruction(self, instruction: str) -> str:
            return f"rendered: {instruction}"

    setattr(modules["harbor.agents.installed.base"], "BaseInstalledAgent", BaseInstalledAgent)
    setattr(
        modules["harbor.agents.model_connection"],
        "ModelConnectionSpec",
        ModelConnectionSpec,
    )
    setattr(modules["harbor.environments.base"], "BaseEnvironment", BaseEnvironment)
    setattr(modules["harbor.models.agent.context"], "AgentContext", AgentContext)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return BaseInstalledAgent


def test_harbor_agent_lifecycle_and_context_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_HARBOR_LOGFIRE", "1")
    monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-test-token")
    base = _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    KediAgent = agent_module.KediAgent
    import kedi.integrations.harbor as integration

    assert integration.__getattr__("KediAgent") is KediAgent
    assert issubclass(KediAgent, base)

    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"wheel")
    environment = SimpleNamespace(
        dependencies=(),
        root_calls=[],
        agent_calls=[],
        configs=[],
        uploaded=[],
    )

    async def upload_file(local: Path, remote: str) -> None:
        environment.uploaded.append((Path(local), remote))

    environment.upload_file = upload_file
    with pytest.raises(ValueError, match="lowercase Git revision"):
        KediAgent(
            logs_dir=tmp_path / "invalid-logs",
            model_name="openrouter/test-model",
            tree_sitter_kedi_revision="ABC",
        )
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="openrouter/test-model",
        kedi_wheel=wheel,
        history_enabled=False,
        prefix_cache_enabled=False,
        artifacts_enabled=False,
        runner_timeout_seconds=10,
        manifest_digest="manifest",
        tree_sitter_kedi_revision="a" * 40,
    )
    asyncio.run(agent.install(environment))
    assert environment.uploaded[0][1].endswith("/kedi.whl")
    assert "python_venv" in environment.dependencies
    assert "build_tools" in environment.dependencies
    assert (
        "tree-sitter-kedi @ git+https://github.com/kedi-lang/tree-sitter-kedi.git@" + "a" * 40
    ) in environment.root_calls[-1]["command"]
    install_command = environment.root_calls[-1]["command"]
    assert "sys.version_info < (3, 10)" in install_command
    assert "uv==0.11.21" in install_command
    assert "uv venv --python 3.11 --managed-python --seed" in install_command
    assert "logfire[httpx]==4.41.0" in install_command

    context = SimpleNamespace(metadata=None)
    asyncio.run(agent.run("solve task", environment, context))
    assert context.metadata is None
    assert environment.configs[0]["content"] == "rendered: solve task"
    run_call = environment.agent_calls[-1]
    assert run_call["timeout_sec"] == 40
    assert run_call["command"].startswith("set -o pipefail; ")
    assert "--no-history-enabled" in run_call["command"]
    assert "--no-artifacts-enabled" in run_call["command"]
    assert "sk-provider-secret-value" not in run_call["command"]
    assert "logfire-test-token" not in run_call["command"]
    assert run_call["env"]["KEDI_HARBOR_LOGFIRE"] == "1"
    assert run_call["env"]["LOGFIRE_TOKEN"] == "logfire-test-token"

    agent.logs_dir.mkdir(parents=True)
    (agent.logs_dir / "kedi-result.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "adapter": "pydantic",
                "usage": {
                    "requests": 2,
                    "tool_calls": 3,
                    "input_tokens": 100,
                    "cache_read_tokens": 60,
                    "cache_write_tokens": 20,
                    "output_tokens": 10,
                    "uncached_input_tokens": 40,
                    "cost_usd": 0.01,
                },
                "verification": {"state": "passed"},
            }
        ),
        encoding="utf-8",
    )
    agent.populate_context_post_run(context)
    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 60
    assert context.n_output_tokens == 10
    assert context.cost_usd == 0.01
    assert context.metadata["kedi_state"] == "completed"


def test_harbor_agent_transfers_codex_auth_privately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    KediAgent = agent_module.KediAgent

    auth_file = tmp_path / "auth.json"
    source_secret = "source-refresh-secret-canary"
    access_secret = "short-lived-access-canary"
    auth_file.write_text(json.dumps({"tokens": {"refresh_token": source_secret}}))
    helper = ModuleType("codex_auth_helper")
    helper_calls: list[Any] = []

    class CodexAuthConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.auth_path = kwargs["auth_path"]
            self.refresh_margin = kwargs["refresh_margin"]
            helper_calls.append(self)

    class CodexAuthState:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class CodexAuthStore:
        def __init__(self, path: Path) -> None:
            self.path = path

        def write_state(self, state: Any) -> None:
            self.path.write_text(
                json.dumps(
                    {
                        "auth_mode": state.auth_mode,
                        "tokens": {
                            "access_token": state.access_token,
                            "account_id": state.account_id,
                            "refresh_token": state.refresh_token,
                        },
                    }
                )
            )

    state = SimpleNamespace(
        account_id="account-fixture",
        auth_mode="chatgpt",
        expires_at=None,
        last_refresh=None,
    )

    class TokenManager:
        current_state = state

        async def get_access_token(self) -> str:
            return access_secret

    class Client:
        token_manager = TokenManager()

        async def close(self) -> None:
            helper_calls.append("closed")

    setattr(helper, "CodexAuthConfig", CodexAuthConfig)
    setattr(helper, "CodexAuthState", CodexAuthState)
    setattr(helper, "CodexAuthStore", CodexAuthStore)
    setattr(helper, "create_codex_async_openai", lambda **_: Client())
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)
    environment = SimpleNamespace(
        dependencies=(),
        root_calls=[],
        agent_calls=[],
        configs=[],
        uploaded=[],
        uploaded_contents={},
        default_user="agent",
    )

    async def upload_file(local: Path, remote: str) -> None:
        environment.uploaded.append((Path(local), remote))
        environment.uploaded_contents[remote] = Path(local).read_text()

    environment.upload_file = upload_file
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="codex/gpt-5.6-luna",
        kedi_package="kedi==fixture",
        codex_auth_file=auth_file,
    )

    asyncio.run(agent.install(environment))
    install_command = environment.root_calls[-1]["command"]
    assert "codex-auth-helper==1.6.1" in install_command
    assert source_secret not in install_command

    asyncio.run(agent.run("solve", environment, SimpleNamespace(metadata=None)))

    uploaded_auth = [
        item for item in environment.uploaded if item[1] == "/installed-agent/kedi-codex-auth.json"
    ]
    assert len(uploaded_auth) == 1
    assert uploaded_auth[0][0] != auth_file.resolve()
    uploaded_payload = environment.uploaded_contents["/installed-agent/kedi-codex-auth.json"]
    assert access_secret in uploaded_payload
    assert "kedi-ephemeral-no-refresh" in uploaded_payload
    assert source_secret not in uploaded_payload
    assert helper_calls[-1] == "closed"
    assert helper_calls[0].refresh_margin == timedelta(minutes=30)
    assert "chmod 600" in environment.root_calls[-1]["command"]
    assert "chown agent" in environment.root_calls[-1]["command"]
    run_call = environment.agent_calls[-1]
    assert run_call["env"]["KEDI_CODEX_AUTH_FILE"] == ("/installed-agent/kedi-codex-auth.json")
    assert "trap" in run_call["command"]
    assert "rm -f -- /installed-agent/kedi-codex-auth.json" in run_call["command"]
    assert access_secret not in run_call["command"]
    assert access_secret not in json.dumps(run_call["env"])

    environment.default_user = None
    asyncio.run(
        agent._upload_private_file(
            environment,
            source=auth_file,
            destination=PurePosixPath("/installed-agent/no-owner-auth.json"),
        )
    )
    assert environment.root_calls[-1]["command"] == (
        "chmod 600 /installed-agent/no-owner-auth.json"
    )


def test_harbor_agent_rejects_missing_codex_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent

    with pytest.raises(FileNotFoundError, match="codex login"):
        KediAgent(
            logs_dir=tmp_path,
            model_name="codex/gpt-5.6-luna",
            kedi_package="kedi",
            codex_auth_file=tmp_path / "missing-auth.json",
        )


def test_harbor_agent_full_configuration_and_failure_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    KediAgent = agent_module.KediAgent

    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(ValueError, match="mutually exclusive"):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            kedi_package="kedi",
            kedi_wheel=wheel,
        )
    with pytest.raises(FileNotFoundError):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            kedi_wheel=tmp_path / "missing.whl",
        )
    with pytest.raises(ValueError, match="runner_timeout_seconds"):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            kedi_package="kedi",
            runner_timeout_seconds=0,
        )

    assert KediAgent.name() == "kedi"
    environment = SimpleNamespace(
        dependencies=(),
        root_calls=[],
        agent_calls=[],
        configs=[],
        uploaded=[],
    )

    async def upload_file(local: Path, remote: str) -> None:
        environment.uploaded.append((Path(local), remote))

    environment.upload_file = upload_file
    mcp = SimpleNamespace(model_dump=lambda **_: {"transport": "stdio", "command": "server"})
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        kedi_package="kedi==fixture",
        effort="high",
        model_settings={"temperature": 0},
        compaction_mode="native",
        compaction_threshold=1000,
        skills_dir="/skills",
        mcp_servers=[mcp],
    )
    assert "from kedi import __version__" in cast(str, agent.get_version_command())
    asyncio.run(agent.install(environment))
    assert environment.uploaded == []
    assert "build_tools" not in environment.dependencies
    assert "kedi==fixture" in environment.root_calls[-1]["command"]

    context = SimpleNamespace(metadata={"existing": True})
    asyncio.run(agent.run("solve", environment, context))
    assert context.metadata == {"existing": True}
    assert len(environment.configs) == 3
    assert "mkdir -p" in environment.agent_calls[0]["command"]
    command = environment.agent_calls[-1]["command"]
    assert "--mcp-config" in command
    assert "--model-settings" in command
    assert "--effort high" in command
    assert "--compaction-threshold 1000" in command

    missing_context = SimpleNamespace(metadata={"existing": True})
    agent.populate_context_post_run(missing_context)
    assert missing_context.metadata == {
        "existing": True,
        "kedi_state": "integration_failure",
        "kedi_result_present": False,
    }

    agent.logs_dir.mkdir(parents=True)
    result_path = agent.logs_dir / "kedi-result.json"
    result_path.write_text("{broken", encoding="utf-8")
    invalid_context = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(invalid_context)
    assert invalid_context.metadata["kedi_result_error"] == "JSONDecodeError"

    result_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "adapter": "pydantic",
                "usage": {
                    "input_tokens": True,
                    "cache_read_tokens": "invalid",
                    "output_tokens": None,
                    "cost_usd": False,
                },
            }
        ),
        encoding="utf-8",
    )
    invalid_usage_context = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(invalid_usage_context)
    assert invalid_usage_context.n_input_tokens is None
    assert invalid_usage_context.n_cache_tokens is None
    assert invalid_usage_context.n_output_tokens is None
    assert invalid_usage_context.cost_usd is None

    missing_model = KediAgent(logs_dir=tmp_path, model_name=None, kedi_package="kedi")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        asyncio.run(missing_model.run("solve", environment, SimpleNamespace()))


def test_real_harbor_contract_when_extra_is_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harbor_base = pytest.importorskip("harbor.agents.installed.base")
    BaseInstalledAgent = harbor_base.BaseInstalledAgent
    AgentContext = importlib.import_module("harbor.models.agent.context").AgentContext

    sys.modules.pop("kedi.integrations.harbor.agent", None)
    import kedi.integrations.harbor as integration

    integration.__dict__.pop("agent", None)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent

    agent = KediAgent(
        logs_dir=tmp_path,
        model_name="openrouter/test-model",
        kedi_package="kedi",
    )
    assert isinstance(agent, BaseInstalledAgent)
    assert agent.name() == "kedi"
    assert agent.MODEL_CONNECTION is not None
    assert agent.MODEL_CONNECTION.passthrough is True

    async def no_upload(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def no_exec(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(return_code=0)

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-provider-key")
    monkeypatch.setattr(KediAgent, "_upload_config_text", no_upload)
    monkeypatch.setattr(KediAgent, "_install_skills", no_upload)
    monkeypatch.setattr(KediAgent, "exec_as_agent", no_exec)
    context = AgentContext()
    asyncio.run(agent.run("solve", SimpleNamespace(), context))
    assert context.is_empty(), "Harbor must own the post-run context projection lifecycle"


def test_lazy_harbor_import_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.integrations.harbor as integration

    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    try:
        importlib.import_module("harbor.agents.installed.base")
    except ImportError:
        with pytest.raises(ImportError, match="terminal-bench"):
            integration.__getattr__("KediAgent")
    else:
        assert integration.__getattr__("KediAgent").__name__ == "KediAgent"
    with pytest.raises(AttributeError):
        integration.__getattr__("missing")


def test_lazy_harbor_import_preserves_unrelated_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.integrations.harbor as integration

    original_import = builtins.__import__

    def fail_agent_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "agent" and level == 1:
            raise ImportError("unrelated dependency failed", name="unrelated_dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_agent_import)
    with pytest.raises(ImportError, match="unrelated dependency"):
        integration.__getattr__("KediAgent")
