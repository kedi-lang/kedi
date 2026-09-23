from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, cast

import pytest
from click import ClickException, UsageError
from click.testing import CliRunner

import kedi.integrations.harbor.terminal.audit as terminal_audit_module
import kedi.integrations.harbor.terminal.output as terminal_output_module
import kedi.integrations.harbor.terminal.processes as terminal_processes_module
import kedi.integrations.harbor.terminal.session as terminal_session_module
from kedi.agent_adapter.approvals.policy import ApprovalRequest
from kedi.agent_adapter.hooks import (
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseRequest,
    UserPromptSubmitRequest,
)
from kedi.agent_adapter.invocation import InvocationUsage, report_invocation
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.integrations.harbor import host, runner
from kedi.integrations.harbor import process_supervisor as harbor_process_supervisor
from kedi.integrations.harbor.records import (
    HarborRunPolicy,
    HarnessPolicy,
    RunManifest,
    TrialRecord,
    atomic_write_json,
    current_git_state,
    current_gitlink_revision,
    invocation_usage_dict,
    is_secret_name,
    load_manifest,
    redact_text,
    sha256_file,
)
from kedi.integrations.harbor.terminal.session import TerminalSession
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

    source_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/gpt-test",
        tree_sitter_kedi_revision="a" * 40,
        tree_sitter_kedi_distribution_sha256="b" * 64,
    )
    assert source_manifest.material_digest != manifest.material_digest
    assert (
        replace(manifest, kedi_otel_distribution_sha256="d" * 64).material_digest
        != manifest.material_digest
    )

    codex_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="codex/gpt-test",
        codex_auth_helper_distribution_sha256="c" * 64,
    )
    assert (
        codex_manifest.material_digest
        != RunManifest(
            harbor_revision="harbor-commit",
            kedi_commit="kedi-commit",
            kedi_dirty=False,
            model="codex/gpt-test",
        ).material_digest
    )

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
        ({"retained_process_timeout_seconds": 0}, "positive"),
        ({"retained_process_timeout_seconds": float("inf")}, "positive"),
        ({"history": False}, "prefix_cache requires history"),
        ({"prefix_cache": False}, "stateful Kedi history"),
        ({"subagents": True}, "does not define subagents"),
        ({"dynamic_workflows": True}, "does not define subagents"),
        ({"artifact_threshold_bytes": 0}, "positive"),
        ({"artifact_quota_bytes": 0}, "positive"),
        ({"model_request_limit": 0}, "positive"),
        ({"model_cumulative_input_tokens": 0}, "positive"),
        ({"history_archive_threshold_tokens": 0}, "positive"),
        ({"history_archive_preserve_tokens": 0}, "positive"),
        (
            {
                "history_archive_threshold_tokens": 10_000,
                "history_archive_preserve_tokens": 10_000,
            },
            "lower than",
        ),
        (
            {"artifacts": False, "history_archive_threshold_tokens": 32_000},
            "requires artifacts",
        ),
        (
            {
                "compaction_mode": "native",
                "history_archive_threshold_tokens": 32_000,
            },
            "mutually exclusive",
        ),
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
    with pytest.raises(ValueError, match="requires history"):
        HarnessPolicy(
            history=False,
            prefix_cache=False,
            history_archive_threshold_tokens=32_000,
        )


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
        (
            {"codex_auth_helper_distribution_sha256": "ABC"},
            "lowercase SHA-256",
        ),
        (
            {"codex_auth_helper_distribution_sha256": "0" * 64},
            "requires a codex/ model",
        ),
        ({"kedi_otel_distribution_sha256": "ABC"}, "lowercase SHA-256"),
        ({"tree_sitter_kedi_revision": "ABC"}, "lowercase Git revision"),
        (
            {"tree_sitter_kedi_distribution_sha256": "ABC"},
            "lowercase SHA-256",
        ),
        (
            {"tree_sitter_kedi_distribution_sha256": "0" * 64},
            "requires a pinned",
        ),
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
    assert set(argv) == {
        "process_id",
        "exit_code",
        "duration_seconds",
        "stdout",
        "stderr",
    }
    recorded = terminal.summary()["records"][0]
    assert "print(" in recorded["command"]
    assert "hello" in recorded["command"]
    assert recorded["status"] == "finished"
    assert recorded["stdout_path"].endswith("stdout.bin")

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

    complete_output = terminal.run_command([sys.executable, "-c", "print('b' * 49)"])
    assert complete_output["stdout"] == "b" * 49 + "\n"
    assert "stdout_truncated" not in complete_output

    long_output = terminal.run_command([sys.executable, "-c", "print('a' * 100)"])
    assert long_output["stdout_truncated"] is True
    assert long_output["output_excerpt"] == "head_tail"
    continuation = long_output["output_continuations"]["stdout"]
    assert continuation == {
        "tool": "read_process_output",
        "arguments": {
            "process_id": long_output["process_id"],
            "stream": "stdout",
            "offset_bytes": 0,
            "max_bytes": 65_536,
        },
    }
    assert "omitted" in long_output["stdout"]
    full_page = terminal.read_process_output(**continuation["arguments"])
    assert full_page["complete"] is True
    assert full_page["content"].strip() == "a" * 100
    stream = terminal.stream_process_output(long_output["process_id"], "stdout")
    assert "".join(stream.iter_chunks()).strip() == "a" * 100


def test_terminal_waits_for_background_process_without_stopping_on_timeout(
    terminal: TerminalSession,
) -> None:
    process = terminal.start_process(
        [sys.executable, "-c", "import sys; sys.stdin.readline(); print('done')"]
    )
    process_id = cast(str, process["process_id"])

    running = terminal.wait_process(process_id, timeout_seconds=0.02)
    assert running["wait_completed"] is False
    assert running["status"] == "running"
    assert "stdout" not in running

    terminal.write_process_input(process_id, "continue", append_newline=True)
    finished = terminal.wait_process(process_id, timeout_seconds=5)
    assert finished["wait_completed"] is True
    assert finished["status"] == "finished"
    assert finished["stdout"] == "done\n"


def test_terminal_can_use_a_broader_explicit_cwd_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    terminal = TerminalSession(
        root=workspace,
        cwd_root=tmp_path,
        records_dir=tmp_path / "records",
    )
    try:
        result = terminal.run_command(
            [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=str(external)
        )
        assert result["stdout"].strip() == str(external)
        with pytest.raises(PermissionError, match="allowed cwd root"):
            terminal.run_command([sys.executable, "-c", "pass"], cwd=str(tmp_path.parent))
    finally:
        terminal.close()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("background", [False, True])
def test_terminal_artifact_preview_preserves_full_log_route(
    tmp_path: Path, stream: str, background: bool
) -> None:
    from kedi.artifacts import ArtifactHandle, ArtifactManager, ArtifactPolicy

    terminal = TerminalSession(root=tmp_path, records_dir=tmp_path / "records")
    manager = ArtifactManager(session_id=f"terminal-route-{stream}")
    try:
        payload = "H" * 40_000 + "middle-evidence" + "T" * 60_000
        fd = 1 if stream == "stdout" else 2
        argv = [
            sys.executable,
            "-c",
            f"import os; os.write({fd}, b'H'*40000+b'middle-evidence'+b'T'*60000)",
        ]
        if background:
            started = terminal.start_process(argv)
            terminal.wait_process(started["process_id"], timeout_seconds=3)
            result = terminal.process_status(started["process_id"])
        else:
            result = terminal.run_command(argv)
        stored = manager.artifactize(
            result,
            policy=ArtifactPolicy.patch(
                {"enabled": True, "threshold_bytes": 16_384, "preview_chars": 1200}
            ),
        )
        assert isinstance(stored, ArtifactHandle)
        continuation = result["output_continuations"][stream]
        assert "read_process_output" in stored.ref.preview
        visible_continuations, _ = json.JSONDecoder().raw_decode(
            stored.ref.preview.split('"output_continuations":', 1)[1].lstrip()
        )
        assert visible_continuations[stream] == continuation
        assert result["process_id"] in stored.ref.preview
        assert "head_tail" in stored.ref.preview
        assert "middle-evidence" not in result[stream]
        arguments = dict(continuation["arguments"])
        pages: list[str] = []
        while True:
            page = terminal.read_process_output(**arguments)
            pages.append(page["content"])
            if page["complete"]:
                break
            arguments["offset_bytes"] = page["next_offset_bytes"]
        assert "".join(pages) == payload
    finally:
        manager.close()
        terminal.close()


def test_waiting_again_cannot_revalidate_old_verification(terminal: TerminalSession) -> None:
    verifier = terminal.root / "pytest"
    verifier.write_text("#!/bin/sh\nexit 0\n")
    verifier.chmod(0o700)
    process = terminal.start_process([str(verifier)])
    finished = terminal.wait_process(process["process_id"], timeout_seconds=10)
    assert finished["wait_completed"] is True
    assert finished["exit_code"] == 0
    assert terminal.verification_status()["state"] == "passed"
    terminal.invalidate_verification("subsequent edit")
    terminal.wait_process(process["process_id"], timeout_seconds=2)
    assert terminal.verification_status()["state"] == "stale"


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


def test_terminal_records_bounded_redacted_tool_lifecycle_without_breaking_hooks(
    terminal: TerminalSession,
) -> None:
    common: dict[str, Any] = {
        "run_id": "run",
        "adapter_shortname": "pydantic",
        "tool_call_id": "patch-call",
        "tool_name": "apply_patch",
        "origin": "kedi",
        "arguments": {
            "operation": "update",
            "content": "x" * 5_000,
            "api_key": "fixture-secret-value",
        },
        "metadata": {"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
    }
    terminal.hook_handler(PreToolUseRequest(sequence=1, **common))
    terminal.hook_handler(PostToolUseEvent(sequence=2, result={"updated": True}, **common))
    terminal.hook_handler(
        PostToolUseFailureEvent(
            sequence=3,
            error_type="ValueError",
            error_message="failed with sk-this-must-be-redacted",
            duration_seconds=0.25,
            interrupted=False,
            **common,
        )
    )

    records = [
        json.loads(line)
        for line in (terminal.records_dir / "tool-events.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "pre_tool_use",
        "post_tool_use",
        "post_tool_use_failure",
    ]
    assert sum(record["event"] == "pre_tool_use" for record in records) == 1
    assert sum(record["event"] == "post_tool_use" for record in records) == 1
    assert records[0]["arguments"]["api_key"] == "[REDACTED]"
    assert records[0]["arguments"]["content"]["truncated"] is True
    assert "fixture-secret-value" not in json.dumps(records)
    assert "abcdefghijklmnopqrstuvwxyz" not in json.dumps(records)
    assert "sk-this-must" not in json.dumps(records)
    assert terminal.summary()["tool_events"] == 3
    assert terminal.summary()["tool_failures"] == 1

    terminal._tool_events_path = terminal.root
    terminal.hook_handler(PreToolUseRequest(sequence=4, **common))
    assert terminal.summary()["tool_audit_errors"] == 1

    class BrokenRepresentation:
        def __repr__(self) -> str:
            raise RuntimeError("audit serialization failed")

    terminal.hook_handler(PostToolUseEvent(sequence=5, result=BrokenRepresentation(), **common))
    assert terminal.summary()["tool_audit_errors"] == 2


def test_terminal_tool_audit_serializes_supported_values_and_flushes_fail_open(
    terminal: TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert terminal_audit_module._audit_value(None) is None
    assert terminal_audit_module._audit_value(b"short") == {
        "kind": "bytes",
        "size_bytes": 5,
        "sha256": "f9b0078b5df596d2ea19010c001bbd009e651de2c57e8fb7e355f31eb9d3f739",
        "preview": "short",
        "truncated": False,
    }
    long_bytes = b"x" * 5_000
    assert terminal_audit_module._audit_value(long_bytes)["truncated"] is True
    assert terminal_audit_module._audit_value(Path("fixture.txt")) == "fixture.txt"
    assert terminal_audit_module._audit_value({"nested": {"value": 1}}, depth=4) == {
        "kind": "depth_limit",
        "type": "dict",
    }

    mapping = {f"item-{index}": index for index in range(33)}
    mapped = terminal_audit_module._audit_value(mapping)
    assert mapped["__audit_truncated_items__"] == 1
    assert "item-32" not in mapped
    long_key = "prefix-" + "x" * 5_000 + "-suffix"
    audited_long_key = next(iter(terminal_audit_module._audit_value({long_key: 1})))
    assert len(audited_long_key) == 256
    assert audited_long_key.startswith("prefix-")
    assert audited_long_key.endswith("-suffix")
    assert "sha256:" in audited_long_key
    assert terminal_audit_module._audit_value({"sk-this-must-be-redacted": 1}) == {"[REDACTED]": 1}
    assert terminal_audit_module._audit_value(1.5) == 1.5
    assert terminal_audit_module._audit_value(float("nan")) == {
        "kind": "float",
        "value": "nan",
    }
    assert terminal_audit_module._audit_value(float("inf")) == {
        "kind": "float",
        "value": "inf",
    }
    assert terminal_audit_module._audit_value((1, 2)) == [1, 2]
    sequence = terminal_audit_module._audit_value(tuple(range(33)))
    assert sequence == {
        "kind": "sequence",
        "total_items": 33,
        "items": list(range(32)),
        "truncated": True,
    }
    assert terminal_audit_module._audit_value(SimpleNamespace(value=1)) == "namespace(value=1)"

    terminal.hook_handler(
        UserPromptSubmitRequest(
            run_id="run",
            sequence=1,
            adapter_shortname="pydantic",
            content="ignored",
        )
    )
    assert terminal.summary()["tool_events"] == 0

    terminal._tool_events_path.write_text("{}\n")

    def fail_fsync(_: int) -> None:
        raise OSError("fixture fsync failure")

    monkeypatch.setattr(terminal_processes_module.os, "fsync", fail_fsync)
    terminal._flush_tool_events()
    assert terminal.summary()["tool_audit_errors"] == 1


def test_terminal_retains_only_explicit_execution_processes(tmp_path: Path) -> None:
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        preview_bytes=64,
        output_limit_bytes=1_000_000,
    )
    retained = terminal.start_process(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; print('ready', flush=True); "
                "line=sys.stdin.readline(); print('unexpected-eof', flush=True); time.sleep(30)"
            ),
        ]
    )
    ordinary = terminal.start_process([sys.executable, "-c", "import time; time.sleep(30)"])
    retained_id = cast(str, retained["process_id"])
    ordinary_id = cast(str, ordinary["process_id"])
    try:
        deadline = time.monotonic() + 2
        while "ready" not in terminal.process_status(retained_id)["stdout"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        promoted = terminal.retain_process(retained_id)
        assert promoted["lifetime"] == "execution"
        assert promoted["retained"] is True

        terminal.finish_session()
        terminal.finish_session()
        time.sleep(0.1)
        assert terminal.process_status(retained_id)["status"] == "running"
        assert "unexpected-eof" not in terminal.process_status(retained_id)["stdout"]
        assert terminal.process_status(ordinary_id)["status"] == "finished"
        assert terminal.summary()["retained_processes"] == 1
        terminal._record_retained_process(terminal._process(retained_id))
    finally:
        terminal.close()
    assert terminal.process_status(retained_id)["status"] == "finished"


def test_terminal_retained_process_has_a_bounded_lifetime(tmp_path: Path) -> None:
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        retained_process_timeout_seconds=5,
    )
    process = terminal.start_process([sys.executable, "-c", "import time; time.sleep(30)"])
    process_id = cast(str, process["process_id"])
    try:
        terminal.retain_process(process_id)
        terminal.finish_session()
        managed = terminal._process(process_id)
        assert managed.retention_deadline_monotonic == managed.started_monotonic + 5
        deadline = time.monotonic() + 10
        while terminal.process_status(process_id)["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert terminal.process_status(process_id)["timed_out"] is True
    finally:
        terminal.close()


def test_retained_service_survives_terminal_owner_process_exit(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.txt"
    owner_source = "\n".join(
        (
            "import json, sys, time",
            "from pathlib import Path",
            "from kedi.integrations.harbor.terminal import TerminalSession",
            "root = Path(sys.argv[1])",
            "heartbeat = root / 'heartbeat.txt'",
            "terminal = TerminalSession(root=root, records_dir=root / 'records')",
            'worker = "from pathlib import Path; import sys,time; p=Path(sys.argv[1]); i=0; " '
            "+ \"exec('while True:\\\\n p.write_text(str(i)); i += 1; time.sleep(0.02)')\"",
            "process = terminal.start_process([sys.executable, '-c', worker, str(heartbeat)])",
            "deadline = time.monotonic() + 5",
            "while not heartbeat.exists():",
            "    if time.monotonic() >= deadline: raise TimeoutError('worker did not start')",
            "    time.sleep(0.02)",
            "terminal.retain_process(process['process_id'])",
            "terminal.finish_session()",
            "print(json.dumps(terminal.process_status(process['process_id'])), flush=True)",
        )
    )
    owner = subprocess.run(
        [sys.executable, "-c", owner_source, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    process = json.loads(owner.stdout)

    def heartbeat_value(deadline: float) -> int:
        while time.monotonic() < deadline:
            try:
                return int(heartbeat.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(0.01)
        raise TimeoutError("heartbeat was not readable")

    try:
        deadline = time.monotonic() + 2
        before = heartbeat_value(deadline)
        after = before
        while after <= before:
            assert time.monotonic() < deadline
            time.sleep(0.03)
            after = heartbeat_value(deadline)
        assert process["handed_off"] is True
        assert process["lifetime"] == "execution"
    finally:
        for process_group in (process["pid"], process["supervisor_pid"]):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                pass


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_terminal_reaps_pipe_owners_after_leader_exits(
    terminal: TerminalSession,
    tmp_path: Path,
    background: bool,
    ignore_term: bool,
) -> None:
    child_source = (
        "import os, signal, time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "Path('heartbeat').touch(); Path('ready').touch(); print('child-ready', flush=True); "
        "exec('for _ in range(2250):\\n "
        'with Path("heartbeat").open("ab", buffering=0) as h: h.write(b".")\\n '
        "time.sleep(0.02)')"
    )
    parent_source = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
        "Path('group').write_text(str(os.getpgrp())); "
        "exec('while not Path(\"ready\").exists(): time.sleep(0.01)'); "
        "time.sleep(0.15); print('parent-done', flush=True); sys.exit(7)"
    )
    started = time.monotonic()
    try:
        if background:
            process = terminal.start_process([sys.executable, "-c", parent_source])
            process_id = cast(str, process["process_id"])
            while terminal.process_status(process_id)["status"] == "running":
                assert time.monotonic() - started < 4, "orphaned pipes delayed supervisor exit"
                time.sleep(0.02)
            result = terminal.process_status(process_id)
        else:
            result = terminal.run_command([sys.executable, "-c", parent_source])
            process_id = cast(str, result["process_id"])
        assert time.monotonic() - started < 4, "orphaned pipes delayed foreground completion"
        assert result["exit_code"] == 7
        output = terminal.read_process_output(process_id)
        assert "child-ready" in output["content"]
        assert "parent-done" in output["content"]
        heartbeat = tmp_path / "heartbeat"
        size = heartbeat.stat().st_size
        time.sleep(0.1)
        assert heartbeat.stat().st_size == size, "descendant survived its process-group owner"
    finally:
        group = tmp_path / "group"
        if group.exists():
            try:
                os.killpg(int(group.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_terminal_rejects_invalid_process_retention(terminal: TerminalSession) -> None:
    foreground = terminal.run_command([sys.executable, "-c", "print('done')"])
    with pytest.raises(ValueError, match="background"):
        terminal.retain_process(cast(str, foreground["process_id"]))

    finished = terminal.start_process([sys.executable, "-c", "pass"])
    process_id = cast(str, finished["process_id"])
    deadline = time.monotonic() + 2
    while terminal.process_status(process_id)["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="no longer running"):
        terminal.retain_process(process_id)


def test_terminal_retained_service_survives_agent_deadline(tmp_path: Path) -> None:
    # Give the real supervisor enough time to start even on a loaded CI host. Once it is
    # retained, advance the session's agent deadline directly; retained lifetime is owned by
    # the execution deadline and must not depend on wall-clock scheduling variance in this test.
    agent_deadline = time.monotonic() + 10
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "retained",
        agent_deadline_monotonic=agent_deadline,
        retained_process_timeout_seconds=20,
    )
    try:
        process = terminal.start_process([sys.executable, "-c", "import time; time.sleep(30)"])
        process_id = process["process_id"]
        terminal.retain_process(process_id)
        terminal.deadline_monotonic = time.monotonic() - 1
        terminal.finish_session()
        assert terminal.process_status(process_id)["status"] == "running"
        managed = terminal._process(process_id)
        assert managed.retention_deadline_monotonic > agent_deadline
        with pytest.raises(TimeoutError, match="deadline"):
            terminal._bounded_timeout(None)
    finally:
        terminal.close()


def test_terminal_supervisor_start_and_state_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        TerminalSession(
            root=tmp_path,
            records_dir=tmp_path / "invalid-records",
            retained_process_timeout_seconds=0,
        )

    deadline = time.monotonic() + 10
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "deadline-records",
        deadline_monotonic=deadline,
        retained_process_timeout_seconds=30,
    )
    process = terminal.start_process([sys.executable, "-c", "import time; time.sleep(30)"])
    managed = terminal._process(cast(str, process["process_id"]))
    assert managed.retention_deadline_monotonic == deadline
    terminal.close()

    exited = SimpleNamespace(poll=lambda: 1)
    assert terminal._await_supervisor_start(cast(Any, exited), tmp_path / "missing.json") == {}
    expired = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "expired-records",
        deadline_monotonic=time.monotonic() - 1,
    )
    waiting = SimpleNamespace(poll=lambda: None)
    assert expired._await_supervisor_start(cast(Any, waiting), tmp_path / "missing.json") == {}
    expired.close()

    class BrokenInput:
        def write(self, _: bytes) -> None:
            raise BrokenPipeError

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeSupervisorProcess:
        pid = 987_654

        def __init__(self, *, broken_input: bool, timeout_once: bool) -> None:
            self.stdin = (
                BrokenInput()
                if broken_input
                else SimpleNamespace(
                    write=lambda _: None,
                    flush=lambda: None,
                    close=lambda: None,
                )
            )
            self.timeout_once = timeout_once
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            self.wait_calls += 1
            if self.timeout_once and self.wait_calls == 1:
                raise subprocess.TimeoutExpired("supervisor", timeout)
            return 126

    failures = iter(
        (
            {"phase": "supervisor_failure", "start_error_type": "RuntimeError"},
            {"phase": "running"},
        )
    )
    fake_processes = iter(
        (
            FakeSupervisorProcess(broken_input=True, timeout_once=True),
            FakeSupervisorProcess(broken_input=False, timeout_once=True),
        )
    )
    signals: list[tuple[int, int]] = []
    mocked = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "mocked-records",
    )
    monkeypatch.setattr(
        terminal_processes_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(fake_processes),
    )
    monkeypatch.setattr(mocked, "_await_supervisor_start", lambda *_args: next(failures))
    monkeypatch.setattr(
        mocked,
        "_signal_pid_group",
        lambda process_group, signum: signals.append((process_group, signum)),
    )
    first = mocked.start_process(["first"])
    second = mocked.start_process(["second"])
    assert first["status"] == "failed_to_start"
    assert first["start_error_type"] == "OSError"
    assert second["status"] == "failed_to_start"
    assert signals == [
        (987_654, signal.SIGKILL),
        (987_654, signal.SIGTERM),
        (987_654, signal.SIGKILL),
    ]
    mocked.close()


def test_terminal_supervisor_helpers_and_process_input_cleanup(tmp_path: Path) -> None:
    assert isinstance(
        terminal_processes_module._supervisor_start_error(
            {"start_error_type": "FileNotFoundError", "start_error_message": "missing"}
        ),
        FileNotFoundError,
    )
    assert isinstance(
        terminal_processes_module._supervisor_start_error(
            {"start_error_type": "PermissionError", "start_error_message": "denied"}
        ),
        PermissionError,
    )
    assert isinstance(terminal_processes_module._supervisor_start_error({}), OSError)
    assert terminal_output_module._preview_file(tmp_path / "missing.bin", 8) == (
        "",
        False,
        False,
        0,
    )
    invalid_utf8 = tmp_path / "invalid-utf8.bin"
    invalid_utf8.write_bytes(b"\xff\xfe")
    assert terminal_output_module._preview_file(invalid_utf8, 8) == (
        "\ufffd\ufffd",
        False,
        True,
        2,
    )
    complete_preview = tmp_path / "complete-preview.bin"
    complete_preview.write_bytes(b"abcdefghijkl")
    assert terminal_output_module._preview_file(complete_preview, 8) == (
        "abcdefghijkl",
        False,
        False,
        12,
    )
    truncated_preview = tmp_path / "truncated-preview.bin"
    truncated_preview.write_bytes(b"abcdefghijklmnopqrst")
    rendered, truncated, invalid, total = terminal_output_module._preview_file(truncated_preview, 8)
    assert (truncated, invalid, total) == (True, False, 20)
    assert "4 bytes omitted" in rendered

    class BrokenClose:
        def close(self) -> None:
            raise BrokenPipeError

    terminal_processes_module._close_process_input(cast(Any, SimpleNamespace(stdin=None)))
    terminal_processes_module._close_process_input(cast(Any, SimpleNamespace(stdin=BrokenClose())))

    terminal = TerminalSession(root=tmp_path, records_dir=tmp_path / "state-records")
    result = terminal.run_command([sys.executable, "-c", "pass"])
    managed = terminal._process(cast(str, result["process_id"]))
    state_path = tmp_path / "partial-state.json"
    managed.supervisor_state_path = state_path
    managed.target_pid = 10
    managed.target_pgid = 20
    atomic_write_json(state_path, {"target_pid": 30, "target_pgid": "invalid"})
    terminal._refresh_supervisor_state(managed)
    assert (managed.target_pid, managed.target_pgid) == (30, 20)
    atomic_write_json(state_path, {"target_pid": "invalid", "target_pgid": 40})
    terminal._refresh_supervisor_state(managed)
    assert (managed.target_pid, managed.target_pgid) == (30, 40)
    terminal.close()


def _supervisor_config(
    root: Path,
    command: list[str],
    *,
    retained: bool,
    output_limit_bytes: int = 1_000_000,
    lifetime_seconds: float = 10.0,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        name: root / name
        for name in ("state.json", "control.json", "stdout.bin", "stderr.bin", "combined.bin")
    }
    for name in ("stdout.bin", "stderr.bin", "combined.bin"):
        paths[name].touch()
    atomic_write_json(paths["control.json"], {"retained": retained})
    return {
        "command": command,
        "cwd": str(root),
        "state_path": str(paths["state.json"]),
        "control_path": str(paths["control.json"]),
        "stdout_path": str(paths["stdout.bin"]),
        "stderr_path": str(paths["stderr.bin"]),
        "combined_path": str(paths["combined.bin"]),
        "output_limit_bytes": output_limit_bytes,
        "deadline_monotonic": time.monotonic() + lifetime_seconds,
    }


def test_process_supervisor_success_failure_input_and_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    monkeypatch.setattr(harbor_process_supervisor.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"hello\n")),
    )
    success_root = tmp_path / "success"
    success_config = _supervisor_config(
        success_root,
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdin.readline().strip()); print('err', file=sys.stderr)",
        ],
        retained=True,
    )
    assert harbor_process_supervisor._ProcessSupervisor(success_config).run() == 0
    success_state = json.loads((success_root / "state.json").read_text(encoding="utf-8"))
    assert success_state["phase"] == "finished"
    assert (success_root / "stdout.bin").read_text(encoding="utf-8") == "hello\n"
    assert (success_root / "stderr.bin").read_text(encoding="utf-8") == "err\n"
    assert "hello" in (success_root / "combined.bin").read_text(encoding="utf-8")

    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    failure_root = tmp_path / "failure"
    failure_config = _supervisor_config(
        failure_root,
        [str(tmp_path / "missing-executable")],
        retained=False,
    )
    assert harbor_process_supervisor._ProcessSupervisor(failure_config).run() == 127
    failure_state = json.loads((failure_root / "state.json").read_text(encoding="utf-8"))
    assert failure_state["phase"] == "failed_to_start"
    assert failure_state["start_error_type"] == "FileNotFoundError"

    limited_root = tmp_path / "limited"
    limited_config = _supervisor_config(
        limited_root,
        [
            sys.executable,
            "-c",
            (
                "import signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('x' * 10000, flush=True); time.sleep(30)"
            ),
        ],
        retained=True,
        output_limit_bytes=32,
    )
    assert harbor_process_supervisor._ProcessSupervisor(limited_config).run() != 0
    limited_state = json.loads((limited_root / "state.json").read_text(encoding="utf-8"))
    assert limited_state["output_limited"] is True
    assert (limited_root / "stdout.bin").stat().st_size == 32

    deadline_root = tmp_path / "deadline"
    deadline_config = _supervisor_config(
        deadline_root,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        retained=True,
        lifetime_seconds=0.05,
    )
    assert harbor_process_supervisor._ProcessSupervisor(deadline_config).run() != 0
    deadline_state = json.loads((deadline_root / "state.json").read_text(encoding="utf-8"))
    assert deadline_state["timed_out"] is True


def test_process_supervisor_validation_signals_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    with pytest.raises(ValueError, match="command"):
        harbor_process_supervisor._ProcessSupervisor({"command": []})

    config = _supervisor_config(
        tmp_path / "signals",
        [sys.executable, "-c", "pass"],
        retained=False,
    )
    supervisor = harbor_process_supervisor._ProcessSupervisor(config)
    assert supervisor._is_retained() is False
    Path(config["control_path"]).write_text("not-json", encoding="utf-8")
    assert supervisor._is_retained() is False
    Path(config["control_path"]).write_text("[]", encoding="utf-8")
    assert supervisor._is_retained() is False

    signals: list[tuple[int, int]] = []
    child = SimpleNamespace(pid=123, poll=lambda: None)
    supervisor._child = cast(Any, child)
    monkeypatch.setattr(
        harbor_process_supervisor.os,
        "killpg",
        lambda process_group, signum: signals.append((process_group, signum)),
    )
    supervisor._handle_signal(signal.SIGTERM, None)
    assert signals == [(123, signal.SIGTERM)]
    assert supervisor._requested_signal == signal.SIGTERM
    child.poll = lambda: 0
    supervisor._signal_child(signal.SIGKILL)
    supervisor._child = None
    supervisor._signal_child(signal.SIGKILL)
    supervisor._forward_input()
    supervisor._close_child_input()

    class BrokenInput:
        def write(self, _: bytes) -> None:
            raise BrokenPipeError

        def flush(self) -> None:
            pass

        def close(self) -> None:
            raise BrokenPipeError

    child = SimpleNamespace(pid=123, poll=lambda: None, stdin=BrokenInput())
    supervisor._child = cast(Any, child)
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"input")),
    )
    supervisor._forward_input()
    supervisor._close_child_input()
    child.stdin = None
    supervisor._forward_input()
    supervisor._close_child_input()

    child.stdin = io.BytesIO()
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    atomic_write_json(Path(config["control_path"]), {"retained": False})
    supervisor._termination_deadline = None
    supervisor._forward_input()
    assert supervisor._termination_deadline is not None
    supervisor._termination_deadline = None
    supervisor._request_termination(signal.SIGKILL)
    assert supervisor._termination_deadline is None
    monkeypatch.setattr(
        harbor_process_supervisor.os,
        "killpg",
        lambda *_: (_ for _ in ()).throw(PermissionError("gone")),
    )
    supervisor._signal_child(signal.SIGTERM)

    supervisor.output_limit_bytes = 0
    limited_source = io.BytesIO(b"discarded")
    reader = supervisor._reader("stdout", limited_source, Path(config["stdout_path"]))
    reader.start()
    reader.join(timeout=2)
    assert supervisor._output_limited is True

    Path(config["state_path"]).write_text("[]", encoding="utf-8")
    supervisor._write_state(phase="test")
    assert json.loads(Path(config["state_path"]).read_text(encoding="utf-8"))["phase"] == "test"

    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    assert harbor_process_supervisor.main() == 126
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"invalid\n")),
    )
    assert harbor_process_supervisor.main() == 126
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"[]\n")),
    )
    assert harbor_process_supervisor.main() == 126

    class SuccessfulSupervisor:
        def __init__(self, _: dict[str, Any]) -> None:
            pass

        def run(self) -> int:
            return 7

    monkeypatch.setattr(harbor_process_supervisor, "_ProcessSupervisor", SuccessfulSupervisor)
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"{}\n")),
    )
    assert harbor_process_supervisor.main() == 7

    failure_state = tmp_path / "main-failure.json"

    class FailingSupervisor:
        def __init__(self, _: dict[str, Any]) -> None:
            pass

        def run(self) -> int:
            raise RuntimeError("failed")

    monkeypatch.setattr(harbor_process_supervisor, "_ProcessSupervisor", FailingSupervisor)
    payload = json.dumps({"state_path": str(failure_state)}).encode() + b"\n"
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(payload)),
    )
    assert harbor_process_supervisor.main() == 126
    assert json.loads(failure_state.read_text(encoding="utf-8"))["phase"] == "supervisor_failure"
    monkeypatch.setattr(
        harbor_process_supervisor.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"{}\n")),
    )
    assert harbor_process_supervisor.main() == 126


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
    with pytest.raises(PermissionError, match="escapes"):
        terminal.run_command(["true"], cwd=str(tmp_path.parent))
    tool_spec = python_callable_tool_spec("run_command", terminal.run_command)
    assert tool_spec.metadata is not None
    assert tool_spec.metadata["kedi_tool_kind"] == "terminal"
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
        assert result["output_limited"] is True
        assert result["stdout_bytes"] == 1_024
        assert result["exit_code"] != 0
    finally:
        terminal.close()


def test_background_supervisor_enforces_output_limit_after_handoff(tmp_path: Path) -> None:
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "records",
        output_limit_bytes=1_024,
        preview_bytes=128,
    )
    process = terminal.start_process(
        [
            sys.executable,
            "-c",
            (
                "import signal,sys,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "sys.stdout.write('x' * 100_000); sys.stdout.flush(); time.sleep(30)"
            ),
        ]
    )
    process_id = cast(str, process["process_id"])
    try:
        terminal.retain_process(process_id)
        terminal.finish_session()
        deadline = time.monotonic() + 5
        while terminal.process_status(process_id)["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        result = terminal.process_status(process_id)
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
    assert missing["start_error_type"] == "FileNotFoundError"
    assert "could not start command" in missing["stderr"]
    assert terminal.summary()["commands"] == 1
    assert terminal.summary()["records"][0]["pid"] is None

    missing_verification = terminal.run_command([str(tmp_path / "pytest"), "-q"])
    assert missing_verification["exit_code"] == 127
    assert terminal.verification_status()["state"] == "failed"

    missing_background = terminal.start_process([str(tmp_path / "missing-background")])
    assert missing_background["status"] == "failed_to_start"
    assert terminal.list_processes() == []

    with monkeypatch.context() as context:
        context.setattr(
            terminal_processes_module.subprocess,
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
    preview = terminal_output_module._BytePreview(limit=4)
    preview.append(b"abcdef")
    assert preview.render() == ("abcdef", False, False)
    preview.append(b"ghijkl")
    rendered, truncated, invalid_utf8 = preview.render()
    assert truncated is True
    assert invalid_utf8 is False
    assert "omitted" in rendered

    result = terminal.run_command([sys.executable, "-c", "print('done')"])
    process_id = cast(str, result["process_id"])
    with pytest.raises(RuntimeError, match="not accepting input"):
        terminal.write_process_input(process_id, "late")
    with pytest.raises(ValueError, match="stream must be"):
        terminal.read_process_output(process_id, stream=cast(Any, "invalid"))

    recorded = next(
        record for record in terminal.summary()["records"] if record["process_id"] == process_id
    )
    combined_path = Path(cast(str, recorded["combined_path"]))
    combined_path.unlink()
    assert terminal.read_process_output(process_id)["content"] == ""

    with pytest.raises(ValueError, match="positive"):
        terminal._bounded_timeout(0)
    with pytest.raises(ValueError, match="positive"):
        terminal._bounded_timeout(float("nan"))
    assert terminal._bounded_timeout(600) == terminal.command_timeout_seconds
    with pytest.raises(ValueError, match="retained_process_timeout_seconds"):
        TerminalSession(
            root=terminal.root,
            records_dir=terminal.records_dir / "invalid-retention",
            retained_process_timeout_seconds=float("inf"),
        )
    with pytest.raises(ValueError, match="command_timeout_seconds"):
        TerminalSession(
            root=terminal.root,
            records_dir=terminal.records_dir / "invalid-command-timeout",
            command_timeout_seconds=float("nan"),
        )
    with pytest.raises(ValueError, match="finalization_reserve_seconds"):
        TerminalSession(
            root=terminal.root,
            records_dir=terminal.records_dir / "invalid-finalization-reserve",
            finalization_reserve_seconds=-1,
        )
    terminal.deadline_monotonic = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="deadline"):
        terminal._bounded_timeout(None)
    terminal.deadline_monotonic = time.monotonic() + 1
    assert 0 < terminal._bounded_timeout(10) <= 1

    reserved = TerminalSession(
        root=terminal.root,
        records_dir=terminal.records_dir / "reserved",
        deadline_monotonic=time.monotonic() + 1,
        finalization_reserve_seconds=2,
    )
    with pytest.raises(TimeoutError, match="finalization reserve"):
        reserved._bounded_timeout(None)
    reserved.close()

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/credentials.json")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("VISIBLE_FIXTURE", "yes")
    child_env = terminal_processes_module._sanitized_subprocess_env()
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
    assert terminal_processes_module._is_verification_command(
        "python3.14 -m pytest -q", shell=False
    )
    assert not terminal_processes_module._is_verification_command("'", shell=False)
    assert not terminal_processes_module._is_verification_command("", shell=False)
    assert not terminal_processes_module._is_read_only_command("'", shell=False)
    assert not terminal_processes_module._is_read_only_command("", shell=False)
    assert not terminal_processes_module._is_read_only_command("echo changed", shell=False)
    assert terminal_processes_module._is_read_only_command("cat file.txt", shell=False)

    fake = SimpleNamespace(process=SimpleNamespace(pid=999999))
    monkeypatch.setattr(
        terminal_processes_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )
    TerminalSession._signal_process_group(cast(Any, fake), signal.SIGTERM)

    empty = terminal_output_module._empty_binary_file()
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
    retain = ApprovalRequest.create(
        tool_name="retain_process",
        arguments={"process_id": "process_1"},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    assert runner.benchmark_approval(retain).decision == "allow"
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
        codemode_enabled=False,
        compaction_mode="disabled",
        compaction_threshold=None,
    )
    assert "> history: disabled" in disabled
    assert "> codemode: disabled" in disabled
    parse_program(disabled, source_path="<harbor-agent-stateless>")
    enabled = runner.render_agent_source(
        codemode_enabled=True,
        compaction_mode="disabled",
        compaction_threshold=None,
    )
    assert "> codemode: enabled" in enabled
    assert "bounded tool invocation" not in enabled
    batching = runner.render_agent_source(
        batching_guidance_enabled=True,
        compaction_mode="disabled",
        compaction_threshold=None,
    )
    assert "bounded tool invocation" in batching
    parse_program(batching, source_path="<harbor-agent-batching>")
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
        codemode_enabled=True,
    )

    assert record.state == "completed"
    assert record.usage["requests"] == 1
    assert record.usage["tool_calls"] == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "completed"


@pytest.mark.parametrize("artifacts_enabled", [False, True])
def test_runner_uses_raw_output_without_losing_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts_enabled: bool,
) -> None:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    instruction = (
        "Read status.txt; [expected] and <placeholder> are literal text.\nReport its status."
    )
    summary = "Status is ready; no changes made."
    calls: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert prompts == [instruction]
        assert info.allow_text_output
        assert not info.output_tools
        tools = {tool.name for tool in info.function_tools}
        assert {"read_text_file", "apply_patch", "run_command", "retain_process"} <= tools
        assert ("read_artifact" in tools) is artifacts_enabled
        assert ("run_artifact_code" in tools) is artifacts_enabled
        instructions = info.instructions or ""
        assert "Verify related changes together" in instructions
        assert "Do not claim effects unsupported by tool output." in instructions
        calls.append(tools)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_text_file", {"file_path": "status.txt"}, tool_call_id="read-status"
                    )
                ]
            )
        results = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(results) == 1
        assert results[0].tool_call_id == "read-status"
        assert "ready" in str(results[0].content)
        return ModelResponse(parts=[TextPart(summary)])

    real_bind = runner.kedi.bind

    def bind_with_test_model(**kwargs: Any) -> Any:
        kwargs["adapter"] = PydanticAdapter(FunctionModel(respond))
        kwargs.pop("model", None)
        kwargs.pop("effort", None)
        return real_bind(**kwargs)

    monkeypatch.setattr(runner.kedi, "bind", bind_with_test_model)
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    (tmp_path / "status.txt").write_text("ready", encoding="utf-8")
    record = runner.run_trial(
        instruction=instruction,
        model="test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "logs" / "kedi-result.json",
        trial_id="raw-invoke-smoke",
        artifacts_enabled=artifacts_enabled,
    )

    assert record.state == "completed", record.error_message
    assert record.result_summary == summary
    assert record.usage["requests"] == 2
    assert record.usage["tool_calls"] == 1
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert record.terminal["tool_failures"] == 0


def test_runner_configures_logfire_only_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    fake_logfire = SimpleNamespace(
        configure=lambda **kwargs: calls.append(("configure", kwargs)),
    )
    fake_instrumentor = SimpleNamespace(instrument=lambda **kwargs: calls.append(("kedi", kwargs)))
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.kedi",
        SimpleNamespace(KediInstrumentor=lambda: fake_instrumentor),
    )
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    assert runner._configure_logfire() is False
    assert calls == []

    monkeypatch.setenv("KEDI_HARBOR_LOGFIRE", "1")
    assert runner._configure_logfire() is True
    assert calls == [
        ("configure", {"service_name": "kedi", "send_to_logfire": "if-token-present"}),
        (
            "kedi",
            {
                "runtime_detail": "off",
                "agent_enabled": True,
                "artifacts_enabled": True,
                "instrument_pydantic_ai": True,
                "instrument_httpx": False,
                "capture_content": False,
            },
        ),
    ]


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_runner_enables_artifact_history_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    import kedi.agent_adapter.adapters as adapter_module
    from kedi.agent_adapter.compaction import HistoryArchiveSettings

    constructed: list[dict[str, Any]] = []
    bound: list[dict[str, Any]] = []

    def fake_adapter(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return SimpleNamespace()

    def fake_bind(**kwargs: Any) -> Any:
        bound.append(kwargs)

        def decorate(_: Any) -> Any:
            return lambda _instruction: "done"

        return decorate

    monkeypatch.setattr(
        adapter_module,
        "PydanticAdapter" if adapter_name == "pydantic" else "LangChainAdapter",
        fake_adapter,
    )
    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    enabled = runner.run_trial(
        instruction="fix it",
        model="openai/test",
        adapter=adapter_name,
        workspace=tmp_path,
        result_path=tmp_path / "enabled.json",
        trial_id="archive-enabled",
        history_archive_threshold_tokens=2_000,
        history_archive_preserve_tokens=500,
    )
    disabled = runner.run_trial(
        instruction="fix it",
        model="openai/test",
        adapter=adapter_name,
        workspace=tmp_path,
        result_path=tmp_path / "disabled.json",
        trial_id="archive-disabled",
    )

    assert isinstance(constructed[0]["history_archive"], HistoryArchiveSettings)
    assert constructed[1]["history_archive"] is None
    assert constructed[0]["default_usage_limits"].request_limit == 512
    assert enabled.policy["history_archive_threshold_tokens"] == 2_000
    assert enabled.policy["history_archive_preserve_tokens"] == 500
    assert disabled.policy["history_archive_threshold_tokens"] is None
    assert len(bound) == 2
    assert all(call["skills"] is False for call in bound)


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
    assert record.policy["codemode"] is False
    assert record.policy["finalization_reserve_seconds"] == 0.0
    assert json.loads(result_path.read_text())["state"] == "completed"
    assert captured["skills"] is False
    assert set(captured["hooks"]) == {
        "pre_tool_use",
        "post_tool_use",
        "post_tool_use_failure",
    }
    assert record.policy["request_limit"] == 512
    assert record.policy["cumulative_input_tokens"] == 50_000_000
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


def test_runner_recovers_usage_when_timeout_interrupts_the_invocation_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "timeout" / "kedi-result.json"

    def interrupted_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                event = {
                    "event": "model_request_completed",
                    "response": {
                        "parts": [{"kind": "tool-call"}],
                        "usage": {
                            "input_tokens": 120,
                            "cache_read_tokens": 80,
                            "cache_write_tokens": 0,
                            "output_tokens": 12,
                            "total_tokens": 132,
                            "cost_usd": 0.004,
                        },
                    },
                }
                (result_path.parent / "model-requests.jsonl").write_text(
                    json.dumps(event) + "\n",
                    encoding="utf-8",
                )
                raise runner._TrialTimedOut("fixture timeout")  # noqa: SLF001

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", interrupted_bind)
    record = runner.run_trial(
        instruction="run until timeout",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=result_path,
        trial_id="timeout-usage",
        request_observation=True,
    )

    assert record.state == "timed_out"
    assert record.usage["requests"] == 1
    assert record.usage["tool_calls"] == 1
    assert record.usage["input_tokens"] == 120
    assert record.usage["cache_read_tokens"] == 80
    assert record.usage["output_tokens"] == 12
    assert record.usage["cost_usd"] == 0.004
    assert record.request_observation["usage_recovered"] is True


def test_runner_scopes_generous_configurable_model_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.request_budget import ModelRequestBudget, ModelRequestBudgetLimits

    captured: dict[str, Any] = {}

    def fake_bind(**kwargs: Any) -> Any:
        captured["adapter"] = kwargs["adapter"]

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                captured["limits"] = ModelRequestBudget().limits
                return "done"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    record = runner.run_trial(
        instruction="use the benchmark budget",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "budget-policy" / "result.json",
        trial_id="budget-policy",
        model_request_limit=777,
        model_cumulative_input_tokens=123_000_000,
    )

    assert record.state == "completed"
    assert captured["limits"].request_limit == 777
    assert captured["limits"].cumulative_input_tokens == 123_000_000
    assert captured["adapter"]._default_usage_limits.request_limit == 777
    assert ModelRequestBudget().limits == ModelRequestBudgetLimits()


def test_runner_treats_model_request_budget_exhaustion_as_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.request_budget import ModelRequestBudgetError
    from kedi.errors import KediExecutionError

    terminals: list[TerminalSession] = []

    def exhausted_bind(**kwargs: Any) -> Any:
        start_process = next(item for item in kwargs["tools"] if item.__name__ == "start_process")
        retain_process = next(item for item in kwargs["tools"] if item.__name__ == "retain_process")
        terminals.append(cast(TerminalSession, start_process.__self__))

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                process = start_process([sys.executable, "-c", "import time; time.sleep(30)"])
                retain_process(process["process_id"])
                raise KediExecutionError(
                    "request budget exhausted",
                    original=ModelRequestBudgetError("cumulative model input"),
                )

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", exhausted_bind)
    result_path = tmp_path / "budget" / "result.json"
    record = runner.run_trial(
        instruction="work until the host budget is exhausted",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=result_path,
        trial_id="budget-exhausted",
    )

    assert record.state == "budget_exhausted"
    assert record.failure_phase == "agent"
    assert record.error_type == "KediExecutionError"
    assert record.terminal["retained_processes"] == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == ("budget_exhausted")
    terminals[0].close()


def test_runner_hands_off_retained_processes_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminals: list[TerminalSession] = []

    def retaining_bind(**kwargs: Any) -> Any:
        start_process = next(item for item in kwargs["tools"] if item.__name__ == "start_process")
        retain_process = next(item for item in kwargs["tools"] if item.__name__ == "retain_process")
        process_status = next(item for item in kwargs["tools"] if item.__name__ == "process_status")
        terminals.append(cast(TerminalSession, start_process.__self__))

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                process = start_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('ready', flush=True); time.sleep(30)",
                    ]
                )
                process_id = cast(str, process["process_id"])
                deadline = time.monotonic() + 2
                while "ready" not in process_status(process_id)["stdout"]:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                retain_process(process_id)
                return "service ready"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", retaining_bind)
    completed = runner.run_trial(
        instruction="start service",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "retained" / "result.json",
        trial_id="retained",
    )
    assert completed.state == "completed"
    assert completed.terminal["retained_processes"] == 1
    assert completed.terminal["records"][0]["handed_off"] is True
    assert terminals[0].list_processes()[0]["status"] == "running"
    terminals[0].close()

    def failing_bind(**kwargs: Any) -> Any:
        start_process = next(item for item in kwargs["tools"] if item.__name__ == "start_process")
        retain_process = next(item for item in kwargs["tools"] if item.__name__ == "retain_process")
        terminals.append(cast(TerminalSession, start_process.__self__))

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                process = start_process([sys.executable, "-c", "import time; time.sleep(30)"])
                retain_process(process["process_id"])
                raise RuntimeError("agent failed")

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", failing_bind)
    failed = runner.run_trial(
        instruction="fail after starting service",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "failed-retained" / "result.json",
        trial_id="failed-retained",
    )
    assert failed.state == "agent_failure"
    assert failed.terminal["retained_processes"] == 0
    assert terminals[1].list_processes()[0]["status"] == "finished"


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
def test_runner_unwinds_active_turn_before_deadline_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: str
) -> None:
    from kedi.agent_adapter.execution import run_sync
    from kedi.engine import ThreadEngine

    events: list[str] = []

    def fake_bind(**kwargs: Any) -> Any:
        conversation = kwargs["conversation"]

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                async def work() -> str:
                    try:
                        await asyncio.sleep(10)
                        return "late"
                    finally:
                        events.append("async closed")

                def job() -> str:
                    with conversation.adapter_turn(adapter):
                        return run_sync(work())

                engine = ThreadEngine(max_workers=1)
                try:
                    return engine.schedule([], job).result()
                finally:
                    engine.shutdown()

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", fake_bind)
    record = runner.run_trial(
        instruction="wait",
        model="openai/test",
        adapter=adapter,
        workspace=tmp_path,
        result_path=tmp_path / "result.json",
        trial_id="deadline",
        trial_timeout_seconds=1,
    )
    assert events == ["async closed"]
    assert record.state == "timed_out"
    assert record.error_type == "RunDeadlineExceeded"
    assert not (tmp_path / "cleanup-errors.txt").exists()
    assert json.loads((tmp_path / "result.json").read_text())["state"] == "timed_out"


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
    assert timed_out.policy["finalization_reserve_seconds"] == pytest.approx(0.005)


@pytest.mark.parametrize("expired", [True, False])
def test_runner_absolute_deadline_includes_launch_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    invoked = False

    def slow_bind(**_: Any) -> Any:
        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                nonlocal invoked
                invoked = True
                time.sleep(5)
                return "too late"

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", slow_bind)
    deadline = time.time() + (-1 if expired else 0.5)
    started = time.monotonic()
    record = runner.run_trial(
        instruction="wait",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "deadline" / "result.json",
        trial_id="deadline",
        trial_timeout_seconds=600,
        trial_deadline_unix=deadline,
    )
    assert record.state == "timed_out"
    assert record.failure_phase == ("setup" if expired else "agent")
    assert invoked is not expired
    assert time.monotonic() - started < 3
    assert record.policy["trial_deadline_unix"] == deadline
    assert record.policy["trial_timeout_seconds"] == 600


def test_terminal_time_notice_is_once_only_and_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(terminal_session_module.time, "monotonic", lambda: clock[0])
    terminal = TerminalSession(
        root=tmp_path,
        records_dir=tmp_path / "logs",
        deadline_monotonic=1000,
        finalization_reserve_seconds=20,
    )
    assert "execution_budget" not in terminal.verification_status()
    clock[0] = 880
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(target=lambda: results.append(terminal.verification_status()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    notices = [result["execution_budget"] for result in results if "execution_budget" in result]
    assert len(notices) == 1
    assert notices[0]["remaining_seconds"] == 120
    assert notices[0]["finalization_reserve_seconds"] == 20
    assert all(result["state"] == "not_attempted" for result in results)
    assert "execution_budget" not in terminal.verification_status()
    unlimited = TerminalSession(root=tmp_path, records_dir=tmp_path / "unlimited")
    assert "execution_budget" not in unlimited.verification_status()


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

    def wrapped_timeout_bind(**_: Any) -> Any:
        from kedi.errors import KediExecutionError

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                raise KediExecutionError(
                    "wrapped timeout",
                    original=runner._TrialTimedOut("trial terminated by Harbor"),
                )

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", wrapped_timeout_bind)
    wrapped_timeout = runner.run_trial(
        instruction="timeout",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "wrapped-timeout" / "result.json",
        trial_id="wrapped-timeout",
    )
    assert wrapped_timeout.state == "timed_out"
    assert wrapped_timeout.failure_phase == "agent"
    assert wrapped_timeout.error_type == "_TrialTimedOut"
    assert wrapped_timeout.error_message == "trial terminated by Harbor"

    def wrapped_cancelled_bind(**_: Any) -> Any:
        from kedi.errors import KediExecutionError

        def decorate(_: Any) -> Any:
            def invoke(_: str) -> str:
                raise KediExecutionError(
                    "wrapped cancellation",
                    original=KeyboardInterrupt(),
                )

            return invoke

        return decorate

    monkeypatch.setattr(runner.kedi, "bind", wrapped_cancelled_bind)
    wrapped_cancelled = runner.run_trial(
        instruction="cancel",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "wrapped-cancelled" / "result.json",
        trial_id="wrapped-cancelled",
    )
    assert wrapped_cancelled.state == "cancelled"
    assert wrapped_cancelled.failure_phase == "agent"
    assert wrapped_cancelled.error_type == "KeyboardInterrupt"
    assert wrapped_cancelled.error_message == "trial cancelled"

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

    invalid_infinite = runner.run_trial(
        instruction="invalid",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "invalid-infinite-timeout" / "result.json",
        trial_id="invalid-infinite-timeout",
        trial_timeout_seconds=float("inf"),
    )
    assert invalid_infinite.state == "integration_failure"
    assert invalid_infinite.failure_phase == "setup"
    assert invalid_infinite.policy["trial_timeout_seconds"] == "inf"

    invalid_budget = runner.run_trial(
        instruction="invalid",
        model="openai/test",
        adapter="pydantic",
        workspace=tmp_path,
        result_path=tmp_path / "invalid-budget" / "result.json",
        trial_id="invalid-budget",
        model_request_limit=0,
    )
    assert invalid_budget.state == "integration_failure"
    assert invalid_budget.failure_phase == "setup"
    assert invalid_budget.error_type == "ValueError"


def test_runner_cleanup_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_terminal = runner.TerminalSession

    class CleanupFailureTerminal(original_terminal):
        def finish_session(self) -> None:
            super().finish_session()
            raise RuntimeError("cleanup failed")

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
    assert captured["model_request_limit"] == 512
    assert captured["model_cumulative_input_tokens"] == 50_000_000
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
    monkeypatch.setattr(
        runner,
        "run_trial",
        lambda **_: SimpleNamespace(state="budget_exhausted"),
    )
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
        policy=HarnessPolicy(
            batching_guidance=True,
            compaction_mode="native",
            compaction_threshold=100_000,
        ),
    )
    wheel = tmp_path / "kedi.whl"
    wheel.touch()
    kedi_otel_distribution = tmp_path / "opentelemetry_instrumentation_kedi.whl"
    kedi_otel_distribution.touch()
    codex_auth_helper_distribution = tmp_path / "codex_auth_helper.whl"
    codex_auth_helper_distribution.touch()
    tree_sitter_distribution = tmp_path / "tree-sitter-kedi.tar.gz"
    tree_sitter_distribution.touch()
    command = host._harbor_run_command(
        manifest,
        harbor_command="harbor",
        kedi_wheel=wheel,
        kedi_package=None,
        codex_auth_helper_distribution=codex_auth_helper_distribution,
        kedi_otel_distribution=kedi_otel_distribution,
        jobs_dir=tmp_path / "jobs",
        job_name="pilot",
        tree_sitter_kedi_distribution=tree_sitter_distribution,
    )
    assert command[:3] == ["harbor", "run", "--dataset"]
    assert command[3] == "terminal-bench/terminal-bench-2-1@6"
    assert "kedi.integrations.harbor:KediAgent" in command
    assert f"tree_sitter_kedi_revision={'a' * 40}" in command
    assert f"tree_sitter_kedi_distribution={tree_sitter_distribution.resolve()}" in command
    assert f"kedi_otel_distribution={kedi_otel_distribution.resolve()}" in command
    assert f"codex_auth_helper_distribution={codex_auth_helper_distribution.resolve()}" in command
    assert command.count("--include-task-name") == 2
    task_filters = [
        command[index + 1] for index, value in enumerate(command) if value == "--include-task-name"
    ]
    assert task_filters == ["terminal-bench/task-a", "terminal-bench/task-b"]
    assert f"manifest_digest={manifest.material_digest}" in command
    assert "history_enabled=true" in command
    assert "artifacts_enabled=true" in command
    assert "codemode_enabled=false" in command
    assert "batching_guidance_enabled=true" in command
    assert "retained_process_timeout_seconds=3600.0" in command
    assert "model_request_limit=512" in command
    assert "model_cumulative_input_tokens=50000000" in command
    assert not any("history_archive" in item for item in command)
    assert "command_timeout_seconds=600.0" in command
    assert command[command.index("--timeout-multiplier") + 1] == "1.25"
    assert command[command.index("--agent-timeout-multiplier") + 1] == "1.5"
    assert command[command.index("--verifier-timeout-multiplier") + 1] == "1.75"
    assert command[command.index("--agent-setup-timeout-multiplier") + 1] == "2.0"
    assert command[command.index("--environment-build-timeout-multiplier") + 1] == "2.25"
    assert command[command.index("--max-retries") + 1] == "2"
    assert command.count("--yes") == 1
    assert command[command.index("--retry-include") + 1] == "ApiRateLimitError"
    assert command[command.index("--retry-exclude") + 1] == "AgentTimeoutError"

    archive_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        model="openai/gpt-test",
        policy=HarnessPolicy(
            history_archive_threshold_tokens=32_000,
            history_archive_preserve_tokens=12_000,
        ),
    )
    archive_command = host._harbor_run_command(
        archive_manifest,
        harbor_command="harbor",
        kedi_wheel=wheel,
        kedi_package=None,
        jobs_dir=tmp_path / "jobs",
        job_name="archive",
    )
    assert "history_archive_threshold_tokens=32000" in archive_command
    assert "history_archive_preserve_tokens=12000" in archive_command

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


def test_host_run_resumes_the_same_started_job_and_skips_a_completed_job(
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
    job_dir = tmp_path / "jobs" / "pilot"
    job_dir.mkdir(parents=True)
    manifest.write(job_dir / "kedi-manifest.json")
    (job_dir / "lock.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(host, "_resolve_executable", lambda *_args, **_kwargs: "harbor")
    monkeypatch.setattr(host, "_validate_harbor_version", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    arguments = [
        "run",
        str(manifest_path),
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--job-name",
        "pilot",
    ]

    resumed = CliRunner().invoke(host.cli, arguments)

    assert resumed.exit_code == 0, resumed.output
    assert calls == [["harbor", "job", "resume", "-p", str(job_dir.resolve())]]

    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-09-11T00:00:00Z",
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 1,
                    "n_cancelled_trials": 0,
                    "n_pending_trials": 0,
                    "n_running_trials": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    completed = CliRunner().invoke(host.cli, arguments)

    assert completed.exit_code == 0, completed.output
    assert "already complete" in completed.output
    assert len(calls) == 1

    explicit_resume = CliRunner().invoke(host.cli, ["resume", str(job_dir)])
    assert explicit_resume.exit_code == 0, explicit_resume.output
    assert "already complete" in explicit_resume.output
    assert len(calls) == 1

    (job_dir / "result.json").write_text("{broken", encoding="utf-8")
    assert host._job_is_complete(job_dir) is False


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
            "--model-request-limit",
            "777",
            "--model-cumulative-input-tokens",
            "123000000",
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
    assert manifest.policy.model_request_limit == 777
    assert manifest.policy.model_cumulative_input_tokens == 123_000_000

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

    tree_sitter_distribution = tmp_path / "tree-sitter-kedi.tar.gz"
    tree_sitter_distribution.write_bytes(b"exact-tree-sitter-distribution")
    tree_sitter_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        tree_sitter_kedi_revision="a" * 40,
        tree_sitter_kedi_distribution_sha256=sha256_file(tree_sitter_distribution),
        model="openai/gpt-test",
    )
    assert (
        host._resolve_tree_sitter_kedi_distribution(
            tree_sitter_manifest,
            distribution=tree_sitter_distribution,
        )
        == tree_sitter_distribution.resolve()
    )
    with pytest.raises(UsageError, match="requires --tree-sitter-kedi-distribution"):
        host._resolve_tree_sitter_kedi_distribution(
            tree_sitter_manifest,
            distribution=None,
        )
    with pytest.raises(ClickException, match="SHA-256"):
        host._resolve_tree_sitter_kedi_distribution(
            tree_sitter_manifest,
            distribution=changed,
        )
    with pytest.raises(UsageError, match="does not record"):
        host._resolve_tree_sitter_kedi_distribution(
            manifest,
            distribution=tree_sitter_distribution,
        )

    kedi_otel_distribution = tmp_path / "opentelemetry_instrumentation_kedi.whl"
    kedi_otel_distribution.write_bytes(b"exact-kedi-otel-distribution")
    kedi_otel_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        kedi_otel_distribution_sha256=sha256_file(kedi_otel_distribution),
        model="openai/gpt-test",
    )
    assert (
        host._resolve_kedi_otel_distribution(
            kedi_otel_manifest,
            distribution=kedi_otel_distribution,
        )
        == kedi_otel_distribution.resolve()
    )
    with pytest.raises(UsageError, match="requires --kedi-otel-distribution"):
        host._resolve_kedi_otel_distribution(
            kedi_otel_manifest,
            distribution=None,
        )
    with pytest.raises(ClickException, match="SHA-256"):
        host._resolve_kedi_otel_distribution(
            kedi_otel_manifest,
            distribution=changed,
        )
    with pytest.raises(UsageError, match="does not record"):
        host._resolve_kedi_otel_distribution(
            manifest,
            distribution=kedi_otel_distribution,
        )

    codex_auth_helper_distribution = tmp_path / "codex_auth_helper.whl"
    codex_auth_helper_distribution.write_bytes(b"exact-codex-helper-distribution")
    codex_helper_manifest = RunManifest(
        harbor_revision="harbor-commit",
        kedi_commit="kedi-commit",
        kedi_dirty=False,
        codex_auth_helper_distribution_sha256=sha256_file(codex_auth_helper_distribution),
        model="codex/gpt-test",
    )
    assert (
        host._resolve_codex_auth_helper_distribution(
            codex_helper_manifest,
            distribution=codex_auth_helper_distribution,
        )
        == codex_auth_helper_distribution.resolve()
    )
    with pytest.raises(UsageError, match="requires --codex-auth-helper-distribution"):
        host._resolve_codex_auth_helper_distribution(
            codex_helper_manifest,
            distribution=None,
        )
    with pytest.raises(ClickException, match="SHA-256"):
        host._resolve_codex_auth_helper_distribution(
            codex_helper_manifest,
            distribution=changed,
        )
    with pytest.raises(UsageError, match="does not record"):
        host._resolve_codex_auth_helper_distribution(
            manifest,
            distribution=codex_auth_helper_distribution,
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
            self.logger = logging.getLogger("kedi.tests.harbor")
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


@pytest.mark.parametrize(
    ("policy", "explicit", "expected"),
    [
        ({}, None, 900),
        ({"timeout_multiplier": 2}, None, 1800),
        ({"timeout_multiplier": 3, "agent_timeout_multiplier": 0.5}, None, 450),
        (
            {
                "agent": {"override_timeout_sec": 1000, "max_timeout_sec": 800},
                "agent_timeout_multiplier": 2,
            },
            None,
            1600,
        ),
        ({}, 120, 120),
        ({}, 1200, 900),
    ],
)
def test_harbor_agent_resolves_task_deadline_and_accounts_for_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, Any],
    explicit: int | None,
    expected: float,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor import agent as agent_module

    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True)
    (logs.parent / "config.json").write_text(json.dumps(policy))
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir.parent / "task.toml").write_text("[agent]\ntimeout_sec = 900\n")
    env = SimpleNamespace(environment_dir=environment_dir, configs=[], agent_calls=[])
    agent = agent_module.KediAgent(
        logs_dir=logs, model_name="openai/test", runner_timeout_seconds=explicit
    )
    clock = [1000.0]
    monkeypatch.setattr(agent_module.time, "time", lambda: clock[0])

    async def upload(_: Any, **kwargs: Any) -> None:
        env.configs.append(kwargs)
        clock[0] += 7

    monkeypatch.setattr(agent, "_upload_config_text", upload)
    asyncio.run(agent.run("task", env, SimpleNamespace(metadata=None)))
    call = env.agent_calls[-1]
    assert f"--trial-timeout-seconds {expected}" in call["command"]
    assert f"--trial-deadline-unix {1000.0 + expected}" in call["command"]
    assert call["timeout_sec"] == expected - 7 + 30


def test_harbor_agent_lifecycle_and_context_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_HARBOR_LOGFIRE", "1")
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "none")
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    credentials_dir = tmp_path / "logfire"
    credentials_dir.mkdir()
    (credentials_dir / "logfire_credentials.json").write_text(
        json.dumps(
            {
                "token": "logfire-test-token",
                "project_name": "test",
                "project_url": "https://example.test/project",
                "logfire_api_url": "https://example.test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOGFIRE_CREDENTIALS_DIR", str(credentials_dir))
    base = _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    KediAgent = agent_module.KediAgent
    import kedi.integrations.harbor as integration

    assert integration.__getattr__("KediAgent") is KediAgent
    assert issubclass(KediAgent, base)

    wheel = tmp_path / "kedi.whl"
    wheel.write_bytes(b"wheel")
    kedi_otel_distribution = tmp_path / "opentelemetry_instrumentation_kedi.whl"
    kedi_otel_distribution.write_bytes(b"instrumentation")
    tree_sitter_distribution = tmp_path / "tree-sitter-kedi.tar.gz"
    tree_sitter_distribution.write_bytes(b"source distribution")
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
        kedi_otel_distribution=kedi_otel_distribution,
        history_enabled=False,
        prefix_cache_enabled=False,
        artifacts_enabled=False,
        runner_timeout_seconds=10,
        manifest_digest="manifest",
        tree_sitter_kedi_revision="a" * 40,
        tree_sitter_kedi_distribution=tree_sitter_distribution,
    )
    asyncio.run(agent.install(environment))
    assert environment.uploaded[0][1].endswith("/kedi.whl")
    assert environment.uploaded[1][1].endswith("/tree-sitter-kedi.tar.gz")
    assert environment.uploaded[2][1].endswith("/opentelemetry_instrumentation_kedi.whl")
    assert "python_venv" in environment.dependencies
    assert "build_tools" in environment.dependencies
    install_command = environment.root_calls[-1]["command"]
    assert "/installed-agent/tree-sitter-kedi.tar.gz" in install_command
    assert "git+https://github.com/kedi-lang/tree-sitter-kedi.git" not in install_command
    assert "sys.version_info" not in install_command
    assert "python3 -m venv /installed-agent/kedi-bootstrap" in install_command
    assert "python3 -m venv /installed-agent/kedi &&" not in install_command
    assert "uv==0.11.21" in install_command
    assert install_command.count("--retries 20 --timeout 120") == 2
    assert "uv venv --python 3.11 --managed-python --seed" in install_command
    assert "logfire[httpx]==4.41.0" in install_command
    assert "/installed-agent/opentelemetry_instrumentation_kedi.whl" in install_command
    assert "opentelemetry-instrumentation-kedi==0.1.0" not in install_command

    revision_environment = SimpleNamespace(
        dependencies=(),
        root_calls=[],
        agent_calls=[],
        configs=[],
        uploaded=[],
        upload_file=upload_file,
    )
    revision_agent = KediAgent(
        logs_dir=tmp_path / "revision-logs",
        model_name="openrouter/test-model",
        kedi_package="kedi",
        tree_sitter_kedi_revision="a" * 40,
    )
    asyncio.run(revision_agent.install(revision_environment))
    revision_install_command = revision_environment.root_calls[-1]["command"]
    assert (
        f"tree-sitter-kedi @ git+https://github.com/kedi-lang/tree-sitter-kedi.git@{'a' * 40}"
    ) in revision_install_command

    context = SimpleNamespace(metadata=None)
    asyncio.run(agent.run("solve task", environment, context))
    assert context.metadata is None
    assert environment.configs[0]["content"] == "rendered: solve task"
    run_call = environment.agent_calls[-1]
    assert run_call["timeout_sec"] == 40
    assert run_call["command"].startswith("set -o pipefail; ")
    assert "/installed-agent/kedi-runner.pid" in run_call["command"]
    assert "| tee" not in run_call["command"]
    assert "--no-history-enabled" in run_call["command"]
    assert "--no-artifacts-enabled" in run_call["command"]
    assert "--no-codemode-enabled" in run_call["command"]
    assert "--model-request-limit 512" in run_call["command"]
    assert "--model-cumulative-input-tokens 50000000" in run_call["command"]
    assert "sk-provider-secret-value" not in run_call["command"]
    assert "logfire-test-token" not in run_call["command"]
    assert run_call["env"]["KEDI_HARBOR_LOGFIRE"] == "1"
    assert run_call["env"]["KEDI_WORKSPACE_POLICY"] == "none"
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


def test_harbor_agent_installs_pinned_codex_helper_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEDI_HARBOR_LOGFIRE", raising=False)
    base = _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent
    assert issubclass(KediAgent, base)

    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    helper_wheel = tmp_path / "codex_auth_helper-1.8.0-py3-none-any.whl"
    helper_wheel.write_bytes(b"local helper")
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
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="codex/gpt-test",
        kedi_package="kedi",
        codex_auth_file=auth_file,
        codex_auth_helper_distribution=helper_wheel,
    )
    asyncio.run(agent.install(environment))

    assert environment.uploaded == [
        (helper_wheel.resolve(), "/installed-agent/codex_auth_helper-1.8.0-py3-none-any.whl")
    ]
    install_command = environment.root_calls[-1]["command"]
    assert "/installed-agent/codex_auth_helper-1.8.0-py3-none-any.whl" in install_command
    assert "codex-auth-helper[websocket]==1.8.0" not in install_command

    with pytest.raises(ValueError, match="requires a codex/ model"):
        KediAgent(
            logs_dir=tmp_path / "invalid-logs",
            model_name="openai/gpt-test",
            kedi_package="kedi",
            codex_auth_helper_distribution=helper_wheel,
        )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_harbor_runner_wrapper_records_actual_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    import kedi.integrations.harbor.agent as agent_module

    logs = tmp_path / "logs"
    logs.mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    executable = runtime / "bin" / "python"
    executable.write_text(f"#!/bin/sh\nprintf 'runner-output\\n'\nexit {exit_code}\n")
    executable.chmod(0o700)
    pid_file = tmp_path / "runner.pid"
    monkeypatch.setattr(agent_module, "_RUNNER_PID", PurePosixPath(pid_file))
    monkeypatch.setattr(agent_module.KediAgent, "_VENV", PurePosixPath(runtime))
    agent = agent_module.KediAgent(
        logs_dir=logs, environment_logs_dir=PurePosixPath(logs), model_name="openai/test"
    )
    environment = SimpleNamespace(configs=[], agent_calls=[])

    async def no_skills(*_args: Any, **_kwargs: Any) -> None:
        pass

    monkeypatch.setattr(agent, "_install_skills", no_skills)
    asyncio.run(agent.run("fixture", environment, SimpleNamespace()))
    command = environment.agent_calls[-1]["command"]
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5)
    assert result.returncode == exit_code
    record = json.loads((logs / "runner-exit.json").read_text())
    assert record["exit_code"] == exit_code
    assert record["finished_at"].endswith("Z")
    assert (logs / "kedi.txt").read_text() == "runner-output\n"
    assert not pid_file.exists()
    assert not (logs / ".runner-exit.tmp").exists()


def test_harbor_agent_finalizes_remote_runner_when_harbor_cancels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        kedi_package="kedi",
    )
    calls: list[dict[str, Any]] = []

    async def no_upload(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def cancel_then_finalize(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            raise asyncio.CancelledError
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(agent, "_upload_config_text", no_upload)
    monkeypatch.setattr(agent, "_install_skills", no_upload)
    monkeypatch.setattr(agent, "exec_as_agent", cancel_then_finalize)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run("solve", SimpleNamespace(), SimpleNamespace(metadata=None)))

    assert len(calls) == 2
    assert "/installed-agent/kedi-runner.pid" in calls[0]["command"]
    assert "kill -TERM" in calls[1]["command"]
    assert "kedi.integrations.harbor.runner" in calls[1]["command"]
    assert calls[1]["timeout_sec"] == 25


def test_harbor_agent_bounds_unresponsive_remote_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    monkeypatch.setattr(agent_module, "_RUNNER_FINALIZATION_TIMEOUT_SECONDS", 0.05, raising=False)
    agent = agent_module.KediAgent(
        logs_dir=tmp_path / "logs", model_name="openai/test", kedi_package="kedi"
    )
    cleaned = False

    async def stalled_exec(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal cleaned
        try:
            await asyncio.Event().wait()
        finally:
            cleaned = True

    monkeypatch.setattr(agent, "exec_as_agent", stalled_exec)

    async def check() -> None:
        task = asyncio.create_task(agent._finalize_cancelled_runner(SimpleNamespace(), "trial"))
        try:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in done, "remote cleanup has no host-side deadline"
            with pytest.raises(asyncio.TimeoutError):
                task.result()
            assert cleaned
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_harbor_agent_repeated_cancellation_does_not_leave_cleanup_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    agent = agent_module.KediAgent(
        logs_dir=tmp_path / "logs", model_name="openai/test", kedi_package="kedi"
    )

    async def no_upload(*_args: Any, **_kwargs: Any) -> None:
        pass

    monkeypatch.setattr(agent, "_upload_config_text", no_upload)
    monkeypatch.setattr(agent, "_install_skills", no_upload)

    async def check() -> None:
        finalizing = asyncio.Event()
        cleaned = asyncio.Event()
        calls = 0

        async def remote(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError
            finalizing.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(agent, "exec_as_agent", remote)
        task = asyncio.create_task(agent.run("solve", SimpleNamespace(), SimpleNamespace()))
        await asyncio.wait_for(finalizing.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(check())


@pytest.mark.parametrize("exit_code", [0, 23])
def test_harbor_setup_records_phases_and_preserves_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor.agent import _setup_command

    log = tmp_path / "new logs" / "setup with spaces.log"
    command = _setup_command(
        (
            ("bootstrap", "printf 'bootstrap-output\\n'"),
            ("runtime", f"printf 'runtime-output\\n'; (exit {exit_code})"),
            ("last", "printf 'last-output\\n'"),
        ),
        PurePosixPath(log),
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5)
    assert result.returncode == exit_code
    assert result.stdout == log.read_text()
    assert "phase=bootstrap state=completed" in result.stdout
    assert "runtime-output" in result.stdout
    assert ("phase=runtime state=completed" in result.stdout) is (exit_code == 0)
    assert ("last-output" in result.stdout) is (exit_code == 0)


def test_harbor_setup_keeps_partial_output_when_install_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor.agent import _setup_command

    log = tmp_path / "setup.log"
    command = _setup_command(
        (("bootstrap", "printf 'before-wait\\n'; sleep 30"), ("last", "true")),
        PurePosixPath(log),
    )
    process = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not log.exists() or "before-wait" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert process.poll() is None
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=3)
    output = log.read_text()
    assert "phase=bootstrap state=started" in output
    assert "before-wait" in output
    assert "state=completed" not in output
    assert "phase=last" not in output


def test_harbor_setup_stops_when_an_early_and_list_command_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    from kedi.integrations.harbor.agent import _setup_command

    log = tmp_path / "setup.log"
    command = _setup_command(
        (("bootstrap", "false && printf 'must-not-install'"), ("runtime", "true")),
        PurePosixPath(log),
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "phase=bootstrap state=started" in result.stdout
    assert "state=completed" not in result.stdout
    assert "phase=runtime" not in result.stdout
    assert "must-not-install" not in result.stdout


def test_harbor_agent_preserves_cancellation_when_remote_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent
    agent = KediAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        kedi_package="kedi",
    )
    calls = 0

    async def no_upload(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def fail_finalization(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        raise RuntimeError("remote environment already closed")

    monkeypatch.setattr(agent, "_upload_config_text", no_upload)
    monkeypatch.setattr(agent, "_install_skills", no_upload)
    monkeypatch.setattr(agent, "exec_as_agent", fail_finalization)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run("solve", SimpleNamespace(), SimpleNamespace(metadata=None)))

    assert calls == 2


def test_harbor_logfire_token_resolution_is_optional_and_env_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    agent_module = importlib.import_module("kedi.integrations.harbor.agent")
    credentials_dir = tmp_path / "logfire"
    credentials_dir.mkdir()
    credentials_path = credentials_dir / "logfire_credentials.json"
    monkeypatch.setenv("LOGFIRE_CREDENTIALS_DIR", str(credentials_dir))
    monkeypatch.setenv("LOGFIRE_TOKEN", "explicit-token")
    assert agent_module._resolve_logfire_token() == "explicit-token"

    monkeypatch.delenv("LOGFIRE_TOKEN")
    assert agent_module._resolve_logfire_token() is None
    credentials_path.write_text("not-json", encoding="utf-8")
    assert agent_module._resolve_logfire_token() is None
    credentials_path.write_text('{"token": 42}', encoding="utf-8")
    assert agent_module._resolve_logfire_token() is None


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
    assert "codex-auth-helper[websocket]==1.8.0" in install_command
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


@pytest.mark.parametrize("archival", [False, True])
def test_harbor_agent_full_configuration_and_failure_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archival: bool,
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
    with pytest.raises(FileNotFoundError):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            tree_sitter_kedi_distribution=tmp_path / "missing.tar.gz",
        )
    with pytest.raises(ValueError, match="runner_timeout_seconds"):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            kedi_package="kedi",
            runner_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="runner_timeout_seconds"):
        KediAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            kedi_package="kedi",
            runner_timeout_seconds=cast(Any, float("inf")),
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
        compaction_mode="disabled" if archival else "native",
        compaction_threshold=None if archival else 1000,
        skills_dir="/skills",
        history_archive_threshold_tokens=32000 if archival else None,
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
    assert ("--compaction-threshold 1000" in command) is not archival
    assert ("--history-archive-threshold-tokens 32000" in command) is archival

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

    result_path.write_text("[]", encoding="utf-8")
    non_object_context = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(non_object_context)
    assert non_object_context.metadata["kedi_result_error"] == "TypeError"

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


def test_harbor_agent_recovers_newer_request_usage_after_runner_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_harbor(monkeypatch)
    monkeypatch.delitem(sys.modules, "kedi.integrations.harbor.agent", raising=False)
    KediAgent = importlib.import_module("kedi.integrations.harbor.agent").KediAgent
    agent = KediAgent(logs_dir=tmp_path / "logs", model_name="openai/test")
    agent.logs_dir.mkdir(parents=True)
    (agent.logs_dir / "kedi-result.json").write_text(
        json.dumps(
            {
                "state": "timed_out",
                "adapter": "pydantic",
                "usage": {
                    "requests": 0,
                    "tool_calls": 0,
                    "input_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "uncached_input_tokens": 0,
                    "cost_usd": None,
                },
                "verification": {"state": "not_run"},
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "event": "model_request_completed",
            "response": {
                "parts": [{"kind": "tool-call"}],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_tokens": 60,
                    "cache_write_tokens": 5,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "cost_usd": 0.01,
                },
            },
        },
        {
            "event": "model_request_completed",
            "response": {
                "parts": [{"kind": "text"}],
                "usage": {
                    "input_tokens": 200,
                    "cache_read_tokens": 150,
                    "cache_write_tokens": 0,
                    "output_tokens": 20,
                    "total_tokens": 220,
                    "cost_usd": 0.02,
                },
            },
        },
    ]
    (agent.logs_dir / "model-requests.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(metadata={"harbor": "preserved"})

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 300
    assert context.n_cache_tokens == 210
    assert context.n_output_tokens == 30
    assert context.cost_usd == pytest.approx(0.03)
    assert context.metadata["harbor"] == "preserved"
    assert context.metadata["kedi_state"] == "timed_out"
    assert context.metadata["kedi_requests"] == 2
    assert context.metadata["kedi_tool_calls"] == 1
    assert context.metadata["kedi_uncached_input_tokens"] == 90
    assert context.metadata["kedi_usage_source"] == "model_request_evidence"
    assert context.metadata["kedi_usage_recovered"] is True

    (agent.logs_dir / "kedi-result.json").unlink()
    evidence_only_context = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(evidence_only_context)
    assert evidence_only_context.n_input_tokens == 300
    assert evidence_only_context.metadata["kedi_state"] == "integration_failure"
    assert evidence_only_context.metadata["kedi_result_present"] is False
    assert evidence_only_context.metadata["kedi_usage_recovered"] is True


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
