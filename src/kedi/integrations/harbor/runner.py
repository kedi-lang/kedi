from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tempfile
import time
import traceback
from importlib.resources import files
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, NoReturn, Sequence, cast

import kedi

from ...agent_adapter.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ...agent_adapter.conversation import ConversationState
from ...agent_adapter.invocation import InvocationUsage, observe_invocations
from ...agent_profile import McpServerSpec
from ...agent_profile.mcp import mcp_spec_from_mapping
from ...artifacts import ArtifactPolicy
from ...effort import EFFORT_VALUES, EffortValue
from .records import (
    TrialRecord,
    TrialState,
    atomic_write_json,
    invocation_usage_dict,
    redact_text,
    utc_now,
)
from .terminal import TerminalSession

_ALLOWED_MUTATING_TOOLS = frozenset(
    {
        "apply_patch",
        "create_directory",
        "remove_directory",
        "remove_file",
        "release_artifact",
        "run_artifact_code",
        "run_command",
        "run_shell",
        "start_process",
        "start_shell",
        "stop_process",
        "write_process_input",
        "write_text_file",
    }
)


def benchmark_approval(request: ApprovalRequest) -> ApprovalDecision:
    """Non-interactive policy for tools already bounded to the task container."""

    if request.risk == "sensitive":
        return ApprovalDecision.deny(reason="benchmark policy denies sensitive operations")
    if request.risk == "read_only":
        return ApprovalDecision.allow(reason="read-only operation inside the task sandbox")
    if request.tool_name in _ALLOWED_MUTATING_TOOLS:
        return ApprovalDecision.allow(reason="sandbox-bounded benchmark operation")
    return ApprovalDecision.deny(reason="tool is outside the benchmark allowlist")


def render_agent_source(
    *,
    history_enabled: bool = True,
    compaction_mode: str,
    compaction_threshold: int | None,
) -> str:
    source = (
        files("kedi.integrations.harbor")
        .joinpath("benchmark_agent.kedi")
        .read_text(encoding="utf-8")
    )
    if compaction_mode not in {"disabled", "native"}:
        raise ValueError("compaction_mode must be disabled or native")
    if compaction_threshold is not None and compaction_threshold <= 0:
        raise ValueError("compaction_threshold must be positive")
    if compaction_mode == "disabled" and compaction_threshold is not None:
        raise ValueError("compaction_threshold requires native compaction")
    if not history_enabled:
        if compaction_mode != "disabled":
            raise ValueError("native compaction requires history")
        return source.replace("# KEDI_HARBOR_HISTORY", "> history: disabled", 1)
    history = ["> history:", "    enabled: true", f"    compaction_mode: {compaction_mode}"]
    if compaction_threshold is not None:
        history.append(f"    compaction_threshold: `{compaction_threshold}`")
    return source.replace("# KEDI_HARBOR_HISTORY", "\n".join(history), 1)


def run_trial(
    *,
    instruction: str,
    model: str,
    adapter: str,
    workspace: Path,
    result_path: Path,
    trial_id: str,
    manifest_digest: str | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    effort: str | None = None,
    model_settings: Mapping[str, Any] | None = None,
    history_enabled: bool = True,
    artifacts_enabled: bool = True,
    compaction_mode: str = "disabled",
    compaction_threshold: int | None = None,
    artifact_threshold_bytes: int = 16_384,
    artifact_quota_bytes: int = 256_000_000,
    command_timeout_seconds: float = 120.0,
    terminal_preview_bytes: int = 32_768,
    terminal_output_limit_bytes: int = 512_000_000,
    trial_timeout_seconds: float | None = None,
) -> TrialRecord:
    started_at = utc_now()
    started_monotonic = time.monotonic()
    workspace = workspace.resolve()
    result_path = result_path.resolve()
    records_dir = result_path.parent
    records_dir.mkdir(parents=True, exist_ok=True)
    deadline = (
        started_monotonic + trial_timeout_seconds if trial_timeout_seconds is not None else None
    )
    terminal = TerminalSession(
        root=workspace,
        records_dir=records_dir,
        command_timeout_seconds=command_timeout_seconds,
        preview_bytes=terminal_preview_bytes,
        output_limit_bytes=terminal_output_limit_bytes,
        deadline_monotonic=deadline,
        artifacts_enabled=artifacts_enabled,
    )
    usage = InvocationUsage()
    state: TrialState = "completed"
    result_summary: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    failure_phase: str | None = None
    previous_cwd = Path.cwd()
    signal_state = _SignalState(terminal)
    source_path: Path | None = None
    conversation: ConversationState | None = None
    phase = "setup"
    cleanup_errors: list[str] = []

    def observe(item: InvocationUsage) -> None:
        nonlocal usage
        usage = usage + item

    try:
        if trial_timeout_seconds is not None and trial_timeout_seconds <= 0:
            raise ValueError("trial_timeout_seconds must be positive")
        if not workspace.is_dir():
            raise NotADirectoryError(str(workspace))
        os.chdir(workspace)
        signal_state.install(timeout_seconds=trial_timeout_seconds)
        source = render_agent_source(
            history_enabled=history_enabled,
            compaction_mode=compaction_mode,
            compaction_threshold=compaction_threshold,
        )
        fd, raw_source_path = tempfile.mkstemp(
            prefix="kedi-harbor-agent-",
            suffix=".kedi",
            dir=records_dir,
        )
        os.close(fd)
        source_path = Path(raw_source_path)
        source_path.write_text(source, encoding="utf-8")
        if history_enabled:
            conversation = ConversationState(session_id=f"harbor_{trial_id}")
        artifacts: ArtifactPolicy | bool
        if artifacts_enabled:
            artifacts = ArtifactPolicy.patch(
                {
                    "enabled": True,
                    "store": "file",
                    "path": records_dir / "artifacts",
                    "threshold_bytes": artifact_threshold_bytes,
                    "session_quota_bytes": artifact_quota_bytes,
                }
            )
        else:
            artifacts = False

        @kedi.bind(
            file=source_path,
            model=model,
            adapter=adapter,
            effort=_effort(effort),
            settings=dict(model_settings or {}),
            tools=terminal.tools,
            mcp_servers=mcp_servers,
            approval=ApprovalPolicy.dynamic(benchmark_approval),
            hooks={"post_tool_use": terminal.hook_handler},
            skills=True,
            artifacts=artifacts,
            conversation=conversation,
        )
        def solve(instruction: str) -> str:
            raise AssertionError(  # pragma: no cover - kedi.bind replaces this function
                "Kedi bind wrappers do not execute the Python function body"
            )

        phase = "agent"
        with observe_invocations(observe):
            result_summary = solve(instruction)
        result_summary = redact_text(result_summary)
    except _TrialTimedOut as exc:
        state = "timed_out"
        failure_phase = phase
        error_type = type(exc).__name__
        error_message = str(exc)
    except KeyboardInterrupt as exc:
        state = "cancelled"
        failure_phase = phase
        error_type = type(exc).__name__
        error_message = "trial cancelled"
    except BaseException as exc:
        state = "agent_failure" if phase == "agent" else "integration_failure"
        failure_phase = phase
        error_type = type(exc).__name__
        error_message = redact_text(str(exc))
        _write_error(records_dir, exc)
    finally:
        for cleanup_name, cleanup in (
            ("conversation", conversation.close if conversation is not None else None),
            ("terminal", terminal.close),
            ("signals", signal_state.restore),
            ("cwd", lambda: os.chdir(previous_cwd)),
            (
                "source",
                (lambda: source_path.unlink(missing_ok=True)) if source_path is not None else None,
            ),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as exc:
                cleanup_errors.append(
                    f"{cleanup_name}: {type(exc).__name__}: {redact_text(str(exc))}"
                )

    if cleanup_errors:
        cleanup_message = "; ".join(cleanup_errors)
        (records_dir / "cleanup-errors.txt").write_text(
            cleanup_message + "\n",
            encoding="utf-8",
        )
        if state == "completed":
            state = "integration_failure"
            failure_phase = "teardown"
            error_type = "CleanupError"
            error_message = cleanup_message

    terminal_summary = terminal.summary()
    duration = max(0.0, time.monotonic() - started_monotonic)
    record = TrialRecord(
        schema_version=1,
        trial_id=trial_id,
        state=state,
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=duration,
        instruction_digest=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        workspace=str(workspace),
        model=model,
        adapter=adapter,
        manifest_digest=manifest_digest,
        usage=invocation_usage_dict(usage),
        verification=terminal_summary["verification"],
        terminal={key: value for key, value in terminal_summary.items() if key != "verification"},
        policy={
            "history": history_enabled,
            "artifacts": artifacts_enabled,
            "prefix_cache": history_enabled,
            "compaction_mode": compaction_mode,
            "compaction_threshold": compaction_threshold,
            "artifact_threshold_bytes": artifact_threshold_bytes,
            "artifact_quota_bytes": artifact_quota_bytes,
            "command_timeout_seconds": command_timeout_seconds,
            "terminal_preview_bytes": terminal_preview_bytes,
            "terminal_output_limit_bytes": terminal_output_limit_bytes,
            "trial_timeout_seconds": trial_timeout_seconds,
        },
        result_summary=result_summary,
        failure_phase=failure_phase,
        error_type=error_type,
        error_message=error_message,
    )
    atomic_write_json(result_path, record.to_dict())
    return record


class _TrialTimedOut(TimeoutError):
    pass


class _SignalState:
    def __init__(self, terminal: TerminalSession) -> None:
        self.terminal = terminal
        self._previous: dict[int, Any] = {}
        self._previous_alarm: tuple[float, float] | None = None

    def install(self, *, timeout_seconds: float | None) -> None:
        if threading_is_main():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            if hasattr(signal, "SIGALRM"):
                self._previous[signal.SIGALRM] = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, self._handle)
                if timeout_seconds is not None:
                    self._previous_alarm = signal.setitimer(
                        signal.ITIMER_REAL,
                        timeout_seconds,
                    )

    def restore(self) -> None:
        if self._previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        if self._previous_alarm is not None and self._previous_alarm[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *self._previous_alarm)
        self._previous_alarm = None
        self._previous.clear()

    def _handle(self, signum: int, _: FrameType | None) -> NoReturn:
        self.terminal.close()
        if signum == signal.SIGTERM or signum == getattr(signal, "SIGALRM", None):
            raise _TrialTimedOut("trial terminated by Harbor")
        raise KeyboardInterrupt


def _write_error(records_dir: Path, exc: BaseException) -> None:
    try:
        (records_dir / "error.txt").write_text(
            redact_text("".join(traceback.format_exception(exc))),
            encoding="utf-8",
        )
    except OSError:
        return


def threading_is_main() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


def _mcp_servers(path: Path | None) -> tuple[McpServerSpec, ...]:
    if path is None:
        return ()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise TypeError("MCP configuration must be a JSON list")
    servers: list[McpServerSpec] = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("each MCP configuration must be an object")
        servers.append(mcp_spec_from_mapping(value))
    return tuple(servers)


def _settings(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("model settings must be a JSON object")
    return value


def _effort(value: str | None) -> EffortValue | None:
    if value is None:
        return None
    if value not in EFFORT_VALUES:
        raise ValueError(f"effort must be one of: {', '.join(EFFORT_VALUES)}")
    return cast(EffortValue, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kedi-harbor-runner")
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="pydantic")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--manifest-digest")
    parser.add_argument("--mcp-config", type=Path)
    parser.add_argument("--model-settings", type=Path)
    parser.add_argument("--effort", choices=EFFORT_VALUES)
    parser.add_argument(
        "--history-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--artifacts-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compaction-mode", choices=("disabled", "native"), default="disabled")
    parser.add_argument("--compaction-threshold", type=int)
    parser.add_argument("--artifact-threshold-bytes", type=int, default=16_384)
    parser.add_argument("--artifact-quota-bytes", type=int, default=256_000_000)
    parser.add_argument("--command-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--terminal-preview-bytes", type=int, default=32_768)
    parser.add_argument("--terminal-output-limit-bytes", type=int, default=512_000_000)
    parser.add_argument("--trial-timeout-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record = run_trial(
        instruction=args.instruction_file.read_text(encoding="utf-8"),
        model=args.model,
        adapter=args.adapter,
        workspace=args.workspace,
        result_path=args.result_file,
        trial_id=args.trial_id,
        manifest_digest=args.manifest_digest,
        mcp_servers=_mcp_servers(args.mcp_config),
        effort=args.effort,
        model_settings=_settings(args.model_settings),
        history_enabled=args.history_enabled,
        artifacts_enabled=args.artifacts_enabled,
        compaction_mode=args.compaction_mode,
        compaction_threshold=args.compaction_threshold,
        artifact_threshold_bytes=args.artifact_threshold_bytes,
        artifact_quota_bytes=args.artifact_quota_bytes,
        command_timeout_seconds=args.command_timeout_seconds,
        terminal_preview_bytes=args.terminal_preview_bytes,
        terminal_output_limit_bytes=args.terminal_output_limit_bytes,
        trial_timeout_seconds=args.trial_timeout_seconds,
    )
    return 0 if record.state == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["benchmark_approval", "main", "render_agent_source", "run_trial"]
