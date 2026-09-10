from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from ...agent_adapter.request_budget import (
    ModelRequestBudgetError,
    ModelRequestBudgetLimits,
    use_model_request_budget_limits,
)
from ...agent_profile import McpServerSpec
from ...agent_profile.mcp import mcp_spec_from_mapping
from ...artifacts import ArtifactPolicy
from ...effort import EFFORT_VALUES, EffortValue
from ...errors import KediExecutionError
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
        "retain_process",
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
_FINALIZATION_RESERVE_SECONDS = 15.0
_BATCHING_GUIDANCE = """    For repeated, independent, or matrix-like operations, use one bounded tool invocation or
    script and return a concise aggregate instead of creating one model turn per operation. Reuse
    successful results unless relevant state changed."""


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
    codemode_enabled: bool = False,
    batching_guidance_enabled: bool = False,
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
    source = source.replace(
        "# KEDI_HARBOR_CODEMODE",
        "> codemode: enabled" if codemode_enabled else "> codemode: disabled",
        1,
    )
    source = source.replace(
        "    # KEDI_HARBOR_BATCHING_GUIDANCE",
        _BATCHING_GUIDANCE if batching_guidance_enabled else "",
        1,
    )
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
    codemode_enabled: bool = False,
    batching_guidance_enabled: bool = False,
    request_observation: bool = False,
    compaction_mode: str = "disabled",
    compaction_threshold: int | None = None,
    history_archive_threshold_tokens: int | None = None,
    history_archive_preserve_tokens: int = 12_000,
    artifact_threshold_bytes: int = 16_384,
    artifact_quota_bytes: int = 256_000_000,
    command_timeout_seconds: float = 600.0,
    terminal_preview_bytes: int = 32_768,
    terminal_output_limit_bytes: int = 512_000_000,
    retained_process_timeout_seconds: float = 3_600.0,
    model_request_limit: int = 512,
    model_cumulative_input_tokens: int = 50_000_000,
    trial_timeout_seconds: float | None = None,
    trial_deadline_unix: float | None = None,
) -> TrialRecord:
    logfire_enabled = _configure_logfire()
    started_at = utc_now()
    started_monotonic = time.monotonic()
    trial_timeout_valid = trial_timeout_seconds is None or (
        math.isfinite(trial_timeout_seconds) and trial_timeout_seconds > 0
    )
    recorded_trial_timeout: float | str | None = (
        trial_timeout_seconds if trial_timeout_valid else repr(trial_timeout_seconds)
    )
    deadline_valid = trial_deadline_unix is None or math.isfinite(trial_deadline_unix)
    remaining_timeout = trial_timeout_seconds if trial_timeout_valid else None
    if trial_deadline_unix is not None and deadline_valid:
        remaining = trial_deadline_unix - time.time()
        remaining_timeout = (
            min(remaining_timeout, remaining) if remaining_timeout is not None else remaining
        )
    workspace = workspace.resolve()
    result_path = result_path.resolve()
    records_dir = result_path.parent
    records_dir.mkdir(parents=True, exist_ok=True)
    deadline = started_monotonic + remaining_timeout if remaining_timeout is not None else None
    finalization_reserve_seconds = (
        min(_FINALIZATION_RESERVE_SECONDS, max(0.0, remaining_timeout) * 0.1)
        if remaining_timeout is not None
        else 0.0
    )
    terminal = TerminalSession(
        root=workspace,
        cwd_root=Path("/"),
        records_dir=records_dir,
        command_timeout_seconds=command_timeout_seconds,
        preview_bytes=terminal_preview_bytes,
        output_limit_bytes=terminal_output_limit_bytes,
        agent_deadline_monotonic=deadline,
        finalization_reserve_seconds=finalization_reserve_seconds,
        artifacts_enabled=artifacts_enabled,
        retained_process_timeout_seconds=retained_process_timeout_seconds,
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
    request_evidence: Any | None = None
    request_limit = model_request_limit
    phase = "setup"
    cleanup_errors: list[str] = []

    def observe(item: InvocationUsage) -> None:
        nonlocal usage
        usage = usage + item

    try:
        if not trial_timeout_valid:
            raise ValueError("trial_timeout_seconds must be positive")
        if not deadline_valid:
            raise ValueError("trial_deadline_unix must be finite")
        if deadline is not None and deadline <= time.monotonic():
            raise _TrialTimedOut("trial deadline elapsed before runner startup")
        budget_limits = ModelRequestBudgetLimits(
            cumulative_input_tokens=model_cumulative_input_tokens,
            request_limit=model_request_limit,
        )
        if not workspace.is_dir():
            raise NotADirectoryError(str(workspace))
        if history_archive_threshold_tokens is not None and adapter not in {
            "pydantic",
            "langchain",
        }:
            raise ValueError("history archival requires the Pydantic or LangChain adapter")
        if request_observation and adapter != "pydantic":
            raise ValueError("request observation currently requires the Pydantic adapter")
        os.chdir(workspace)
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise _TrialTimedOut("trial deadline elapsed during runner setup")
        signal_state.install(timeout_seconds=remaining)
        source = render_agent_source(
            history_enabled=history_enabled,
            codemode_enabled=codemode_enabled,
            batching_guidance_enabled=batching_guidance_enabled,
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

        runtime_adapter: str | object = adapter
        archive_settings = None
        if history_archive_threshold_tokens is not None:
            from ...agent_adapter.compaction import HistoryArchiveSettings

            archive_settings = HistoryArchiveSettings(
                threshold_tokens=history_archive_threshold_tokens,
                preserve_recent_tokens=history_archive_preserve_tokens,
                minimum_reduction_tokens=min(
                    2_000,
                    max(1, history_archive_threshold_tokens - history_archive_preserve_tokens),
                ),
            )
        if adapter == "pydantic":
            from pydantic_ai import UsageLimits

            from ...agent_adapter.adapters import PydanticAdapter

            capabilities: list[object] = []
            if request_observation:
                from .request_observer import PydanticRequestEvidenceCapability

                request_evidence = PydanticRequestEvidenceCapability(
                    records_dir=records_dir,
                    trial_id=trial_id,
                    manifest_digest=manifest_digest,
                )
                capabilities.append(request_evidence)

            runtime_adapter = PydanticAdapter(
                default_usage_limits=UsageLimits(request_limit=request_limit),
                capabilities=capabilities,
                history_archive=archive_settings,
            )
        elif adapter == "langchain":
            from pydantic_ai import UsageLimits

            from ...agent_adapter.adapters import LangChainAdapter

            runtime_adapter = LangChainAdapter(
                default_usage_limits=UsageLimits(request_limit=request_limit),
                history_archive=archive_settings,
            )

        @kedi.bind(
            file=source_path,
            model=model,
            adapter=runtime_adapter,
            effort=_effort(effort),
            settings=dict(model_settings or {}),
            tools=terminal.tools,
            mcp_servers=mcp_servers,
            approval=ApprovalPolicy.dynamic(benchmark_approval),
            hooks={
                "pre_tool_use": terminal.hook_handler,
                "post_tool_use": terminal.hook_handler,
                "post_tool_use_failure": terminal.hook_handler,
            },
            skills=False,
            artifacts=artifacts,
            conversation=conversation,
        )
        def solve(instruction: str) -> str:
            raise AssertionError(  # pragma: no cover - kedi.bind replaces this function
                "Kedi bind wrappers do not execute the Python function body"
            )

        phase = "agent"
        with use_model_request_budget_limits(budget_limits), observe_invocations(observe):
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
        if timed_out := _find_wrapped_exception(exc, _TrialTimedOut):
            state = "timed_out"
            failure_phase = phase
            error_type = type(timed_out).__name__
            error_message = str(timed_out)
        elif _find_wrapped_exception(exc, KeyboardInterrupt) is not None:
            state = "cancelled"
            failure_phase = phase
            error_type = "KeyboardInterrupt"
            error_message = "trial cancelled"
        else:
            if phase == "agent" and _is_model_request_budget_error(exc):
                state = "budget_exhausted"
            else:
                state = "agent_failure" if phase == "agent" else "integration_failure"
            failure_phase = phase
            error_type = type(exc).__name__
            error_message = redact_text(str(exc))
            _write_error(records_dir, exc)
    finally:
        terminal_cleanup = (
            terminal.finish_session
            if state in {"completed", "budget_exhausted"}
            else terminal.close
        )
        for cleanup_name, cleanup in (
            ("conversation", conversation.close if conversation is not None else None),
            ("terminal", terminal_cleanup),
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
        try:
            terminal.close()
        except BaseException as exc:
            cleanup_errors.append(
                f"terminal destroy: {type(exc).__name__}: {redact_text(str(exc))}"
            )
        cleanup_message = "; ".join(cleanup_errors)
        (records_dir / "cleanup-errors.txt").write_text(
            cleanup_message + "\n",
            encoding="utf-8",
        )
        if state in {"completed", "budget_exhausted"}:
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
            "codemode": codemode_enabled,
            "batching_guidance": batching_guidance_enabled,
            "request_observation": request_observation,
            "prefix_cache": history_enabled,
            "compaction_mode": compaction_mode,
            "compaction_threshold": compaction_threshold,
            "history_archive_threshold_tokens": history_archive_threshold_tokens,
            "history_archive_preserve_tokens": history_archive_preserve_tokens,
            "artifact_threshold_bytes": artifact_threshold_bytes,
            "artifact_quota_bytes": artifact_quota_bytes,
            "command_timeout_seconds": command_timeout_seconds,
            "terminal_preview_bytes": terminal_preview_bytes,
            "terminal_output_limit_bytes": terminal_output_limit_bytes,
            "retained_process_timeout_seconds": retained_process_timeout_seconds,
            "trial_timeout_seconds": recorded_trial_timeout,
            "trial_deadline_unix": (
                trial_deadline_unix if deadline_valid else repr(trial_deadline_unix)
            ),
            "finalization_reserve_seconds": finalization_reserve_seconds,
            "logfire": logfire_enabled,
            "request_limit": request_limit,
            "cumulative_input_tokens": model_cumulative_input_tokens,
        },
        request_observation=(
            {"enabled": False} if request_evidence is None else request_evidence.summary().to_dict()
        ),
        result_summary=result_summary,
        failure_phase=failure_phase,
        error_type=error_type,
        error_message=error_message,
    )
    atomic_write_json(result_path, record.to_dict())
    return record


def _configure_logfire() -> bool:
    if os.getenv("KEDI_HARBOR_LOGFIRE") != "1":
        return False
    import logfire
    from opentelemetry.instrumentation.kedi import KediInstrumentor

    logfire.configure(service_name="kedi", send_to_logfire="if-token-present")
    KediInstrumentor().instrument(
        runtime_detail="off",
        agent_enabled=True,
        artifacts_enabled=True,
        instrument_pydantic_ai=True,
        instrument_httpx=False,
        capture_content=False,
    )
    return True


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


def _is_model_request_budget_error(exc: BaseException) -> bool:
    return _find_wrapped_exception(exc, ModelRequestBudgetError) is not None


def _find_wrapped_exception(
    exc: BaseException,
    exception_type: type[BaseException],
) -> BaseException | None:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, exception_type):
            return current
        if isinstance(current, KediExecutionError):
            current = current.original
        else:
            current = current.__cause__
    return None


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
    parser.add_argument(
        "--codemode-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--batching-guidance-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--request-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--compaction-mode", choices=("disabled", "native"), default="disabled")
    parser.add_argument("--compaction-threshold", type=int)
    parser.add_argument("--history-archive-threshold-tokens", type=int)
    parser.add_argument("--history-archive-preserve-tokens", type=int, default=12_000)
    parser.add_argument("--artifact-threshold-bytes", type=int, default=16_384)
    parser.add_argument("--artifact-quota-bytes", type=int, default=256_000_000)
    parser.add_argument("--command-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--terminal-preview-bytes", type=int, default=32_768)
    parser.add_argument("--terminal-output-limit-bytes", type=int, default=512_000_000)
    parser.add_argument("--retained-process-timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--model-request-limit", type=int, default=512)
    parser.add_argument("--model-cumulative-input-tokens", type=int, default=50_000_000)
    parser.add_argument("--trial-timeout-seconds", type=float)
    parser.add_argument("--trial-deadline-unix", type=float)
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
        codemode_enabled=args.codemode_enabled,
        batching_guidance_enabled=args.batching_guidance_enabled,
        request_observation=args.request_observation,
        compaction_mode=args.compaction_mode,
        compaction_threshold=args.compaction_threshold,
        history_archive_threshold_tokens=args.history_archive_threshold_tokens,
        history_archive_preserve_tokens=args.history_archive_preserve_tokens,
        artifact_threshold_bytes=args.artifact_threshold_bytes,
        artifact_quota_bytes=args.artifact_quota_bytes,
        command_timeout_seconds=args.command_timeout_seconds,
        terminal_preview_bytes=args.terminal_preview_bytes,
        terminal_output_limit_bytes=args.terminal_output_limit_bytes,
        retained_process_timeout_seconds=args.retained_process_timeout_seconds,
        model_request_limit=args.model_request_limit,
        model_cumulative_input_tokens=args.model_cumulative_input_tokens,
        trial_timeout_seconds=args.trial_timeout_seconds,
        trial_deadline_unix=args.trial_deadline_unix,
    )
    return 0 if record.state in {"completed", "budget_exhausted"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["benchmark_approval", "main", "render_agent_source", "run_trial"]
