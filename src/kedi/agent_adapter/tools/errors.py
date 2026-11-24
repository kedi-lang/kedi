"""Model-facing tool errors, independent of the framework's retry envelope."""

from __future__ import annotations

from ...errors import KediExecutionError
from .specs import ToolSpec


def correctable_tool_error(tool: ToolSpec, exc: Exception) -> str | None:
    from pydantic_ai.exceptions import RunCancelled

    from kedi.run_budget import RunBudgetExceeded

    original = exc.original if isinstance(exc, KediExecutionError) else exc
    if isinstance(exc, (RunBudgetExceeded, RunCancelled)) or isinstance(
        original, (RunBudgetExceeded, RunCancelled)
    ):
        return None
    record = getattr(exc, "tool_call", None)
    if getattr(exc, "kedi_result_failure_stage", None) is not None or (
        record is not None and getattr(record, "execution_state", None) == "returned"
    ):
        return None
    metadata = tool.metadata or {}
    kind = metadata.get("kedi_tool_kind") or metadata.get("kind")
    if kind == "codemode_control":
        from ..approvals.policy import ToolApprovalDenied
        from ..hooks.models import HookDeniedError

        if isinstance(exc, (ToolApprovalDenied, HookDeniedError)):
            return str(exc)
    if isinstance(original, CapabilityActivationError):
        return str(original)
    if kind == "filesystem" and isinstance(
        original, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
    ):
        return (
            f"Filesystem request rejected: {original}. Use repository-relative paths only, "
            "inspect unknown paths with get_file_info, use list_directory for directories, "
            "and read_text_file only for regular files."
        )
    if kind == "filesystem":
        return (
            f"Filesystem operation failed: {original}. Correct the arguments, inspect the "
            "target when needed, and retry the operation."
        )
    if kind == "terminal":
        return (
            f"Terminal request rejected: {original}. Keep working directories inside the "
            "assigned task root, correct invalid arguments or process IDs, and retry."
        )
    if kind == "dynamic_workflow":
        from ..subagents.workflow import WorkflowError

        if isinstance(original, WorkflowError):
            return str(original)
    if kind in {"subagent", "subagent_lifecycle"}:
        # Shared sanitized feedback; adapters distinguish argument corrections from execution
        # failures only after argument validation. Budgets and cancellation still end the run.
        from ..subagents.runs import safe_subagent_error_message

        message = safe_subagent_error_message(exc)
        if isinstance(exc, KediExecutionError):
            return message or "Subagent failed"
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if kind == "artifact_management":
        from kedi.artifacts.errors import artifact_tool_error_message

        return artifact_tool_error_message(original)
    if kind is None and isinstance(original, _DOMAIN_ERRORS):
        # An application tool (a Kedi procedure or Python tool) rejected its input or could not
        # reach its data, after its own retries: the calling model sees why and can correct the
        # arguments or continue without the result. The framework's tool retry limit and the
        # run budget still bound how often that happens. Other exceptions stay fatal.
        detail = str(original).strip()
        return (
            f"Tool {tool.name} failed: {type(original).__name__}"
            + (f": {detail[:600]}" if detail else "")
            + ". Correct the arguments if they caused this, or continue without this result."
        )
    return None


class TerminalToolFailure(RuntimeError):
    """Model-facing execution failure that must not consume argument corrections."""


def terminal_tool_failure(tool: ToolSpec, exc: Exception) -> TerminalToolFailure | None:
    """Classify a failure after validation, without changing direct tool execution."""
    metadata = tool.metadata or {}
    kind = metadata.get("kedi_tool_kind") or metadata.get("kind")
    if kind not in {"subagent", "subagent_lifecycle"}:
        return None
    original = exc.original if isinstance(exc, KediExecutionError) else exc
    if isinstance(original, CapabilityActivationError):
        return None
    message = correctable_tool_error(tool, exc)
    return TerminalToolFailure(message) if message is not None else None


# ValueError and LookupError for rejected input or missing records, OSError for unreachable
# data (timeouts, connections, files).
_DOMAIN_ERRORS = (ValueError, LookupError, OSError)


class ToolArgumentError(ValueError):
    """Tool call arguments that do not fit the tool's signature; the model can correct them."""


class RequiredToolCallError(RuntimeError):
    """Raised when an adapter returns before a required tool completes."""


class CapabilityActivationError(ValueError):
    """A model called a capability-owned target before its native activation."""
