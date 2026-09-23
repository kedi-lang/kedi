"""Kedi tool corrections, independent of the framework's retry envelope."""

from __future__ import annotations

from ...errors import KediExecutionError
from .specs import ToolSpec


def correctable_tool_error(tool: ToolSpec, exc: Exception) -> str | None:
    from kedi.run_budget import RunBudgetExceeded

    if isinstance(exc, RunBudgetExceeded):
        return None
    record = getattr(exc, "tool_call", None)
    if getattr(exc, "kedi_result_failure_stage", None) is not None or (
        record is not None and getattr(record, "execution_state", None) == "returned"
    ):
        return None
    metadata = tool.metadata or {}
    kind = metadata.get("kedi_tool_kind") or metadata.get("kind")
    original = exc.original if isinstance(exc, KediExecutionError) else exc
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
    if kind == "subagent" and isinstance(exc, KediExecutionError):
        from ..subagents.runs import safe_subagent_error_message

        return safe_subagent_error_message(exc) or "Subagent failed"
    if kind == "artifact_management":
        from kedi.artifacts.errors import artifact_tool_error_message

        return artifact_tool_error_message(original)
    return None


class RequiredToolCallError(RuntimeError):
    """Raised when an adapter returns before a required tool completes."""
