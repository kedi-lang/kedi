from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .errors import RequiredToolCallError
from .specs import ToolSpec


@dataclass(slots=True)
class RequiredToolCalls:
    """Track tools that must return before an adapter response can be accepted."""

    names: frozenset[str]
    attempted: set[str] = field(default_factory=set)
    returned: set[str] = field(default_factory=set)


_required_tool_calls: ContextVar[RequiredToolCalls | None] = ContextVar(
    "kedi_required_tool_calls",
    default=None,
)

_defer_artifact_admission: ContextVar[bool] = ContextVar(
    "kedi_defer_artifact_admission",
    default=False,
)

_artifact_admission_context: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "kedi_artifact_admission_context",
    default=None,
)


@contextmanager
def defer_artifact_admission() -> Iterator[None]:
    """Keep nested tool results raw until an owning composite tool returns."""

    token = _defer_artifact_admission.set(True)
    try:
        yield
    finally:
        _defer_artifact_admission.reset(token)


@contextmanager
def artifact_admission_scope(*, manager: Any, policy: Any) -> Iterator[None]:
    """Expose live artifact resources to adapter-native tool middleware."""

    token = _artifact_admission_context.set((manager, policy))
    try:
        yield
    finally:
        _artifact_admission_context.reset(token)


def current_artifact_admission() -> tuple[Any, Any] | None:
    """Return the active process-local artifact boundary, if one exists."""

    return _artifact_admission_context.get()


def current_required_tool_calls() -> RequiredToolCalls | None:
    """Return the required-tool state for the active adapter turn."""

    return _required_tool_calls.get()


@contextmanager
def required_tool_call_scope(
    tools: Sequence[ToolSpec],
    *,
    inherit: bool = False,
) -> Iterator[None]:
    """Track required tools in an isolated frame, optionally reusing an adapter-owned frame."""

    names = frozenset(
        tool.name for tool in tools if (tool.metadata or {}).get("required_before_output") is True
    )
    current = _required_tool_calls.get()
    if inherit and current is not None and current.names == names:
        yield
        return

    token = _required_tool_calls.set(RequiredToolCalls(names) if names else None)
    try:
        yield
    finally:
        _required_tool_calls.reset(token)


def validate_required_tool_calls() -> None:
    """Reject an adapter result if a required tool did not return successfully."""

    state = _required_tool_calls.get()
    if state is None or state.names <= state.returned:
        return
    missing_names = state.names - state.returned
    missing = ", ".join(sorted(missing_names))
    if missing_names & state.attempted:
        raise RequiredToolCallError(
            f"Required tool {missing} did not complete successfully; the adapter response "
            "cannot be accepted"
        )
    raise RequiredToolCallError(
        f"Required tool {missing} was not called; the adapter response cannot be accepted"
    )


class ToolScope:
    def __init__(self) -> None:
        self._scoped_tools: ContextVar[tuple[Any, ...]] = ContextVar(
            "kedi_scoped_tools", default=()
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[Any] = ()) -> Iterator[None]:
        token = self._scoped_tools.set(())
        try:
            for tool in tools:
                self.register_tool(tool)
            yield
        finally:
            self._scoped_tools.reset(token)

    def register_tool(self, tool: Any) -> None:
        current = self._scoped_tools.get()
        name = getattr(tool, "name", None)
        if name is not None:
            current = tuple(
                registered for registered in current if getattr(registered, "name", None) != name
            )
        self._scoped_tools.set((*current, tool))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._scoped_tools.get()
