from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from kedi.lang.ast import ProcedureDef


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., Any]
    name: str
    description: str | None = None
    json_schema: dict[str, Any] | None = None
    return_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    proc: ProcedureDef | None = None


@runtime_checkable
class SupportsToolRegistration(Protocol):
    def register_tool(self, tool: ToolSpec) -> None:
        """Register one adapter-native tool for the active agent run scope."""
        ...

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        """Create a temporary tool registration scope for one agent run."""
        ...


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
