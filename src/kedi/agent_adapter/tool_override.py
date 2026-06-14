from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import ProcedureDef


@runtime_checkable
class SupportsToolRegistration(Protocol):
    def register_kedi_tool(self, rt: KediRuntime, proc: ProcedureDef) -> None:
        """Register a Kedi procedure as an adapter-native tool."""
        ...

    def register_python_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a Python callable as an adapter-native tool."""
        ...

    def register_tool(self, tool: Any) -> None:
        """Register one adapter-native tool for the active agent run scope."""
        ...

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        """Create a temporary tool registration scope for one agent run."""
        ...


class ToolScope:
    def __init__(self) -> None:
        self._scoped_tools: list[Any] = []

    @contextmanager
    def tool_scope(self, tools: Sequence[Any] = ()) -> Iterator[None]:
        previous = list(self._scoped_tools)
        self._scoped_tools = []
        for tool in tools:
            self.register_tool(tool)
        try:
            yield
        finally:
            self._scoped_tools = previous

    def register_tool(self, tool: Any) -> None:
        name = getattr(tool, "name", None)
        if name is not None:
            self._scoped_tools = [
                registered
                for registered in self._scoped_tools
                if getattr(registered, "name", None) != name
            ]
        self._scoped_tools.append(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return tuple(self._scoped_tools)
