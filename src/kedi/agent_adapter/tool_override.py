from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool


@runtime_checkable
class SupportsToolRegistration(Protocol):
    def register_tool(self, tool: Tool) -> None:
        """Register one tool for the active agent run scope."""
        ...

    def tool_scope(self, tools: Sequence[Tool] = ()) -> AbstractContextManager[None]:
        """Create a temporary tool registration scope for one agent run."""
        ...


class ToolScope:
    def __init__(self) -> None:
        self._scoped_tools: list[Tool] = []

    @contextmanager
    def tool_scope(self, tools: Sequence[Tool] = ()) -> Iterator[None]:
        previous = list(self._scoped_tools)
        self._scoped_tools = []
        for tool in tools:
            self.register_tool(tool)
        try:
            yield
        finally:
            self._scoped_tools = previous

    def register_tool(self, tool: Tool) -> None:
        name = getattr(tool, "name", None)
        if name is not None:
            self._scoped_tools = [
                registered
                for registered in self._scoped_tools
                if getattr(registered, "name", None) != name
            ]
        self._scoped_tools.append(tool)

    @property
    def scoped_tools(self) -> tuple[Tool, ...]:
        return tuple(self._scoped_tools)
