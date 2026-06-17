from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kedi.agent_adapter.tool_override import ToolScope


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_tool_scope_is_context_local_across_concurrent_tasks() -> None:
    """BUG-021: overlapping tool scopes on one ToolScope must stay disjoint."""
    scope = ToolScope()
    ready = asyncio.Event()

    async def run_scope(label: str, seen: dict[str, tuple[str, ...]]) -> None:
        with scope.tool_scope([_tool(label)]):
            # Let the other task enter its own scope before we observe ours,
            # so a shared instance list would leak across the two runs.
            if label == "a":
                ready.set()
            else:
                await ready.wait()
            await asyncio.sleep(0)
            scope.register_tool(_tool(f"{label}2"))
            await asyncio.sleep(0)
            seen[label] = tuple(tool.name for tool in scope.scoped_tools)

    async def main() -> dict[str, tuple[str, ...]]:
        seen: dict[str, tuple[str, ...]] = {}
        await asyncio.gather(run_scope("a", seen), run_scope("b", seen))
        return seen

    seen = asyncio.run(main())

    assert seen["a"] == ("a", "a2")
    assert seen["b"] == ("b", "b2")
    # After both runs finish the shared default scope is empty again.
    assert scope.scoped_tools == ()


def test_tool_scope_register_dedups_by_name_within_scope() -> None:
    scope = ToolScope()
    with scope.tool_scope():
        first = _tool("search")
        second = _tool("search")
        unnamed = SimpleNamespace()
        scope.register_tool(first)
        scope.register_tool(unnamed)
        scope.register_tool(second)
        assert scope.scoped_tools == (unnamed, second)
    assert scope.scoped_tools == ()


def test_tool_scope_restores_previous_scope_on_exit() -> None:
    scope = ToolScope()
    outer = _tool("outer")
    inner = _tool("inner")
    scope.register_tool(outer)
    with scope.tool_scope([inner]):
        assert scope.scoped_tools == (inner,)
    assert scope.scoped_tools == (outer,)
