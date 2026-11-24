"""Ancestor approval policies cap a child's Kedi tools on adapters without native approvals."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import pytest

from kedi.agent_adapter import AdapterCapabilities, AgentAdapterKind, ToolSpec
from kedi.lang import compile_program, parse_program


def program(parent_approval: str | None) -> str:
    parent = f"    > approval: {parent_approval}\n" if parent_approval else ""
    return (
        "@delete_file(path: str) -> str:\n"
        "  ###\n"
        "  Delete a file.\n"
        "  ###\n"
        "  > tool:\n"
        "    risk: mutating\n"
        "  = `'deleted ' + path`\n"
        "\n"
        "> profile: worker:\n"
        "    ###\n"
        "    Deletes files on request.\n"
        "    ###\n"
        "    > approval: allow\n"
        "    > use: delete_file\n"
        "\n"
        "> profile: coordinator:\n"
        f"{parent}"
        "    > subagent: worker\n"
        "\n"
        "> use: coordinator\n"
        "[answer] << Ask the worker to delete notes.txt.\n"
        "= <answer>\n"
    )


class SubagentAdapter:
    """A subagent-capable adapter that relies on Kedi's tool wrappers for approval."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "kedi-approval-probe"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_subagents=True,
        supports_artifacts=True,
    )

    def __init__(self) -> None:
        self.child_outcomes: list[str] = []
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"probe_tools_{id(self)}", default=()
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(tool for tool in tools if isinstance(tool, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))

    def telemetry_model_name(self) -> str | None:
        return None

    @contextmanager
    def subagent_scope(self, request: Any) -> Generator[None, None, None]:
        with self.tool_scope(request.tools):
            yield

    @staticmethod
    def subagent_run_kwargs(_limits: Any) -> dict[str, Any]:
        return {}

    async def _call(self, name: str, **arguments: Any) -> Any:
        tool = next(tool for tool in self._tools.get() if tool.name == name)
        result = tool.fn(**arguments)
        return await result if inspect.isawaitable(result) else result

    async def invoke(self, *, prompt: str, **_kwargs: Any) -> str:
        names = {tool.name for tool in self._tools.get()}
        if "delegate_task" in names:
            result = await self._call("delegate_task", subagent="worker", task="Delete notes.txt.")
            return json.dumps(result, default=str)
        try:
            outcome = await self._call("delete_file", path="notes.txt")
        except Exception as exc:  # the child reports the failure as its result
            outcome = type(exc).__name__
        self.child_outcomes.append(outcome)
        return outcome

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        return asyncio.run(self.invoke(prompt=prompt, **kwargs))

    async def produce_subagent(self, **_kwargs: Any) -> Any:
        raise AssertionError("no structured child result is requested")

    async def produce(self, **_kwargs: Any) -> Any:
        raise AssertionError("no structured output is requested")

    def produce_sync(self, **_kwargs: Any) -> Any:
        raise AssertionError("no structured output is requested")


@pytest.mark.parametrize(
    ("parent_approval", "expected"),
    [
        ("deny", "ToolApprovalDenied"),
        (None, "ToolApprovalDenied"),
        ("allow", "deleted notes.txt"),
    ],
)
def test_parent_policy_caps_child_tools(parent_approval: str | None, expected: str) -> None:
    adapter = SubagentAdapter()
    runtime = compile_program(parse_program(program(parent_approval)), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())

    assert adapter.child_outcomes == [expected]
