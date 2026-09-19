from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..approvals.policy import ApprovalPolicy
from ..approvals.reason import accepts_tool_reason, reason_schema, split_tool_reason
from ..tools.execution import resolve_tool_arguments_async
from ..tools.scope import defer_artifact_admission
from ..tools.specs import ToolResultAdmitter, ToolSpec
from .core import CODEMODE_INSTRUCTIONS, CodeModeLimits, CodeModeSession, CodeModeTool

CONTROL_TOOL_NAMES = frozenset({"search_tools", "get_tool_schema", "execute_code"})

SEARCH_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Optional text used to filter tool names.",
        },
        "limit": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
            "default": None,
            "description": "Maximum names to return.",
        },
        "cursor": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Opaque cursor returned by an earlier search_tools call.",
        },
    },
    "additionalProperties": False,
}

GET_TOOL_SCHEMA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact tool names returned by search_tools.",
        }
    },
    "required": ["tool_names"],
    "additionalProperties": False,
}

EXECUTE_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Monty-compatible Python code to execute.",
        },
        "restart": {
            "type": "boolean",
            "default": False,
            "description": "Reset the per-run sandbox state before executing this code.",
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class CodeModeProjection:
    """Model-facing control tools and their run-scoped execution owner."""

    tools: tuple[Any, ...]
    bridge: CodeModeToolSpecBridge | None = None

    @property
    def enabled(self) -> bool:
        return self.bridge is not None

    async def close(self) -> None:
        if self.bridge is not None:
            await self.bridge.close()


class CodeModeToolSpecBridge:
    """Project Kedi ToolSpecs through one run-scoped CodeMode session."""

    def __init__(
        self,
        tools: Sequence[ToolSpec],
        *,
        limits: CodeModeLimits,
        policies: Sequence[ApprovalPolicy | None],
        adapter_shortname: str,
        tool_reason: bool = False,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            duplicates = _duplicate_tool_names(tools)
            raise ValueError("Duplicate CodeMode tool names: " + ", ".join(duplicates))
        self._policies = tuple(policies)
        self._adapter_shortname = adapter_shortname
        self._tool_reason = tool_reason
        self._session = CodeModeSession(limits)
        self._session.update_tools(
            [_codemode_tool(tool, tool_reason=tool_reason) for tool in tools]
        )
        self._result_admitter = shared_result_admitter(tools)

    @property
    def control_tools(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                fn=self.search_tools,
                name="search_tools",
                description=(
                    "Browse the CodeMode catalog. Returns only exact tool names and an opaque "
                    "cursor; call get_tool_schema before using a selected tool."
                ),
                json_schema=SEARCH_TOOLS_SCHEMA,
                metadata={"kind": "codemode_control"},
                risk="read_only",
                artifact_result="never",
                sequential=True,
            ),
            ToolSpec(
                fn=self.get_tool_schema,
                name="get_tool_schema",
                description=(
                    "Return exact schemas for selected tool names and make those tools callable "
                    "inside execute_code."
                ),
                json_schema=GET_TOOL_SCHEMA_SCHEMA,
                metadata={"kind": "codemode_control"},
                risk="read_only",
                artifact_result="never",
                sequential=True,
            ),
            ToolSpec(
                fn=self.execute_code,
                name="execute_code",
                description=(
                    "Execute one bounded Monty-compatible notebook cell using only tools previously "
                    "hydrated with get_tool_schema. Successful variables persist across calls; "
                    "continue from them instead of repeating successful application tool calls. "
                    "Failures return captured output and the error in separate fields."
                ),
                json_schema=EXECUTE_CODE_SCHEMA,
                metadata={
                    "kind": "codemode_control",
                    "code_arg_name": "code",
                    "code_arg_language": "python",
                },
                risk="read_only",
                artifact_result="never",
                sequential=True,
            ),
        )

    def search_tools(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._session.search_tools(query=query, limit=limit, cursor=cursor)

    def get_tool_schema(self, *, tool_names: list[str]) -> dict[str, Any]:
        return self._session.get_tool_schema(tool_names=tool_names)

    async def execute_code(self, *, code: str, restart: bool = False) -> Any:
        result = await self._session.execute_code(
            code=code,
            restart=restart,
            invoke=self._invoke,
        )
        if self._result_admitter is not None:
            result = await self._result_admitter(result, "execute_code")
        return result

    async def close(self) -> None:
        await self._session.close()

    async def _invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools[tool_name]
        final_arguments = arguments
        reason = None
        if self._tool_reason and accepts_tool_reason(tool):
            final_arguments, reason = split_tool_reason(arguments)
        for policy in self._policies:
            final_arguments = await resolve_tool_arguments_async(
                tool,
                final_arguments,
                policy=policy,
                adapter_shortname=self._adapter_shortname,
                reason=reason,
                tool_reason_enabled=self._tool_reason,
            )
        with defer_artifact_admission():
            result = tool.fn(**final_arguments)
            if inspect.isawaitable(result):
                result = await result
        return result


def codemode_projection(
    tools: Sequence[ToolSpec],
    *,
    limits: CodeModeLimits | None,
    policies: Sequence[ApprovalPolicy | None],
    adapter_shortname: str,
    tool_reason: bool = False,
) -> CodeModeProjection:
    if limits is None:
        return CodeModeProjection(tools=tuple(tools))
    bridge = CodeModeToolSpecBridge(
        tools,
        limits=limits,
        policies=policies,
        adapter_shortname=adapter_shortname,
        tool_reason=tool_reason,
    )
    return CodeModeProjection(tools=bridge.control_tools, bridge=bridge)


def append_codemode_instructions(instructions: str | None, *, enabled: bool) -> str | None:
    if not enabled:
        return instructions
    if not instructions:
        return CODEMODE_INSTRUCTIONS
    return f"{instructions.rstrip()}\n\n{CODEMODE_INSTRUCTIONS}"


def shared_result_admitter(tools: Sequence[ToolSpec]) -> ToolResultAdmitter | None:
    admitters: dict[object, ToolResultAdmitter] = {}
    for tool in tools:
        admitter = tool.result_admitter
        if admitter is None:
            continue
        boundary_key = getattr(admitter, "_kedi_artifact_boundary_key", id(admitter))
        admitters[boundary_key] = admitter
    if len(admitters) > 1:
        raise ValueError("CodeMode tools belong to multiple artifact result boundaries")
    return next(iter(admitters.values()), None)


def approval_policies(
    local_policy: ApprovalPolicy | None,
    ceilings: Sequence[ApprovalPolicy | None],
) -> tuple[ApprovalPolicy | None, ...]:
    policies = tuple(ceilings)
    if local_policy is not None or not policies:
        return (*policies, local_policy)
    return policies


def _codemode_tool(tool: ToolSpec, *, tool_reason: bool = False) -> CodeModeTool:
    if tool.name in CONTROL_TOOL_NAMES:
        raise ValueError(f"CodeMode reserves tool name {tool.name!r}")
    schema = tool.json_schema or {"type": "object", "properties": {}}
    if tool_reason and accepts_tool_reason(tool):
        schema = reason_schema(schema, name=tool.name)
    return CodeModeTool(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=schema,
        return_schema=tool.return_schema,
        metadata=tool.metadata,
        sequential=tool.sequential,
    )


def _duplicate_tool_names(tools: Sequence[ToolSpec]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            duplicates.add(tool.name)
        seen.add(tool.name)
    return sorted(duplicates)


__all__ = [
    "CONTROL_TOOL_NAMES",
    "CodeModeProjection",
    "CodeModeToolSpecBridge",
    "append_codemode_instructions",
    "approval_policies",
    "codemode_projection",
    "shared_result_admitter",
]
