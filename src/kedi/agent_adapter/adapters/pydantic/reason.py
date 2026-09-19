from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, RawToolArgs
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition

from kedi.agent_adapter.approvals.reason import (
    TOOL_REASON_INSTRUCTIONS,
    accepts_tool_reason,
    reason_schema,
    split_tool_reason,
)

if TYPE_CHECKING:
    from .adapter import PydanticAdapter


class ToolReasonCapability(AbstractCapability[Any]):
    def __init__(self, adapter: PydanticAdapter[Any, Any]) -> None:
        self.adapter = adapter

    def get_instructions(self) -> str:
        return TOOL_REASON_INSTRUCTIONS

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        result = []
        for definition in tool_defs:
            tool, _ = self.adapter._hook_tool_for_call(definition.name, definition)
            if accepts_tool_reason(tool):
                definition = replace(
                    definition,
                    parameters_json_schema=reason_schema(
                        definition.parameters_json_schema, name=definition.name
                    ),
                    # Strict providers otherwise turn optional arguments into required ones.
                    strict=False,
                    metadata={**(definition.metadata or {}), "kedi_tool_reason": True},
                )
            result.append(definition)
        return result

    async def before_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        if not (tool_def.metadata or {}).get("kedi_tool_reason"):
            return args
        try:
            values = json.loads(args) if isinstance(args, str) else args
            if not isinstance(values, dict):
                return args
            values, reason = split_tool_reason(values)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        if not ctx.tool_call_approved:
            self.adapter._tool_reasons_var.get()[call.tool_call_id] = reason
        return values

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if any(
            tool.kind == "mcp_server"
            for tool in request_context.model_request_parameters.native_tools
        ):
            raise UserError(
                "tool_reason requires local MCP toolsets; provider-native MCP bypasses Kedi approval"
            )
        return request_context
