from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kedi.agent_adapter.approvals.reason import accepts_tool_reason, reason_schema
from kedi.agent_adapter.tools.specs import ToolSpec

from .tools import _langchain_tool_spec_from_native


def tools_with_reason(tools: Sequence[Any], specs: Sequence[ToolSpec]) -> list[Any]:
    known = {tool.name: tool for tool in specs}
    result = []
    for tool in tools:
        spec = known.get(tool.name) or _langchain_tool_spec_from_native(tool)
        if accepts_tool_reason(spec):
            schema = (
                tool.get_input_schema().model_json_schema()
                if not isinstance(tool.args_schema, dict)
                else tool.args_schema
            )
            tool = tool.model_copy(
                update={
                    "args_schema": reason_schema(schema, name=tool.name),
                    "metadata": {**(tool.metadata or {}), "kedi_tool_reason": True},
                }
            )
        result.append(tool)
    return result
