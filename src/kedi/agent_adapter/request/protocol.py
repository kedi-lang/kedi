"""Kedi-owned protocol parts for backends without native structured output or tool calling.

Kedi renders every model-visible task text. A backend that cannot take an output schema or
tool definitions natively gets them as instruction parts instead: ``protocol.output`` holds
the output schema, ``protocol.tools`` the tool catalog and the call format. The adapter
forwards them verbatim and parses the model's answer; it never writes this text itself.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from .models import InstructionPart, ToolDefinition, thaw_json
from .prompts import (
    PROTOCOL_FINAL_DATA,
    PROTOCOL_FINAL_TEXT,
    PROTOCOL_OUTPUT_INSTRUCTION,
    PROTOCOL_TOOLS_INSTRUCTION,
)

ProtocolPart: TypeAlias = Literal["output", "tools"]

OUTPUT_TOOL_NAME = "final_result"
"""The output tool a backend with native tool calling but no native output contract uses."""


def request_protocol(capabilities: Any) -> frozenset[ProtocolPart]:
    """The protocol parts an adapter's backend needs: those it declares it lacks natively."""

    parts: set[ProtocolPart] = set()
    if getattr(capabilities, "supports_native_structured_output", None) is False:
        parts.add("output")
    if getattr(capabilities, "supports_native_tool_calling", None) is False:
        parts.add("tools")
    return frozenset(parts)


def protocol_parts(
    protocol: Collection[str],
    *,
    output_json_schema: Mapping[str, Any] | None,
    tools: Sequence[ToolDefinition],
) -> tuple[InstructionPart, ...]:
    """The ``protocol.output`` and ``protocol.tools`` parts, in that order."""

    parts: list[InstructionPart] = []
    if "output" in protocol and output_json_schema is not None:
        parts.append(
            InstructionPart(
                identity="protocol.output",
                source="runtime",
                content=(
                    f"<OutputProtocol>\n{PROTOCOL_OUTPUT_INSTRUCTION}\n"
                    f"<OutputSchema>{_json(output_json_schema)}</OutputSchema>\n"
                    "</OutputProtocol>"
                ),
            )
        )
    if "tools" in protocol and tools:
        final = PROTOCOL_FINAL_TEXT if output_json_schema is None else PROTOCOL_FINAL_DATA
        parts.append(
            InstructionPart(
                identity="protocol.tools",
                source="runtime",
                content=(
                    f"<ToolProtocol>\n{PROTOCOL_TOOLS_INSTRUCTION} {final}\n"
                    f"{tool_catalog(tools)}\n</ToolProtocol>"
                ),
            )
        )
    return tuple(parts)


def tool_catalog(tools: Sequence[ToolDefinition]) -> str:
    """Each tool's name, Kedi description and schemas, unchanged."""

    entries = []
    for tool in tools:
        lines = [f'<Tool name="{tool.name}">']
        if tool.description:
            lines.append(f"<Description>{tool.description}</Description>")
        lines.append(f"<InputSchema>{_json(tool.parameters)}</InputSchema>")
        if tool.returns is not None:
            lines.append(f"<ReturnSchema>{_json(tool.returns)}</ReturnSchema>")
        lines.append("</Tool>")
        entries.append("\n".join(lines))
    return "<Tools>\n" + "\n".join(entries) + "\n</Tools>"


def _json(value: Any) -> str:
    return json.dumps(thaw_json(value), ensure_ascii=False, sort_keys=True)


__all__ = [
    "OUTPUT_TOOL_NAME",
    "ProtocolPart",
    "protocol_parts",
    "request_protocol",
    "tool_catalog",
]
