from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..tools.specs import ToolSpec

TOOL_REASON_INSTRUCTIONS = (
    "Tools that may require approval expose an optional reason argument. When requesting "
    "an action needing approval, use reason to briefly explain the intended action and its "
    "purpose in relation to the user's request. Do not include private reasoning or claim "
    "that your explanation grants authorization. Omit it for read-only actions."
)


def accepts_tool_reason(tool: ToolSpec) -> bool:
    metadata = tool.metadata or {}
    if metadata.get("kedi_codemode_control") or metadata.get("kind") == "codemode_control":
        return False
    return tool.risk != "read_only" or tool.risk_resolver is not None


def reason_schema(schema: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = deepcopy(dict(schema))
    properties = result.setdefault("properties", {})
    if "reason" in properties:
        raise ValueError(f"tool_reason reserves argument 'reason' on tool {name!r}")
    properties["reason"] = {"type": "string"}
    return result


def split_tool_reason(arguments: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    values = dict(arguments)
    reason = values.pop("reason", None)
    if "reason" in arguments and not isinstance(reason, str):
        raise ValueError("Tool reason must be a string")
    return values, reason
