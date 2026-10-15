from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from typing import Iterable

from ..tools.specs import ToolSpec
from .models import (
    AgentRunRequest,
    CallKind,
    ConversationCoordinates,
    InstructionPart,
    OutputContract,
    PromptDraft,
    RunMetadata,
    SchemaOutput,
    TextOutput,
    ToolDefinition,
    TypedOutput,
)


@dataclass(frozen=True, slots=True)
class RunRequestAssembler:
    """Build immutable adapter-neutral requests from already resolved inputs."""

    def assemble(
        self,
        *,
        prompt: PromptDraft,
        instructions: Iterable[InstructionPart],
        conversation: ConversationCoordinates | None,
        tool_specs: Iterable[ToolSpec],
        output: OutputContract,
        run_id: str,
        turn_sequence: int,
        call_kind: CallKind,
    ) -> AgentRunRequest:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if turn_sequence < 0:
            raise ValueError("turn_sequence must be non-negative")
        if call_kind not in {"invoke", "produce"}:
            raise ValueError(f"unsupported agent call kind {call_kind!r}")
        if not isinstance(output, (TextOutput, SchemaOutput, TypedOutput)):
            raise TypeError(f"unsupported Kedi output contract {type(output).__name__}")
        if isinstance(output, TextOutput) != (call_kind == "invoke"):
            raise ValueError("invoke requires text output; produce requires structured output")
        instruction_parts = tuple(instructions)
        instruction_identities = tuple(part.identity for part in instruction_parts)
        if len(set(instruction_identities)) != len(instruction_identities):
            raise ValueError("instruction identities must be unique within one request")
        return AgentRunRequest(
            prompt=prompt,
            instructions=instruction_parts,
            conversation=conversation,
            tool_definitions=tuple(_tool_definition(tool) for tool in tool_specs),
            output=output,
            metadata=RunMetadata(
                run_id=run_id,
                turn_sequence=turn_sequence,
                prompt_kind=prompt.kind,
                call_kind=call_kind,
            ),
        )


def conversation_lane_identity(
    adapter: object,
    profile: object,
    tools: Iterable[ToolSpec],
) -> str:
    """Fingerprint continuation inputs without exposing their values in the lane name."""

    default_model = getattr(adapter, "default_model", None)
    model = default_model() if callable(default_model) else None
    payload = {
        "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "shortname": getattr(adapter, "shortname", type(adapter).__name__),
        "default_model": model,
        "model": getattr(profile, "model", None),
        "effort": getattr(profile, "effort", None),
        "settings": dict(getattr(profile, "settings", {})),
        "mcp": [_mcp_definition_payload(server) for server in getattr(profile, "mcp_servers", ())],
        "tools": [_tool_definition_payload(_tool_definition(tool)) for tool in tools],
    }
    material = json.dumps(
        payload,
        default=repr,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = sha256(material).hexdigest()[:24]
    return f"{payload['shortname']}:{digest}"


def _tool_definition(tool: ToolSpec) -> ToolDefinition:
    metadata = tool.metadata or {}
    return ToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=tool.json_schema,
        output_schema=tool.return_schema,
        risk=tool.risk,
        required_before_output=metadata.get("required_before_output") is True,
        artifact_result=tool.artifact_result,
        sequential=tool.sequential,
        capability_owners=() if tool.capability_policy is None else tool.capability_policy.owners,
        deferred_capability=tool.deferred_capability,
    )


def _tool_definition_payload(tool: ToolDefinition) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input": _plain_value(tool.input_schema),
        "output": _plain_value(tool.output_schema),
        "risk": tool.risk,
        "required_before_output": tool.required_before_output,
        "artifact_result": tool.artifact_result,
        "sequential": tool.sequential,
        "capability_owners": tool.capability_owners,
        "deferred_capability": tool.deferred_capability,
    }


def _mcp_definition_payload(server: object) -> object:
    if is_dataclass(server) and not isinstance(server, type):
        return _plain_value(asdict(server))
    return repr(server)


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


__all__ = ["RunRequestAssembler", "conversation_lane_identity"]
