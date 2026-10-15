from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .models import AgentRunRequest, SchemaOutput, TypedOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestDiagnostics:
    call_kind: str
    prompt_kind: str
    prompt_chars: int
    prompt_digest: str
    instruction_parts: int
    instruction_identities: tuple[str, ...]
    instruction_digest: str
    tool_count: int
    tool_names: tuple[str, ...]
    tool_contract_digest: str
    output_kind: str
    output_field_count: int
    output_contract_digest: str
    stateful: bool
    history_count: int
    cache_epoch: int | None
    lane_identity: str | None
    turn_sequence: int


def request_diagnostics(request: AgentRunRequest) -> RequestDiagnostics:
    output_kind = (
        "schema"
        if isinstance(request.output, SchemaOutput)
        else "typed"
        if isinstance(request.output, TypedOutput)
        else "text"
    )
    output_value = (
        request.output.schema
        if isinstance(request.output, SchemaOutput)
        else request.output.output_type
        if isinstance(request.output, TypedOutput)
        else "text"
    )
    conversation = request.conversation
    return RequestDiagnostics(
        call_kind=request.metadata.call_kind,
        prompt_kind=request.prompt.kind,
        prompt_chars=len(request.prompt.content),
        prompt_digest=_digest(request.prompt.content),
        instruction_parts=len(request.instructions),
        instruction_identities=tuple(part.identity for part in request.instructions),
        instruction_digest=_digest(
            tuple((part.identity, part.source, part.content) for part in request.instructions)
        ),
        tool_count=len(request.tool_definitions),
        tool_names=tuple(tool.name for tool in request.tool_definitions),
        tool_contract_digest=_digest(
            tuple(
                (
                    tool.name,
                    tool.description,
                    tool.input_schema,
                    tool.output_schema,
                    tool.risk,
                    tool.required_before_output,
                    tool.artifact_result,
                    tool.sequential,
                    tool.capability_owners,
                    tool.deferred_capability,
                )
                for tool in request.tool_definitions
            )
        ),
        output_kind=output_kind,
        output_field_count=(
            len(request.output.schema) if isinstance(request.output, SchemaOutput) else 0
        ),
        output_contract_digest=_digest(output_value),
        stateful=conversation is not None,
        history_count=0 if conversation is None else conversation.history_count,
        cache_epoch=None if conversation is None else conversation.cache_epoch,
        lane_identity=None if conversation is None else conversation.lane_identity,
        turn_sequence=request.metadata.turn_sequence,
    )


def _digest(value: Any) -> str:
    material = json.dumps(
        _diagnostic_value(value),
        default=repr,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(material).hexdigest()


def _diagnostic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _diagnostic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    return value


__all__ = ["RequestDiagnostics", "request_diagnostics"]
