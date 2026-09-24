"""Compile a task template into one child request and its result contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

from kedi.agent_adapter.output_schema import (
    _portable_output_json_schema,
    output_field_definition,
)
from kedi.agent_adapter.subagents import validate_final_schema
from kedi.dynamic_models import build_dynamic_model
from kedi.errors import KediExecutionError
from kedi.lang.ast import OutputSegment, TaskStmt, TextSegment

from .evaluation import _stringify_segments, merge_template_block_lines
from .type_system import KediType


class _TaskOutputBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra=_portable_output_json_schema,
    )


def prepare_subagent_task(
    rt: Any, stmt: TaskStmt, env: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, TypeAdapter[Any] | None]:
    merged = merge_template_block_lines(stmt.template.lines, loc=stmt.template.loc)
    prompt_segments = []
    field_types: dict[str, Any] = {}
    for segment in merged.segments:
        if not isinstance(segment, OutputSegment):
            prompt_segments.append(segment)
            continue
        contract = KediType.from_node(segment.type_expr, segment.type_annotation)
        resolved = contract.resolve_required(rt, env) or str
        field_types[segment.name] = resolved
        marker = (
            f"[{segment.name}: {segment.type_annotation}]"
            if segment.type_annotation
            else f"[{segment.name}]"
        )
        prompt_segments.append(TextSegment(text=marker, loc=segment.loc))

    prompt = _stringify_segments(rt, prompt_segments, env, context="task")
    schema = None
    output_adapter = None
    if field_types:
        try:
            model: type[BaseModel] = build_dynamic_model(
                "SubagentTaskOutput",
                _TaskOutputBase,
                {
                    name: output_field_definition(field_type)
                    for name, field_type in field_types.items()
                },
            )
            output_adapter = TypeAdapter(model)
            schema = _inline_pydantic_refs(output_adapter.json_schema())
            validate_final_schema(schema)
        except Exception as exc:
            raise KediExecutionError(
                f"Task captures cannot form a valid result schema: {exc}"
            ) from exc
    return prompt, schema, output_adapter


def _inline_pydantic_refs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.get("$defs", {})

    def expand(value: Any, active: frozenset[str]) -> Any:
        if isinstance(value, list):
            return [expand(item, active) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference is not None:
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise ValueError(f"Unsupported task capture schema reference: {reference!r}")
            name = reference[len(prefix) :]
            if name in active:
                raise ValueError(f"Recursive task capture type {name!r} is not supported")
            target = definitions.get(name)
            if not isinstance(target, dict):
                raise ValueError(f"Unknown task capture schema reference: {reference!r}")
            resolved = expand(target, active | {name})
            return {
                **resolved,
                **{key: expand(item, active) for key, item in value.items() if key != "$ref"},
            }
        return {key: expand(item, active) for key, item in value.items() if key != "$defs"}

    return expand(schema, frozenset())
