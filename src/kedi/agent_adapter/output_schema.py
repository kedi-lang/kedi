from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..dynamic_models import build_dynamic_model
from ..resolved_types import ResolvedKediType, annotated_field_metadata


class OutputCorrection(ValueError):
    """Structured output the model can correct; the message is the feedback it receives."""


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def output_field_metadata(type_: ResolvedKediType) -> tuple[Any, str | None]:
    return annotated_field_metadata(type_)


def output_field_definition(
    type_: ResolvedKediType,
) -> tuple[Any, Any]:
    annotation, description = output_field_metadata(type_)
    if description is not None:
        return annotation, Field(description=description)
    return annotation, ...


def output_model_fields(
    output_schema: Mapping[str, ResolvedKediType],
) -> dict[str, tuple[Any, Any]]:
    """Field definitions for an output model, keyed so any Kedi output name works.

    Pydantic hides underscore names and refuses or shadows ``BaseModel`` members, so such an
    output gets an internal field name and keeps its Kedi name as the schema alias.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    for name, type_ in output_schema.items():
        if not name.startswith(("_", "model_")) and not hasattr(BaseModel, name):
            fields[name] = output_field_definition(type_)
            continue
        annotation, description = output_field_metadata(type_)
        field_name = f"kedi_output_{len(fields)}"
        while field_name in output_schema:
            field_name += "_"
        fields[field_name] = (annotation, Field(alias=name, description=description))
    return fields


def output_value(result: Any, name: str, default: Any) -> Any:
    """Read one output by its Kedi name, following the aliases of ``output_model_fields``."""
    if isinstance(result, BaseModel):
        for field_name, info in type(result).model_fields.items():
            if info.alias == name:
                return getattr(result, field_name)
    return getattr(result, name, default)


def _portable_output_json_schema(schema: dict[str, Any]) -> None:
    """Replace fixed-tuple JSON Schema with the portable array subset used by LLM APIs."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            prefix_items = node.pop("prefixItems", None)
            if isinstance(prefix_items, list) and "items" not in node:
                if prefix_items and all(item == prefix_items[0] for item in prefix_items):
                    node["items"] = prefix_items[0]
                elif prefix_items:
                    node["items"] = {"anyOf": prefix_items}
                else:
                    node["items"] = {}
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)


class _PortableOutputModel(BaseModel):
    model_config = ConfigDict(json_schema_extra=_portable_output_json_schema)


def build_output_model(
    output_schema: dict[str, ResolvedKediType],
) -> type[BaseModel]:
    if not output_schema:
        raise ValueError("Output schema cannot be empty.")
    from .request.context import current_agent_run_request

    request = current_agent_run_request()
    description = (
        "Return only the replacement values for the corresponding bracketed template "
        "fields, excluding fixed surrounding text. Preserve full sentences when a field "
        "requests them."
        if request is not None
        and request.prompt.kind == "template"
        and any(part.identity == "template.instructions" for part in request.instructions)
        else None
    )
    return build_dynamic_model(
        "_OutputModel",
        _PortableOutputModel,
        output_model_fields(output_schema),
        __doc__=description,
    )
