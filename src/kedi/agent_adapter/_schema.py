from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

from ..dynamic_models import build_dynamic_model
from ..resolved_types import ResolvedKediType


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def output_field_metadata(type_: ResolvedKediType) -> tuple[Any, str | None]:
    if get_origin(type_) is not Annotated:
        return type_, None

    annotation, first_metadata, *_ = get_args(type_)
    description = first_metadata if isinstance(first_metadata, str) else None
    return annotation, description


def output_field_definition(
    type_: ResolvedKediType,
) -> tuple[Any, Any]:
    annotation, description = output_field_metadata(type_)
    if description is not None:
        return annotation, Field(description=description)
    return annotation, ...


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
    fields = {field: output_field_definition(type_) for field, type_ in output_schema.items()}
    return build_dynamic_model("_OutputModel", _PortableOutputModel, fields)
