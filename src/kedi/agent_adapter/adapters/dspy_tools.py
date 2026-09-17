from __future__ import annotations

import copy
import inspect
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    cast,
)

from kedi.agent_adapter.execution import run_sync
from kedi.agent_adapter.invocation import InvocationUsage, report_invocation

_DSPY_SYNC_TOOL_MARKER = "__kedi_dspy_sync_tool__"


def dspy_sync_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a sync or async callable to DSPy's sync tool-call path."""
    if getattr(fn, _DSPY_SYNC_TOOL_MARKER, False):
        return fn

    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else fn.__call__

    @wraps(target)
    def _sync_tool(*args: Any, **kwargs: Any) -> Any:
        report_invocation(InvocationUsage(tool_calls=1))
        result = target(*args, **kwargs)
        if inspect.isawaitable(result):
            return run_sync(cast(Awaitable[Any], result))
        return result

    setattr(_sync_tool, _DSPY_SYNC_TOOL_MARKER, True)
    return _sync_tool


def dspy_tool_args(json_schema: dict[str, Any]) -> dict[str, Any]:
    defs = json_schema.get("$defs", {})
    return {
        name: _resolve_dspy_arg_schema(schema, defs)
        for name, schema in json_schema.get("properties", {}).items()
    }


def dspy_tool_arg_desc(json_schema: dict[str, Any]) -> dict[str, str]:
    properties = json_schema.get("properties", {})
    return {
        name: description
        for name, schema in properties.items()
        if isinstance(schema, dict) and isinstance((description := schema.get("description")), str)
    }


def _resolve_dspy_arg_schema(
    schema: dict[str, Any],
    defs: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(schema)
    ref = resolved.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = copy.deepcopy(defs[ref.removeprefix("#/$defs/")])
    return _strip_json_schema_titles(resolved)


def _strip_json_schema_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_json_schema_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_strip_json_schema_titles(item) for item in value]
    return value
