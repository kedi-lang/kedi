"""Model-facing schemas for delegation tools, independent of runtime state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator

from kedi.agent_adapter.tools.specs import ToolSpec


def build_subagent_tool_specs(
    *,
    allowed: tuple[str, ...],
    catalog: Mapping[str, str | None],
    supports_background: bool,
    delegate_task: Callable[..., Any],
    continue_subagent: Callable[..., Any],
    wait_subagent: Callable[..., Any],
    cancel_subagent: Callable[..., Any],
    subagent_status: Callable[..., Any],
    control_tools: tuple[str, ...] | None = None,
) -> tuple[ToolSpec, ...]:
    """``control_tools`` names the tools offered besides ``delegate_task`` (``None``: all).

    Background delegation is offered only with ``wait_subagent``, so its result can be awaited.
    """
    offered = (lambda name: True) if control_tools is None else (lambda name: name in control_tools)
    supports_background = supports_background and offered("wait_subagent")
    entries = [
        f"- {name}: {description}" if description else f"- {name}"
        for name, description in catalog.items()
    ]
    descriptions = "\n\nAvailable subagents:\n" + "\n".join(entries)
    preamble = (
        "Delegate one self-contained task to one direct subagent listed below. "
        "Choose the subagent by its description and include all context it needs; "
        "the child does not inherit the parent conversation. Omit final_schema "
        "to use the child's configured output type; supply it only to override "
        "or add a structured result schema."
        + (
            " Set background=true only for work that can run concurrently, then call "
            "wait_subagent before using its result or returning a final answer."
            if supports_background
            else ""
        )
    )
    properties: dict[str, Any] = {
        "subagent": {
            "type": "string",
            "enum": list(allowed),
            "description": "Configured child profile that should perform the task.",
        },
        "task": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Self-contained task with all context and acceptance criteria needed by the child."
            ),
        },
        "final_schema": {
            "anyOf": [{"type": "object"}, {"type": "null"}],
            "default": None,
            "description": (
                "Optional JSON Schema for the child result. Use standard lowercase "
                "type values such as object, array, string, integer, number, boolean, "
                "and null. Pass a JSON object, not a JSON-encoded string. Omit this "
                "field to use the child profile's configured output type."
            ),
        },
    }
    if supports_background:
        properties["background"] = {
            "type": "boolean",
            "default": False,
            "description": (
                "Start without blocking. Call wait_subagent with the returned run_id "
                "before using the result or returning a final answer."
            ),
        }

    continuation_properties: dict[str, Any] = {
        "run_id": {
            "type": "string",
            "description": "Completed run whose child conversation should continue.",
        },
        "task": {
            "type": "string",
            "minLength": 1,
            "description": "New follow-up task for the same child conversation.",
        },
        "final_schema": properties["final_schema"],
    }
    if supports_background:
        continuation_properties["background"] = properties["background"]

    tools = [
        ToolSpec(
            fn=delegate_task,
            name="delegate_task",
            description=preamble + descriptions,
            json_schema={
                "type": "object",
                "properties": properties,
                "required": ["subagent", "task"],
                "additionalProperties": False,
            },
            return_schema=(
                {
                    "oneOf": [
                        _delegation_result_schema(),
                        _subagent_handle_schema(),
                    ]
                }
                if supports_background
                else _delegation_result_schema()
            ),
            metadata={
                "source": "kedi",
                "kind": "subagent",
                "kedi_subagent_catalog": dict(catalog),
                "kedi_subagent_preamble": preamble,
            },
            # Child tools enforce the child profile's own approval policy. Spawning
            # an isolated child is not itself a filesystem or environment mutation.
            risk="read_only",
        ),
        ToolSpec(
            fn=continue_subagent,
            name="continue_subagent",
            description=(
                "Continue the latest completed turn of a child conversation. The same child "
                "profile receives its bounded verified history; the continuation consumes a "
                "new delegation budget slot and returns a new run_id."
            ),
            json_schema={
                "type": "object",
                "properties": continuation_properties,
                "required": ["run_id", "task"],
                "additionalProperties": False,
            },
            return_schema=(
                {
                    "oneOf": [
                        _delegation_result_schema(),
                        _subagent_handle_schema(),
                    ]
                }
                if supports_background
                else _delegation_result_schema()
            ),
            metadata={"source": "kedi", "kind": "subagent"},
            risk="read_only",
        ),
    ]
    if not supports_background:
        return schema_validated(_offered(tools, offered))

    tools.extend(
        (
            ToolSpec(
                fn=wait_subagent,
                name="wait_subagent",
                description=(
                    "Wait for a background subagent and return its task_summary and "
                    "final_result. When a caller timeout expires while the child is still "
                    "running, return its running handle instead of failing the parent turn; "
                    "call wait_subagent again. Completed runs may be waited more than once."
                ),
                json_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "timeout": {
                            "anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}],
                            "default": None,
                            "description": "Maximum wait in seconds, not milliseconds.",
                        },
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
                return_schema={
                    "oneOf": [
                        _delegation_result_schema(),
                        _subagent_handle_schema(),
                    ]
                },
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
            ToolSpec(
                fn=cancel_subagent,
                name="cancel_subagent",
                description=(
                    "Request cancellation of a background subagent. Cancellation is "
                    "idempotent; call wait_subagent to observe terminal completion."
                ),
                json_schema=_run_id_schema(),
                return_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "cancel_requested": {"type": "boolean"},
                        "status": {"type": "string"},
                    },
                    "required": ["run_id", "cancel_requested", "status"],
                    "additionalProperties": False,
                },
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
            ToolSpec(
                fn=subagent_status,
                name="subagent_status",
                description=(
                    "Inspect a background subagent's lifecycle status without waiting for it."
                ),
                json_schema=_run_id_schema(),
                return_schema=_subagent_handle_schema(),
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
        )
    )
    return schema_validated(_offered(tools, offered))


def _offered(tools: list[ToolSpec], offered: Callable[[str], bool]) -> list[ToolSpec]:
    return [tool for tool in tools if tool.name == "delegate_task" or offered(tool.name)]


def _delegation_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "subagent": {"type": "string"},
            "task_summary": {"type": "string"},
            "final_result": {},
        },
        "required": ["run_id", "subagent", "task_summary", "final_result"],
        "additionalProperties": False,
    }


def _subagent_handle_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "subagent": {"type": "string"},
            "status": {"type": "string"},
            "input_available": {"type": "boolean"},
        },
        "required": ["run_id", "subagent", "status"],
        "additionalProperties": False,
    }


def _run_id_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
        "additionalProperties": False,
    }


def schema_validated(specs: list[ToolSpec] | tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
    """Enforce each tool's advertised JSON Schema so a malformed call is reported, not raised."""
    return tuple(
        replace(spec, argument_validator=_schema_argument_validator(spec)) for spec in specs
    )


def _schema_argument_validator(spec: ToolSpec) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    schema = spec.json_schema or {"type": "object"}
    validator = Draft202012Validator(schema)
    allowed = ", ".join(sorted(schema.get("properties", {}))) or "none"

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(arguments)
        problems = [
            f"{'.'.join(str(part) for part in error.absolute_path) or 'arguments'}: {error.message}"
            for error in sorted(validator.iter_errors(values), key=lambda error: error.path)
        ]
        if problems:
            raise ValueError(
                f"Invalid arguments for {spec.name}: {'; '.join(problems)}. Allowed arguments: {allowed}."
            )
        return values

    return validate
