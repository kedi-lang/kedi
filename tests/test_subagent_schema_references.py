from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Literal

import pytest
from jsonschema import ValidationError
from pydantic import BaseModel, TypeAdapter

from kedi.agent_adapter.subagents import (
    effective_final_schema,
    validate_structured_result,
    wrapped_final_schema,
)
from kedi.agent_adapter.subagents.schema import (
    MAX_FINAL_SCHEMA_BYTES,
    normalize_final_schema,
    validate_final_schema,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.lang import compile_program, parse_program


class Person(BaseModel):
    name: str
    role: Literal["owner", "reviewer"]


class Assignment(BaseModel):
    owner: Person
    reviewers: list[Person]
    backup: Person | None


WIRE_ASSIGNMENT = {
    "owner": {"name": "Alex", "role": "owner"},
    "reviewers": [{"name": "Robin", "role": "reviewer"}],
    "backup": None,
}


def test_pydantic_defs_are_inlined_without_mutating_or_aliasing_input():
    schema = TypeAdapter(Assignment).json_schema()
    before = deepcopy(schema)
    assert "$defs" in schema
    normalized = normalize_final_schema(schema)
    assert schema == before
    assert "$defs" not in normalized
    assert normalized["properties"]["owner"]["properties"]["role"]["enum"] == ["owner", "reviewer"]
    assert normalized["properties"]["reviewers"]["items"]["type"] == "object"
    assert normalized["properties"]["backup"]["anyOf"][0]["type"] == "object"
    normalized["properties"]["owner"]["properties"]["name"]["type"] = "integer"
    assert normalized["properties"]["reviewers"]["items"]["properties"]["name"]["type"] == "string"
    assert schema == before


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_and_profile_schemas_share_normalization_and_validation(explicit):
    schema = effective_final_schema(
        AgentProfile(name="worker", output_type=Assignment),
        TypeAdapter(Assignment).json_schema() if explicit else None,
    )
    validate_final_schema(schema)
    wrapped = wrapped_final_schema(schema)
    envelope = {"task_summary": "Assigned the review.", "final_result": WIRE_ASSIGNMENT}
    assert validate_structured_result(envelope, wrapped) == envelope
    bad = deepcopy(envelope)
    bad["final_result"]["reviewers"][0]["role"] = "invented-role"
    with pytest.raises(ValidationError):
        validate_structured_result(bad, wrapped)


@pytest.mark.parametrize(
    "name, reference",
    [
        ("person/name", "#/$defs/person~1name"),
        ("person~name", "#/$defs/person~0name"),
        ("person name", "#/$defs/person%20name"),
        ("person", "#%2F$defs%2Fperson"),
        ("", "#/$defs/"),
    ],
)
def test_defs_pointer_names_use_json_pointer_escaping(name, reference):
    schema = {"$defs": {name: {"type": "integer", "minimum": 1}}, "$ref": reference}
    assert normalize_final_schema(schema) == {"type": "integer", "minimum": 1}


@pytest.mark.parametrize("reference", ["#/allOf/0", "#/definitions/Count", "#/$defs/Alias"])
def test_document_local_pointer_targets_and_chained_definitions(reference):
    schema = {
        "allOf": [{"type": "integer"}],
        "definitions": {"Count": {"type": "integer"}},
        "$defs": {"Alias": {"$ref": "#/definitions/Count"}},
        "$ref": reference,
    }
    wrapped = wrapped_final_schema(schema)
    validate_structured_result({"task_summary": "Counted", "final_result": 3}, wrapped)
    with pytest.raises(ValidationError):
        validate_structured_result({"task_summary": "Counted", "final_result": "three"}, wrapped)


def test_reference_siblings_are_conjunctions_not_weaker_overrides():
    schema = {
        "$defs": {"Count": {"type": "integer", "minimum": 5}},
        "$ref": "#/$defs/Count",
        "minimum": 1,
        "maximum": 8,
    }
    wrapped = wrapped_final_schema(schema)
    validate_structured_result({"task_summary": "Counted", "final_result": 6}, wrapped)
    for invalid in (2, 9, "6"):
        with pytest.raises(ValidationError):
            validate_structured_result(
                {"task_summary": "Counted", "final_result": invalid}, wrapped
            )


def test_draft_seven_ignores_reference_siblings_as_required_by_its_dialect():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {"Count": {"type": "integer"}},
        "$ref": "#/definitions/Count",
        "type": "string",
    }
    wrapped = wrapped_final_schema(schema)
    validate_structured_result({"task_summary": "Counted", "final_result": 6}, wrapped)


@pytest.mark.parametrize("keyword", ["type", "disallow", "extends"])
def test_legacy_schema_positions_cannot_hide_external_references(keyword):
    schema = {
        "$schema": "http://json-schema.org/draft-03/schema#",
        keyword: [{"$ref": "https://example.invalid/schema"}],
    }
    with pytest.raises(ValueError, match=r"does not support \$ref"):
        validate_final_schema(schema)


def test_legacy_inline_type_schemas_resolve_only_document_local_definitions():
    schema = {
        "$schema": "http://json-schema.org/draft-03/schema#",
        "definitions": {"Count": {"type": "integer"}},
        "type": [{"$ref": "#/definitions/Count"}],
    }
    wrapped = wrapped_final_schema(schema)
    validate_structured_result({"task_summary": "Counted", "final_result": 6}, wrapped)
    with pytest.raises(ValidationError):
        validate_structured_result({"task_summary": "Counted", "final_result": "six"}, wrapped)


def test_data_keywords_and_property_names_are_not_mistaken_for_reference_keywords():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}, "$id": {"type": "string"}},
        "default": {"$ref": "https://example.invalid/data", "$defs": {"literal": True}},
        "examples": [{"$id": "literal"}],
        "const": {"$ref": "literal", "$id": "identifier"},
    }
    normalized = normalize_final_schema(schema)
    assert normalized == schema
    validate_structured_result(
        {"task_summary": "Returned", "final_result": schema["const"]}, wrapped_final_schema(schema)
    )


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.invalid/schema",
        "file:///tmp/schema.json",
        "other.json#/$defs/Item",
        "#anchor",
        None,
        1,
    ],
)
def test_external_and_non_pointer_references_remain_rejected(reference):
    with pytest.raises(ValueError, match=r"does not support \$ref"):
        validate_final_schema({"$ref": reference})


@pytest.mark.parametrize("keyword", ["$id", "$dynamicRef", "$recursiveRef"])
def test_unsupported_resource_and_dynamic_reference_keywords_fail_closed(keyword):
    with pytest.raises(ValueError, match="does not support"):
        validate_final_schema({"$defs": {"Unused": {keyword: "https://example.invalid/schema"}}})


@pytest.mark.parametrize(
    "reference",
    ["#/$defs/Missing", "#/allOf/4", "#/allOf/-1", "#/allOf/00", "#/allOf/0/type/missing"],
)
def test_unresolved_pointers_fail_with_a_clear_error(reference):
    with pytest.raises(ValueError, match="Unknown final_schema reference"):
        validate_final_schema({"allOf": [{"type": "string"}], "$ref": reference})


def test_pointer_target_must_be_a_schema_and_escape_sequences_must_be_valid():
    with pytest.raises(ValueError, match="must target a schema"):
        validate_final_schema({"$ref": "#/description", "description": "a description"})
    with pytest.raises(ValueError, match="Invalid final_schema JSON pointer"):
        validate_final_schema({"$defs": {"bad~2": {}}, "$ref": "#/$defs/bad~2"})
    with pytest.raises(ValueError, match=r"does not support \$ref"):
        validate_final_schema(
            {"default": {"$ref": "https://example.invalid/schema"}, "$ref": "#/default"}
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#"},
        {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"},
    ],
)
def test_recursive_references_are_rejected_without_infinite_expansion(schema):
    with pytest.raises(ValueError, match="Recursive final_schema reference"):
        validate_final_schema(schema)


@pytest.mark.parametrize("value", [True, False])
def test_local_pointer_can_target_a_boolean_schema(value):
    schema = {"$defs": {"Allowed": value}, "$ref": "#/$defs/Allowed"}
    assert normalize_final_schema(schema) == ({} if value else {"not": {}})


def test_expansion_is_bounded_before_an_exponential_graph_can_be_materialized():
    definitions = {"Leaf": {"type": "string", "description": "x" * 2000}}
    for index in range(20):
        previous = "Leaf" if index == 0 else f"Level{index - 1}"
        definitions[f"Level{index}"] = {"allOf": [{"$ref": f"#/$defs/{previous}"}] * 2}
    schema = {"$defs": definitions, "$ref": "#/$defs/Level19"}
    with pytest.raises(ValueError, match="nesting depth|size limit"):
        validate_final_schema(schema)


def test_repeated_large_definitions_cannot_exceed_the_expanded_byte_limit():
    schema = {
        "$defs": {"Text": {"type": "string", "description": "x" * 2000}},
        "allOf": [{"$ref": "#/$defs/Text"} for _ in range(30)],
    }
    assert len(str(schema)) < MAX_FINAL_SCHEMA_BYTES
    with pytest.raises(ValueError, match="expanded final_schema.*size limit"):
        validate_final_schema(schema)


def test_preserving_the_root_dialect_cannot_bypass_the_expanded_size_limit(monkeypatch):
    import kedi.agent_adapter.subagents.schema as schema_module

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"Count": {"type": "integer"}},
        "$ref": "#/$defs/Count",
    }
    input_size = len(json.dumps(schema, separators=(",", ":")))
    normalized = normalize_final_schema(schema)
    assert len(json.dumps(normalized, separators=(",", ":"))) > input_size
    monkeypatch.setattr(schema_module, "MAX_FINAL_SCHEMA_BYTES", input_size)
    with pytest.raises(ValueError, match="expanded final_schema.*size limit"):
        normalize_final_schema(schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["pydantic", "shared-fallback"])
@pytest.mark.parametrize("route", ["capture", "profile", "explicit"])
@pytest.mark.parametrize("origin", ["python", "kedi"])
async def test_nested_defs_work_through_real_child_execution_and_preserve_native_models(
    native, route, origin
):
    from tests.test_subagent_json_output import child_adapter

    final = {"assignment": WIRE_ASSIGNMENT} if route == "capture" else WIRE_ASSIGNMENT
    adapter, requests, _ = child_adapter(
        [{"task_summary": "Assigned the review.", "final_result": final}], native=native
    )
    output = "  > output: Assignment\n" if route == "profile" else ""
    prompt = (
        "The assignment is [assignment: Assignment]."
        if route == "capture"
        else "Assign the review."
    )
    source = (
        "> profile: worker:\n  > instructions: Return the assignment.\n"
        + output
        + "> profile: coordinator:\n  > subagent: worker\n> use: coordinator\n"
    )
    if origin == "kedi":
        source = (
            '~Person(name: str, role: Literal["owner", "reviewer"])\n'
            "~Assignment(owner: Person, reviewers: list[Person], backup: Person | None)\n" + source
        )
    if route != "explicit":
        source += (
            "> task [job]: worker:\n  >> "
            + prompt
            + "\n> await [result]: job\n= `result.output"
            + (".assignment" if route == "capture" else "")
            + "`\n"
        )
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=adapter,
            runtime_globals={"Assignment": Assignment} if origin == "python" else {},
        )
        try:
            if route == "explicit":
                profile = runtime._evaluate_profile_definition(
                    runtime._profile_registry["coordinator"]
                )
                tool = next(
                    tool
                    for tool in runtime._resolve_agent_tool_specs(profile, adapter)
                    if tool.name == "delegate_task"
                )
                result = await tool.fn(
                    subagent="worker",
                    task=prompt,
                    final_schema=TypeAdapter(Assignment).json_schema(),
                )
                assert result["final_result"] == WIRE_ASSIGNMENT
            else:
                result = await asyncio.to_thread(runtime.run_main)
                assert isinstance(result, Assignment if origin == "python" else BaseModel)
                assert isinstance(result.owner, Person if origin == "python" else BaseModel)
                assert isinstance(result.reviewers[0], Person if origin == "python" else BaseModel)
                assert result.model_dump(mode="json") == WIRE_ASSIGNMENT
        finally:
            await runtime.aclose()
    assert len(requests) == 1
