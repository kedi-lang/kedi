from __future__ import annotations

from typing import Annotated, Literal, Mapping, Optional

import pytest
from kedi_typesafe.core.schema import build_evaluation_plan
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from kedi import Constraints
from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.lang import compile_program, parse_program
from kedi.lang.procedure_schema import procedure_json_schema, procedure_return_json_schema
from kedi.typesafe import BooleanCriteria, ChoiceCriteria, Probability, Rubric
from tests.mock_adapter import MockAdapter
from tests.utils_llm import run_without_llm


@pytest.mark.parametrize("builder", [PydanticAdapter.type_builder, LangChainAdapter.type_builder])
@pytest.mark.parametrize(
    ("annotation", "path", "description"),
    [
        (list[Annotated[str, "Element"]], ("items",), "Element"),
        (dict[str, Annotated[str, "Value"]], ("additionalProperties",), "Value"),
        (dict[Annotated[str, "Key"], int], ("propertyNames",), "Key"),
        (Mapping[str, Annotated[str, "Mapped value"]], ("additionalProperties",), "Mapped value"),
        (Annotated[str, "Union branch"] | int, ("anyOf", 0), "Union branch"),
        (Optional[Annotated[str, "Optional value"]], ("anyOf", 0), "Optional value"),
        (
            list[dict[str, Annotated[str, "Deep value"]]],
            ("items", "additionalProperties"),
            "Deep value",
        ),
    ],
)
def test_adapter_output_schemas_preserve_nested_descriptions(
    builder, annotation, path, description
) -> None:
    schema = builder({"value": annotation}).model_json_schema()["properties"]["value"]
    for segment in path:
        schema = schema[segment]
    assert schema["description"] == description


def test_kedi_type_definition_preserves_outer_and_item_descriptions() -> None:
    schema = run_without_llm(
        '~Box(value: Annotated[list[Annotated[str, "Element"]], "Collection"])\n'
        "= `Box.model_json_schema()`\n"
    )["properties"]["value"]

    assert schema["description"] == "Collection"
    assert schema["items"]["description"] == "Element"


def test_worlds_richest_people_template_passes_item_description_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        ">> What's the worlds richest [persons: list[Annotated[str, \"The name and surname "
        'of the person"]]]\n'
        "= `persons`\n"
    )
    captured: list[dict] = []
    original_builder = PydanticAdapter.type_builder

    def recording_builder(output_schema):
        model = original_builder(output_schema)
        captured.append(model.model_json_schema())
        return model

    monkeypatch.setattr(PydanticAdapter, "type_builder", staticmethod(recording_builder))
    adapter = PydanticAdapter(
        TestModel(call_tools=[], custom_output_args={"persons": ["Example Person"]})
    )

    result = compile_program(parse_program(source), adapter=adapter).run_main()

    assert result == ["Example Person"]
    assert captured[0]["properties"]["persons"]["items"]["description"] == (
        "The name and surname of the person"
    )


def test_procedure_schema_preserves_descriptions_at_every_level() -> None:
    source = (
        '@inspect(value: Annotated[str, "Input"], '
        'items: list[Annotated[str, "Item"]], '
        'fallback: Annotated[str, "Default"] = `"empty"`) '
        '-> Annotated[list[Annotated[str, "Result item"]], "Result"]:\n'
        "  = `items`\n"
    )
    program = parse_program(source)
    runtime = compile_program(program, adapter=MockAdapter())
    proc = program.procedures[0]

    inputs = procedure_json_schema(runtime, proc)["properties"]
    output = procedure_return_json_schema(runtime, proc)

    assert inputs["value"]["description"] == "Input"
    assert inputs["items"]["items"]["description"] == "Item"
    assert inputs["fallback"]["description"] == "Default"
    assert inputs["fallback"]["default"] == "empty"
    assert output["description"] == "Result"
    assert output["items"]["description"] == "Result item"


def test_nested_description_preserves_constraints_and_validation() -> None:
    score = Annotated[float, "Score", Constraints(ge=0, le=2)]
    model = PydanticAdapter.type_builder({"scores": list[score]})
    item = model.model_json_schema()["properties"]["scores"]["items"]

    assert item["description"] == "Score"
    assert item["minimum"] == 0
    assert item["maximum"] == 2
    assert model(scores=[1.5]).scores == [1.5]
    with pytest.raises(ValidationError):
        model(scores=[3.0])


def test_nested_constraint_without_description_is_preserved() -> None:
    model = PydanticAdapter.type_builder({"scores": list[Annotated[int, Constraints(ge=0)]]})
    item = model.model_json_schema()["properties"]["scores"]["items"]

    assert item["minimum"] == 0
    assert "description" not in item
    assert model(scores=[1]).scores == [1]
    with pytest.raises(ValidationError):
        model(scores=[-1])


def test_optional_jev_choice_preserves_criteria_and_instructions() -> None:
    # Python 3.10 unions hash their members; ChoiceCriteria contains unhashable metadata.
    # Annotating the nullable type keeps the same schema without hashing that metadata.
    choice = Annotated[
        Literal["keep", "drop"] | None,
        "Select the matching route",
        ChoiceCriteria({"keep": "Retain it", "drop": "Discard it"}),
    ]
    schema = PydanticAdapter.type_builder({"route": choice}).model_json_schema()
    plan = build_evaluation_plan(schema)

    assert plan.questions[0].instructions == "Select the matching route"
    assert plan.questions[0].criteria == {"keep": "Retain it", "drop": "Discard it"}
    assert plan.questions[0].nullable is True


@pytest.mark.parametrize(
    ("annotation", "description", "kind"),
    [
        (Annotated[Probability, "Decision confidence"], "Decision confidence", None),
        (
            Annotated[Annotated[float, Rubric(["Poor", "Good"])], "Answer quality"],
            "Answer quality",
            "score",
        ),
        (
            Annotated[
                Annotated[bool, BooleanCriteria(true="Yes", false="No")],
                "Check the answer",
            ],
            "Check the answer",
            "noul",
        ),
        (
            Annotated[
                Annotated[
                    Literal["keep", "drop"],
                    ChoiceCriteria({"keep": "Keep", "drop": "Drop"}),
                ],
                "Pick an option",
            ],
            "Pick an option",
            "choice",
        ),
    ],
)
def test_jev_alias_keeps_its_criteria_and_new_description(annotation, description, kind) -> None:
    for field_type, nested in ((annotation, False), (list[annotation], True)):
        schema = PydanticAdapter.type_builder({"value": field_type}).model_json_schema()
        field = schema["properties"]["value"]
        if nested:
            field = field["items"]
        assert field["description"] == description
        assert field.get("x-typesafe", {}).get("kind") == kind


def test_native_kedi_annotation_can_describe_probability_alias() -> None:
    source = """```
from kedi.typesafe import Probability
```
~Result(value: Annotated[Probability, "Decision confidence"])
= `Result.model_json_schema()`
"""
    field = run_without_llm(source)["properties"]["value"]
    assert field["description"] == "Decision confidence"
    assert field["minimum"] == 0
    assert field["maximum"] == 1
