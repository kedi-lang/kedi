from __future__ import annotations

import ast
import re
from decimal import Decimal
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from kedi import Constraints
from kedi.agent_adapter.output_schema import build_output_model
from tests.utils_llm import run_without_llm


@pytest.mark.parametrize(
    ("base", "options", "valid", "invalid"),
    [
        (float, {"ge": 0, "le": 2}, 0.0, -0.1),
        (float, {"ge": 0, "le": 2}, 2.0, 2.1),
        (float, {"gt": 0, "lt": 2}, 1.0, 0.0),
        (float, {"gt": 0, "lt": 2}, 1.0, 2.0),
        (int, {"multiple_of": 5}, 10, 11),
        (str, {"min_length": 1, "max_length": 3}, "abc", ""),
        (str, {"min_length": 1, "max_length": 3}, "a", "abcd"),
        (list[int], {"min_length": 1, "max_length": 2}, [1], []),
        (list[int], {"min_length": 1, "max_length": 2}, [1, 2], [1, 2, 3]),
        (str, {"pattern": r"^[A-Z]+$"}, "ABC", "abc"),
        (str, {"pattern": re.compile(r"^[A-Z]+$")}, "ABC", "abc"),
        (int, {"strict": True}, 1, "1"),
        (float, {"allow_inf_nan": False}, 1.0, float("inf")),
        (float, {"allow_inf_nan": False}, 1.0, float("nan")),
        (Decimal, {"max_digits": 4}, Decimal("12.34"), Decimal("123.45")),
        (Decimal, {"decimal_places": 2}, Decimal("1.23"), Decimal("1.234")),
    ],
)
def test_constraints_match_field(
    base: Any, options: dict[str, Any], valid: Any, invalid: Any
) -> None:
    constrained = TypeAdapter(Annotated[base, Constraints(**options)])
    reference = TypeAdapter(Annotated[base, Field(**options)])
    assert constrained.json_schema() == reference.json_schema()
    assert constrained.validate_python(valid) == reference.validate_python(valid)
    with pytest.raises(ValidationError) as error:
        constrained.validate_python(invalid)
    with pytest.raises(ValidationError) as expected:
        reference.validate_python(invalid)
    assert error.value.errors(include_input=False) == expected.value.errors(include_input=False)


def test_empty_constraints_preserve_type_behavior() -> None:
    constrained = TypeAdapter(Annotated[int, Constraints()])
    assert constrained.json_schema() == TypeAdapter(int).json_schema()
    assert constrained.validate_python("1") == 1


def test_constraints_in_python_model() -> None:
    class Review(BaseModel):
        score: Annotated[float, Constraints(ge=0, le=2)]
        title: str = Constraints(min_length=1, max_length=20)

    assert Review(score=1.5, title="Review").score == 1.5
    with pytest.raises(ValidationError):
        Review(score=3, title="Review")
    with pytest.raises(ValidationError):
        Review(score=1, title="")
    with pytest.raises(ValidationError):
        Review(score=1)


@pytest.mark.parametrize(
    "keyword", ["default", "default_factory", "alias", "description", "json_schema_extra"]
)
def test_non_constraint_field_options_are_rejected(keyword: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Constraints(**{keyword: None})


def test_kedi_output_schema_preserves_constraints() -> None:
    model = build_output_model(
        {
            "score": Annotated[float, "Evaluation score", Constraints(ge=0, le=2)],
        }
    )
    schema = model.model_json_schema()["properties"]["score"]
    assert schema["description"] == "Evaluation score"
    assert schema["minimum"] == 0
    assert schema["maximum"] == 2
    assert model(score=1.5).score == 1.5
    with pytest.raises(ValidationError):
        model(score=3)


@pytest.mark.parametrize("value", [0.0, 1.5, 2.0])
def test_constraints_in_kedi_custom_type(value: float) -> None:
    source = f"""```
from typing import Annotated
from kedi import Constraints
Score = Annotated[
    float,
    "Evaluation score",
    Constraints(ge=0, le=2),
]
```
~Review(
  score: `Score`
)
[review: Review] = `Review(score={value!r})`
= `review.score`
"""
    assert run_without_llm(source) == value


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_kedi_custom_type_rejects_out_of_range_values(value: float) -> None:
    source = f"""```
from typing import Annotated
from kedi import Constraints
Score = Annotated[float, Constraints(ge=0, le=2)]
```
~Review(score: `Score`)
[review: Review] = `Review(score={value!r})`
= `review.score`
"""
    with pytest.raises(RuntimeError, match="ValidationError"):
        run_without_llm(source)


@pytest.mark.parametrize("description", ["", '"Confidence of score", '])
def test_native_annotated_accepts_computed_constraint_metadata(description: str) -> None:
    source = f"""```
from kedi import Constraints
```
~Probability(
  score: float,
  confidence: Annotated[float, {description}`Constraints(ge=0, le=1)`]
)
[value: Probability] = `Probability(0.5, 0.9)`
= `value.model_dump(), Probability.model_json_schema()`
"""
    value, schema = run_without_llm(source)
    assert value == {"score": 0.5, "confidence": 0.9}
    assert schema["properties"]["confidence"]["minimum"] == 0
    assert schema["properties"]["confidence"]["maximum"] == 1
    with pytest.raises(RuntimeError, match="ValidationError"):
        run_without_llm(source.replace("Probability(0.5, 0.9)", "Probability(0.5, 1.1)"))


@pytest.mark.parametrize(
    "annotation,valid,invalid",
    [
        ('list[Annotated[float, "Score", `Constraints(ge=0, le=1)`]]', "[0.5]", "[2.0]"),
        (
            'dict[str, Annotated[float, "Score", `Constraints(ge=0, le=1)`]]',
            "{'x': 0.5}",
            "{'x': 2.0}",
        ),
        ('Optional[Annotated[float, "Score", `Constraints(ge=0, le=1)`]]', "None", "2.0"),
        ('Annotated[float, "Score", `Constraints(ge=0, le=1)`] | None', "0.5", "2.0"),
    ],
)
def test_computed_constraint_metadata_in_nested_native_types(annotation, valid, invalid):
    source = "```\nfrom kedi import Constraints\n```\n" + f"~Box(value: {annotation})\n"
    assert run_without_llm(source + f"= `Box({valid}).value`\n") == ast.literal_eval(valid)
    with pytest.raises(RuntimeError, match="ValidationError"):
        run_without_llm(source + f"= `Box({invalid}).value`\n")


@pytest.mark.parametrize(
    "annotation",
    ["`Constraints(ge=0)`", "list[`Constraints(ge=0)`]", 'Annotated[`Constraints(ge=0)`, "Score"]'],
)
def test_computed_metadata_is_still_rejected_in_type_positions(annotation):
    source = "```\nfrom kedi import Constraints\n```\n" + f"~Box(value: {annotation})\n"
    with pytest.raises(TypeError, match="does not evaluate to a valid type"):
        run_without_llm(source)


def test_native_constraints_need_explicit_python_and_import():
    from kedi.lang.parser import KediParseError, parse_program

    with pytest.raises(KediParseError):
        parse_program('~Box(value: Annotated[float, "Score", Constraints(ge=0)])\n')
    with pytest.raises(TypeError, match="Constraints.*not defined"):
        run_without_llm('~Box(value: Annotated[float, "Score", `Constraints(ge=0)`])\n')


def test_native_capture_sends_computed_constraints_to_the_model():
    from pydantic_ai.models.test import TestModel

    from kedi.agent_adapter.adapters import PydanticAdapter
    from kedi.lang import compile_program, parse_program

    model = TestModel(call_tools=[], custom_output_args={"confidence": 0.9})
    source = (
        "```\nfrom kedi import Constraints\n```\n"
        '>> The confidence is [confidence: Annotated[float, "Decision confidence", '
        "`Constraints(ge=0, le=1)`]].\n"
        "= `confidence`\n"
    )
    assert compile_program(parse_program(source), adapter=PydanticAdapter(model)).run_main() == 0.9
    assert model.last_model_request_parameters is not None
    tool = model.last_model_request_parameters.output_tools[0]
    field = tool.parameters_json_schema["properties"]["confidence"]
    assert field["description"] == "Decision confidence"
    assert field["minimum"] == 0
    assert field["maximum"] == 1
