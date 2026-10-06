import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from pathlib import Path

from kedi.agent_adapter import PydanticAdapter
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program

ROOT = Path(__file__).parent / "files/decision_shapes"


@pytest.mark.parametrize(
    "name,output,expected",
    [
        (
            "nested",
            {
                "review": {
                    "delivery": {"damaged": True, "damage_probability": 0.96},
                    "category": "billing",
                    "quality": 1.9,
                }
            },
            {
                "damaged": True,
                "probability_positive": True,
                "category": "billing",
                "accepted": True,
            },
        ),
        ("labels", {"labels": ["billing", "technical"]}, ["billing", "technical"]),
        ("optional_choice", {"team": "billing"}, {"team": "billing"}),
        ("optional_choice", {"team": None}, {"team": None}),
        (
            "tool_dispatch",
            {"team": "billing"},
            {"receipt": "billing-queued", "effects": ["billing"]},
        ),
        ("tool_dispatch", {"team": "access"}, {"receipt": "access-queued", "effects": ["access"]}),
        (
            "candidates",
            {"contact": {"active": "help@example.org", "current_document": True}},
            {"active": "help@example.org", "current_document": True},
        ),
        (
            "candidates",
            {"contact": {"active": None, "current_document": True}},
            {"active": None, "current_document": True},
        ),
    ],
)
def test_decision_shape_fixture_executes_not_only_parses(name, output, expected):
    model = TestModel(call_tools=[], custom_output_args=output)
    engine = SequentialEngine()
    runtime = None
    try:
        runtime = compile_program(
            parse_program((ROOT / f"{name}.kedi").read_text()),
            adapter=PydanticAdapter(model),
            engine=engine,
            runtime_globals={"evidence": "Fixture data"},
        )
        assert runtime.run_main() == expected
    finally:
        if runtime is not None:
            asyncio.run(runtime.aclose())
        engine.shutdown()
