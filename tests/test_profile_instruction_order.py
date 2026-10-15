from __future__ import annotations

import pytest

from kedi.agent_profile.compile import profile_stmt_to_definition
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        ("  > use: base\n  > instructions: Local rules.\n", "Local rules."),
        ("  > instructions: Local rules.\n  > use: base\n", "Base rules."),
    ],
)
def test_profile_instructions_and_nested_use_follow_source_order(members, expected):
    source = (
        "> profile: base:\n  > instructions: Base rules.\n"
        "> profile: reviewer:\n"
        + members
        + "> use: reviewer\n>> The review is [review: str].\n= <review>\n"
    )
    adapter = MockAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert expected in adapter.instruction_calls[0]
    excluded = "Base rules." if expected == "Local rules." else "Local rules."
    assert excluded not in adapter.instruction_calls[0]


def test_profile_source_order_retains_tool_blocks_and_directive_locations():
    program = parse_program(
        "> profile: review:\n"
        "  > use:\n    first\n    second\n"
        "  > instructions: Review carefully.\n"
        "  > use: other_profile\n"
        "  > effort: low\n",
        source_path="review.kedi",
    )
    definition = profile_stmt_to_definition(program.profile_defs["review"])
    assert [member.kind for member in definition.members] == ["tools", "system", "use", "effort"]
    assert all(member.loc is not None for member in definition.members)
    assert [member.loc.start_line for member in definition.members] == [2, 5, 6, 7]


@pytest.mark.parametrize("bound_model", [None, "definition-model"])
def test_captured_instruction_absence_does_not_disable_unbound_model_fallback(bound_model):
    source = (
        (f"> model: {bound_model}\n" if bound_model is not None else "")
        + "@report() -> str:\n  >> The status is [status: str].\n  = <status>\n"
        "@later_tool() -> str:\n  = done\n"
        "> instructions: Caller-only guidance.\n> model: caller-model\n"
        "> use: later_tool\n= <report()>\n"
    )
    adapter = MockAdapter(default_model="adapter-model")
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert adapter.model_calls == [bound_model or "caller-model"]
    assert "Caller-only guidance." not in adapter.instruction_calls[0]
    assert "later_tool" not in adapter.tool_calls[0]
