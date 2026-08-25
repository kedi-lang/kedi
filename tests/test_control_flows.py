from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, get_args, get_origin

import pytest

import kedi
from kedi.agent_adapter import AgentAdapter
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_profile import current_block_agent_profile
from kedi.lang import (
    ConditionalLoopStmt,
    IfStmt,
    LoopStmt,
    TextSegment,
    compile_program,
    parse_program,
)
from kedi.lang.ast import CallSegment, VarArg
from kedi.lang.compiler.control_flow import loop_binding
from kedi.lang.parser import KediParseError
from kedi.lang.parser.cst_to_ast import _Walker
from tests.mock_adapter import MockAdapter


class ClaimAdapter(MockAdapter):
    def __init__(self, classify) -> None:
        super().__init__()
        self.classify = classify
        self.claim_calls: list[tuple[str, object]] = []

    def produce_sync(
        self,
        *,
        template: str,
        output_schema=None,
        output_type=None,
        **_kwargs: Any,
    ) -> bool:
        self.claim_calls.append((template, output_type))
        return self.classify(template)


class BareClaimAdapter(AgentAdapter[bool]):
    """Structured-output adapter without profile or tool override seams."""

    kind = "agent-framework"
    shortname = "bare-claim"
    capabilities = AdapterCapabilities(supports_structured_output=True)

    async def produce(self, *, template: str, **kwargs: Any) -> bool:
        return self.produce_sync(template=template, **kwargs)

    def produce_sync(self, *, template: str, **_kwargs: Any) -> bool:
        return "Claim: established" in template

    async def invoke(self, *, prompt: str, **_kwargs: Any) -> str:
        return prompt

    def invoke_sync(self, *, prompt: str, **_kwargs: Any) -> str:
        return prompt


def test_parse_if_owns_optional_else_and_nested_statements() -> None:
    program = parse_program(
        """
> if: `outer`:
  > if: `inner`:
    [result] = both
  > else:
    [result] = outer
> else:
  [result] = neither
""".lstrip()
    )

    outer = program.ordered_toplevel[0]
    assert isinstance(outer, IfStmt)
    assert outer.condition_code == "outer"
    assert outer.else_body is not None
    assert len(outer.else_body) == 1
    assert isinstance(outer.body[0], IfStmt)
    assert outer.body[0].else_body is not None
    assert outer.condition_loc is not None
    assert outer.condition_loc.start_line == 1


def test_parse_loop_preserves_binder_iterable_and_body() -> None:
    program = parse_program(
        """
> loop [n]: `range(1, 6)`:
  `total += n`
""".lstrip()
    )

    loop = program.ordered_toplevel[0]
    assert isinstance(loop, LoopStmt)
    assert loop.binder == "n"
    assert loop.iterable_code == "range(1, 6)"
    assert len(loop.body) == 1
    assert loop.binder_loc is not None
    assert loop.binder_loc.start_col == 9
    assert loop.iterable_loc is not None


@pytest.mark.parametrize(
    "source",
    [
        "> else:\n  [result] = no\n",
        "> if: `True`:\n  [result] = yes\n> else:\n  [result] = no\n> else:\n  [result] = duplicate\n",
        "> if: `True`:\n",
        "> loop [n]: `range(3)`:\n",
        "> loop [n: int]: `range(3)`:\n  [result] = <n>\n",
        "> if: True:\n  [result] = no\n",
        "> elif: `True`:\n  [result] = no\n",
        "> loop [n] in `range(3)`:\n  [result] = <n>\n",
        "> loop [n]: range(3):\n  [result] = <n>\n",
    ],
)
def test_control_flow_rejects_malformed_headers_and_bodies(source: str) -> None:
    with pytest.raises(KediParseError):
        parse_program(source)


def test_parse_distinguishes_deterministic_and_template_conditions() -> None:
    program = parse_program(
        """
> if: `ready`:
  [deterministic] = yes
> if: `ready`
  [classified] = yes
> loop: `ready`:
  `ready = False`
> loop: <work> remains unfinished
  [checked] = yes
""".lstrip()
    )

    deterministic_if, classified_if, deterministic_loop, classified_loop = program.ordered_toplevel
    assert isinstance(deterministic_if, IfStmt)
    assert deterministic_if.condition_code == "ready"
    assert deterministic_if.condition_segments is None
    assert isinstance(classified_if, IfStmt)
    assert classified_if.condition_code == ""
    assert classified_if.condition_segments is not None
    assert isinstance(deterministic_loop, ConditionalLoopStmt)
    assert deterministic_loop.condition_code == "ready"
    assert deterministic_loop.condition_segments is None
    assert isinstance(classified_loop, ConditionalLoopStmt)
    assert classified_loop.condition_segments is not None


def test_template_condition_preserves_internal_colons_and_nested_call_arguments() -> None:
    program = parse_program(
        "> if: Region: <judge(candidate, `value`, <inner()>)> is accepted\n  [result] = yes\n"
    )

    condition = program.ordered_toplevel[0]
    assert isinstance(condition, IfStmt)
    assert condition.condition_segments is not None
    assert isinstance(condition.condition_segments[0], TextSegment)
    assert "".join(
        segment.text for segment in condition.condition_segments if isinstance(segment, TextSegment)
    ).startswith("Region: ")


def test_condition_output_scan_ignores_non_literal_call_arguments() -> None:
    segment = CallSegment(name="judge", args=[VarArg(name="candidate")])

    assert _Walker._condition_outputs([segment]) == []


@pytest.mark.parametrize(
    "condition",
    [
        "[answer] is correct",
        "<judge(candidate [answer])>",
    ],
)
def test_template_condition_rejects_output_segments_with_specific_diagnostic(
    condition: str,
) -> None:
    with pytest.raises(KediParseError) as error:
        parse_program(f"> if: {condition}\n  [result] = yes\n")

    assert error.value.diagnostics[0].code == "K-CONDITION-OUTPUT"
    assert "cannot declare output fields" in error.value.diagnostics[0].message


def test_template_condition_allows_escaped_literal_brackets() -> None:
    program = parse_program(r"> if: \[literal\] is present" "\n  [result] = yes\n")
    condition = program.ordered_toplevel[0]

    assert isinstance(condition, IfStmt)
    assert condition.condition_segments is not None
    assert isinstance(condition.condition_segments[0], TextSegment)
    assert condition.condition_segments[0].text == "[literal] is present"


def test_top_level_if_requires_an_exact_bool() -> None:
    runtime = compile_program(parse_program("> if: `1`:\n  [result] = no\n"))
    with pytest.raises(Exception, match="exact bool"):
        runtime.run_main()


def test_top_level_if_executes_only_selected_branch() -> None:
    runtime = compile_program(
        parse_program(
            """
[hits: int] = `0`
> if: `False`:
  `hits += 100`
> else:
  `hits += 1`
= `hits`
""".lstrip()
        )
    )
    assert runtime.run_main() == 1


def test_template_if_renders_native_segments_and_uses_annotated_bool() -> None:
    adapter = ClaimAdapter(lambda prompt: "Claim: Ankara has 6 letters" in prompt)
    runtime = compile_program(
        parse_program(
            """
[city] = Ankara
[minimum: int] = `5`
> if: <city> has `len(city)` letters
  [status] = accepted
> else:
  [status] = rejected
= <status>
""".lstrip()
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "accepted"
    assert len(adapter.claim_calls) == 1
    prompt, output_type = adapter.claim_calls[0]
    assert prompt == (
        "Return true only if the following claim can be established as true from the available "
        "context; otherwise return false.\nClaim: Ankara has 6 letters"
    )
    assert get_origin(output_type) is Annotated
    assert get_args(output_type)[0] is bool
    assert "available context" in get_args(output_type)[1]
    assert "result" not in runtime._globals_env


def test_template_if_treats_unestablished_claim_as_false() -> None:
    adapter = ClaimAdapter(lambda _prompt: False)
    runtime = compile_program(
        parse_program(
            "> if: Atlantis exists\n  [result] = yes\n> else:\n  [result] = no\n= <result>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "no"


def test_template_if_supports_adapter_without_override_seams_or_instructions() -> None:
    runtime = compile_program(
        parse_program(
            "> artifacts:\n"
            "  enabled: false\n\n"
            "> if: established\n"
            "  [result] = yes\n"
            "> else:\n"
            "  [result] = no\n"
            "= <result>\n"
        ),
        adapter=BareClaimAdapter(),
    )

    assert runtime.run_main() == "yes"


def test_template_if_requires_an_exact_bool_from_the_adapter() -> None:
    runtime = compile_program(
        parse_program("> if: established\n  [result] = yes\n"),
        adapter=ClaimAdapter(lambda _prompt: 1),
    )

    with pytest.raises(TypeError, match="exact bool"):
        runtime.run_main()


def test_conditional_loop_re_evaluates_python_condition_before_each_iteration() -> None:
    checks: list[int] = []

    def should_continue(value: int) -> bool:
        checks.append(value)
        return value < 3

    runtime = compile_program(
        parse_program(
            """
[count: int] = `0`
> loop: `should_continue(count)`:
  `count += 1`
= `count`
""".lstrip()
        ),
        runtime_globals={"should_continue": should_continue},
    )

    assert runtime.run_main() == 3
    assert checks == [0, 1, 2, 3]


def test_template_conditional_loop_rerenders_claim_each_iteration() -> None:
    seen_claims: list[str] = []

    def classify(prompt: str) -> bool:
        claim = prompt.rsplit("Claim: ", 1)[1]
        seen_claims.append(claim)
        return int(claim.split()[0]) < 3

    runtime = compile_program(
        parse_program(
            """
[count: int] = `0`
> loop: <count> remains below `3`
  `count += 1`
= `count`
""".lstrip()
        ),
        adapter=ClaimAdapter(classify),
    )

    assert runtime.run_main() == 3
    assert seen_claims == [
        "0 remains below 3",
        "1 remains below 3",
        "2 remains below 3",
        "3 remains below 3",
    ]


def test_conditional_loop_allows_exact_limit_then_raises_before_next_body() -> None:
    runtime = compile_program(
        parse_program(
            """
[count: int] = `0`
> loop: `True`:
  `count += 1`
""".lstrip()
        ),
        loop_iteration_limit=2,
    )

    with pytest.raises(kedi.LoopIterationLimitError, match=r"iteration limit \(2\)"):
        runtime.run_main()
    assert runtime._globals_env["count"] == 2


def test_conditional_loop_false_first_runs_zero_iterations() -> None:
    runtime = compile_program(
        parse_program(
            """
[count: int] = `0`
> loop: `False`:
  `count += 1`
= `count`
""".lstrip()
        ),
        loop_iteration_limit=1,
    )

    assert runtime.run_main() == 0


def test_top_level_conditional_loop_keeps_the_last_body_return() -> None:
    runtime = compile_program(
        parse_program("[count: int] = `0`\n> loop: `count < 2`:\n  `count += 1`\n  = `count`\n")
    )

    assert runtime.run_main() == 2


def test_interactive_python_api_forwards_conditional_loop_limit() -> None:
    with kedi.interactive(loop_iteration_limit=1) as session:
        session.execute("[count: int] = `0`\n")
        with pytest.raises(kedi.LoopIterationLimitError, match=r"iteration limit \(1\)"):
            session.execute("> loop: `True`:\n  `count += 1`\n")
        assert session.execute("= `count`\n") == 1


def test_procedure_if_else_and_nested_if() -> None:
    runtime = compile_program(
        parse_program(
            """
@choose(outer: bool, inner: bool) -> str:
  [result] = neither
  > if: `outer`:
    [result] = outer
    > if: `inner`:
      [result] = both
  = <result>

= <choose(<`True`>, <`True`>)>
""".lstrip()
        )
    )
    assert runtime.run_main() == "both"


def test_loop_is_sequential_and_restores_shadowed_binder() -> None:
    runtime = compile_program(
        parse_program(
            """
[n: int] = `99`
[total: int] = `0`
> loop [n]: `range(1, 6)`:
  `total += n`
= `(total, n)`
""".lstrip()
        )
    )
    assert runtime.run_main() == (15, 99)


def test_loop_accepts_list_and_cleans_new_binder() -> None:
    runtime = compile_program(
        parse_program(
            """
[values: list[int]] = `[]`
> loop [n]: `[1, 2, 3]`:
  `values.append(n**2)`
= `values`
""".lstrip()
        )
    )
    assert runtime.run_main() == [1, 4, 9]
    assert "n" not in runtime._globals_env


def test_loop_accepts_builtin_generator_and_custom_iterables() -> None:
    class CustomIterable:
        def __iter__(self):
            return iter(("custom-a", "custom-b"))

    generator = (value for value in (4, 5))
    runtime = compile_program(
        parse_program(
            """
[seen: list] = `[]`
> loop [value]: `[1, 2]`:
  `seen.append(value)`
> loop [value]: `(3,)`:
  `seen.append(value)`
> loop [value]: `{"key": "value"}`:
  `seen.append(value)`
> loop [value]: `"xy"`:
  `seen.append(value)`
> loop [value]: `generator`:
  `seen.append(value)`
> loop [value]: `custom`:
  `seen.append(value)`
= `seen`
""".lstrip()
        ),
        runtime_globals={"generator": generator, "custom": CustomIterable()},
    )

    assert runtime.run_main() == [1, 2, 3, "key", "x", "y", 4, 5, "custom-a", "custom-b"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (1, "must return an Iterable"),
        (type("NonCallableIterable", (), {"__iter__": None})(), "must return an Iterable"),
        (type("BrokenIterable", (), {"__iter__": lambda self: 1})(), "invalid Iterable"),
        (type("LegacySequence", (), {"__getitem__": lambda self, index: index})(), "Iterable"),
    ],
)
def test_loop_rejects_values_without_a_valid_iterable_contract(
    value: object,
    message: str,
) -> None:
    runtime = compile_program(
        parse_program("> loop [value]: `candidate`:\n  [seen] = <value>\n"),
        runtime_globals={"candidate": value},
    )

    with pytest.raises(TypeError, match=message):
        runtime.run_main()
    assert "value" not in runtime._globals_env


def test_loop_supports_empty_and_negative_ranges() -> None:
    runtime = compile_program(
        parse_program(
            """
[values: list[int]] = `[]`
> loop [n]: `range(5, 0, -2)`:
  `values.append(n)`
> loop [unused]: `range(0)`:
  `values.append(100)`
= `values`
""".lstrip()
        )
    )
    assert runtime.run_main() == [5, 3, 1]
    assert "n" not in runtime._globals_env
    assert "unused" not in runtime._globals_env


def test_incremental_session_preserves_control_flow_writes() -> None:
    with kedi.interactive() as session:
        session.execute("[total: int] = `0`\n")
        session.execute("> loop [n]: `range(1, 4)`:\n  `total += n`\n")
        session.execute("> if: `total == 6`:\n  [status] = passed\n")
        assert session.execute("= `total, status`\n") == (6, "passed")


def test_control_flow_headers_are_evaluated_once() -> None:
    calls: list[str] = []

    def condition() -> bool:
        calls.append("condition")
        return True

    def values() -> range:
        calls.append("range")
        return range(3)

    runtime = compile_program(
        parse_program(
            """
[total: int] = `0`
> if: `condition()`:
  > loop [n]: `values()`:
    `total += n`
= `total`
""".lstrip()
        ),
        runtime_globals={"condition": condition, "values": values},
    )

    assert runtime.run_main() == 3
    assert calls == ["condition", "range"]


def test_nested_loop_restores_each_shadowed_binder() -> None:
    runtime = compile_program(
        parse_program(
            """
[n: int] = `99`
[seen: list] = `[]`
> loop [n]: `range(2)`:
  `seen.append(("outer-before", n))`
  > loop [n]: `range(10, 12)`:
    `seen.append(("inner", n))`
  `seen.append(("outer-after", n))`
= `(seen, n)`
""".lstrip()
        )
    )

    assert runtime.run_main() == (
        [
            ("outer-before", 0),
            ("inner", 10),
            ("inner", 11),
            ("outer-after", 0),
            ("outer-before", 1),
            ("inner", 10),
            ("inner", 11),
            ("outer-after", 1),
        ],
        99,
    )


def test_range_loop_restores_binder_after_body_error() -> None:
    runtime = compile_program(
        parse_program(
            """
[n: int] = `7`
> loop [n]: `range(2)`:
  `raise RuntimeError("stop")`
""".lstrip()
        )
    )

    with pytest.raises(RuntimeError, match="stop"):
        runtime.run_main()
    assert runtime._globals_env["n"] == 7


def test_control_body_profile_state_is_lexical() -> None:
    seen: list[str | None] = []

    def capture_model() -> None:
        seen.append(current_block_agent_profile().model)

    runtime = compile_program(
        parse_program(
            """
> model: outer-model
> if: `True`:
  > model: branch-model
  `capture_model()`
`capture_model()`
> loop [n]: `range(2)`:
  > model: iteration-model
  `capture_model()`
`capture_model()`
""".lstrip()
        ),
        runtime_globals={"capture_model": capture_model},
    )

    runtime.run_main()
    assert seen == [
        "branch-model",
        "outer-model",
        "iteration-model",
        "iteration-model",
        "outer-model",
    ]


def test_top_level_return_inside_selected_branch_is_preserved() -> None:
    runtime = compile_program(parse_program("> if: `True`:\n  = `42`\n"))
    assert runtime.run_main() == 42


def test_procedure_range_result_and_unselected_if_without_else() -> None:
    runtime = compile_program(
        parse_program(
            """
@last() -> str:
  > loop [n]: `range(3)`:
    = item-<n>
  > if: `False`:
    = unreachable

= <last()>
""".lstrip()
        )
    )

    assert runtime.run_main() == "item-2"


def test_procedure_range_without_body_return_runs_every_iteration() -> None:
    runtime = compile_program(
        parse_program(
            """
@sum_range() -> int:
  [total: int] = `0`
  > loop [n]: `range(4)`:
    `total += n`
  = `total`

= `sum_range()`
""".lstrip()
        )
    )

    assert runtime.run_main() == 6


def test_procedure_conditional_loop_re_evaluates_and_keeps_last_return() -> None:
    runtime = compile_program(
        parse_program(
            "@count_to_two() -> int:\n"
            "  [count: int] = `0`\n"
            "  > loop: `count < 2`:\n"
            "    `count += 1`\n"
            "    = `count`\n"
            "  = `count`\n\n"
            "= `count_to_two()`\n"
        )
    )

    assert runtime.run_main() == 2


def test_procedure_conditional_loop_continues_without_body_return() -> None:
    runtime = compile_program(
        parse_program(
            "@count_to_two() -> int:\n"
            "  [count: int] = `0`\n"
            "  > loop: `count < 2`:\n"
            "    `count += 1`\n"
            "  = `count`\n\n"
            "= `count_to_two()`\n"
        )
    )

    assert runtime.run_main() == 2


def test_top_level_control_flow_registers_nested_procedure_and_type() -> None:
    runtime = compile_program(
        parse_program(
            """
> if: `True`:
  ~Box(value: int)
  @make(value: int) -> Box:
    = `Box(value=value)`
= `make(7).value`
""".lstrip()
        )
    )

    assert runtime.run_main() == 7


def test_top_level_range_keeps_last_template_return_and_false_if_is_a_noop() -> None:
    runtime = compile_program(
        parse_program(
            """
> if: `False`:
  = unreachable
> loop [n]: `range(3)`:
  = item-<n>
""".lstrip()
        )
    )

    assert runtime.run_main() == "item-2"


def test_loop_binding_handles_current_env_identity_and_manual_binder_removal() -> None:
    runtime = compile_program(parse_program("= ok\n"))
    env: dict[str, object] = {}
    token = runtime._current_env.set(env)
    try:
        with loop_binding(runtime, env, "n", 1):
            assert env["n"] == 1
            del env["n"]
    finally:
        runtime._current_env.reset(token)

    assert "n" not in env


def test_module_can_export_value_written_by_control_flow(tmp_path: Path) -> None:
    (tmp_path / "values.kedi").write_text(
        "> export:\n  value\n\n> if: `True`:\n  [value] = selected\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: values:\n  value\n\n= <value>\n", encoding="utf-8")

    runtime = compile_program(parse_program(main.read_text(), source_path=str(main)))
    assert runtime.run_main() == "selected"


def test_loop_iteration_limit_must_be_a_positive_integer() -> None:
    program = parse_program("= ok\n")

    with pytest.raises(ValueError, match="positive integer"):
        compile_program(program, loop_iteration_limit=0)
    with pytest.raises(ValueError, match="positive integer"):
        with kedi.context(loop_iteration_limit=False):
            pass
