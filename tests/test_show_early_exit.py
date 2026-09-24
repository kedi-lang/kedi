from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import pytest

import kedi
from kedi.errors import KediExecutionError
from kedi.lang import ShowStmt, compile_program, parse_program
from kedi.lang.parser import KediParseError
from tests.mock_adapter import MockAdapter


def run(source: str, **kwargs):
    runtime = compile_program(parse_program(source), adapter=MockAdapter(), **kwargs)
    return runtime.run_main()


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("Hello.", "Hello."),
        ("`42`", "42"),
        ("`None`", "None"),
        ("`False`", "False"),
        ("`[1, 2]`", "[1, 2]"),
        ("`'first\\nsecond'`", "first\nsecond"),
        (r"Literal \[brackets\].", "Literal [brackets]."),
    ],
)
def test_show_values(expression, expected, capsys):
    program = parse_program(f"> show: {expression}\n")
    assert isinstance(program.ordered_toplevel[0], ShowStmt)
    assert compile_program(program, adapter=MockAdapter()).run_main() is None
    assert capsys.readouterr().out == expected + "\n"


def test_show_once_and_continue(capsys):
    calls = []
    result = run(
        "[city] = Ankara\n"
        "> show: The selected city is <city>.\n"
        "> show: `calls.append(1) or len(calls)`\n"
        "> show: city\n"
        "= `len(calls)`\n",
        runtime_globals={"calls": calls},
    )
    assert result == 1
    assert calls == [1]
    assert capsys.readouterr().out == "The selected city is Ankara.\n1\ncity\n"


@pytest.mark.parametrize(
    "expression",
    [
        "The value is [value].",
        "<echo(The value is [value].)>",
        "<echo(<echo(The value is [value].)>)>",
    ],
)
def test_show_cannot_generate_captures(expression):
    with pytest.raises(KediParseError, match="cannot declare output captures"):
        parse_program(f"> show: {expression}\n")


@pytest.mark.parametrize("legacy", [False, True])
def test_manual_show_rejects_nested_argument_captures_before_execution(legacy):
    from kedi.lang.ast import CallSegment, LiteralArg, OutputSegment, Program

    argument = LiteralArg(value="")
    segments = [OutputSegment(name="answer")]
    if legacy:
        argument.__dict__["__render_segments__"] = segments
    else:
        argument.render_segments = segments
    program = Program(
        toplevel_body=[ShowStmt(segments=[CallSegment(name="echo", args=[argument])])]
    )
    adapter = MockAdapter()
    with pytest.raises(KediParseError, match="cannot declare output captures"):
        compile_program(program, adapter=adapter)
    assert adapter.model_calls == []


def test_show_allows_explicit_model_procedure_and_existing_capture(capsys):
    adapter = MockAdapter()
    program = parse_program(
        "@answer() -> str:\n"
        "  >> The answer is [value].\n"
        "  = <value>\n"
        "@echo(value: str) -> str:\n"
        "  = <value>\n"
        "[existing] = ready\n"
        "> show: <echo(<echo(<existing>)>)>\n"
        "> show: <answer()>\n"
    )
    compile_program(program, adapter=adapter).run_main()
    assert len(adapter.model_calls) == 1
    assert capsys.readouterr().out.splitlines()[0] == "ready"


@pytest.mark.parametrize("value", [None, 0, False, ""])
def test_typed_template_call_return_preserves_native_falsy_values(value):
    annotation = "str | bool | int | None"
    result = run(
        f"@empty() -> {annotation}:\n  = `{value!r}`\n"
        f"@outer() -> {annotation}:\n  = <empty()>\n"
        "= `outer()`\n"
    )
    assert result == value
    assert type(result) is type(value)


def test_nullable_call_initialization_and_assignment_preserve_none():
    assert (
        run(
            "@empty() -> int | None:\n  = `None`\n"
            "[value: int | None] = <empty()>\n"
            "`assert value is None`\n"
            "[value] := `1`\n"
            "[value] := <empty()>\n"
            "= `value`\n"
        )
        is None
    )


def test_typed_template_text_none_stays_text():
    assert run("@label() -> str | None:\n  = None\n= `label()`\n") == "None"


@pytest.mark.parametrize(
    "source",
    [
        "= `1`\n= `2`\n",
        "= `1`\n> show: hidden\n",
        "@pick() -> int:\n  = `1`\n  = `2`\n",
        "> if: `True`:\n  = yes\n> else:\n  = no\n> show: hidden\n",
    ],
)
def test_unreachable_source_fails_before_execution(source):
    with pytest.raises(KediParseError) as error:
        parse_program(source, source_path="example.kedi")
    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == "kedi.unreachable_statement"
    assert diagnostic.span.source_path == "example.kedi"


@pytest.mark.parametrize("flag,expected", [(True, 1), (False, 2)])
def test_branch_returns_are_not_one_return_per_file(flag, expected):
    assert (
        run(f"@pick(flag: bool) -> int:\n  > if: `flag`:\n    = `1`\n  = `2`\n= `pick({flag!r})`\n")
        == expected
    )


def test_return_does_not_evaluate_later_predicate():
    assert run("> if: `True`:\n  = done\n> if: `1 / 0`:\n  = invalid\n") == "done"


def test_loop_return_closes_iterator():
    events = []

    def values():
        try:
            events.append("first")
            yield 1
            events.append("second")
            yield 2
        finally:
            events.append("closed")

    assert run("> loop [item]: `values()`:\n  = `item`\n", runtime_globals={"values": values}) == 1
    assert events == ["first", "closed"]


def test_empty_loop_falls_through():
    assert run("> loop [item]: `[]`:\n  = `item`\n= `7`\n") == 7


def test_nested_procedure_keeps_own_return(capsys):
    assert (
        run(
            "@outer() -> int:\n"
            "  @inner() -> int:\n    = `2`\n"
            "  > show: `inner()`\n"
            "  = `3`\n"
            "= `outer()`\n"
        )
        == 3
    )
    assert capsys.readouterr().out == "2\n"


@pytest.mark.parametrize(
    "tail",
    [
        "= `1`\n",
        "> if: `False`:\n  = hidden\n",
        "> loop [n]: `[]`:\n  = `n`\n",
    ],
)
def test_interactive_return_rejected_before_prelude(tail, capsys):
    with kedi.interactive() as session:
        with pytest.raises(KediParseError, match="Interactive root returns"):
            session.execute('```\nprint("must not execute")\n```\n' + tail)
    assert capsys.readouterr().out == ""


def test_interactive_procedures_and_show(capsys):
    with kedi.interactive() as session:
        assert session.execute("@answer() -> int:\n  = `42`\n") is None
        assert session.execute("> show: `answer()`\n") is None
        assert session.execute("> show: `answer()`\n") is None
    assert capsys.readouterr().out == "42\n42\n"


def test_spec_interactive_import_example_executes(tmp_path, monkeypatch, capsys):
    spec = (Path(__file__).resolve().parents[1] / "docs/KEDI_DSL.md").read_text()
    source = next(
        block
        for block in re.findall(r"```python\n(.*?)\n```", spec, re.S)
        if 'with kedi.interactive(cwd="examples/cells")' in block
    )
    cells = tmp_path / "examples" / "cells"
    cells.mkdir(parents=True)
    (cells / "helpers.kedi").write_text("[answer: int] = `42`\n> export:\n  answer\n")
    monkeypatch.chdir(tmp_path)
    exec(compile(source, "docs/KEDI_DSL.md", "exec"), {"kedi": kedi})
    assert capsys.readouterr().out == "42\n"


def test_show_failure_is_source_mapped():
    with pytest.raises(KediExecutionError) as error:
        run("> show: `1 / 0`\n")
    assert "division by zero" in str(error.value)


def test_map_returns_are_joined_in_source_order():
    assert run("> loop [n]: `range(3)`:\n  [value: int] = `n`\n> map:\n  = `value`\n= `99`\n") == 0


def test_map_return_does_not_hide_a_sibling_failure():
    def fail_on_one(value: int) -> None:
        if value == 1:
            raise RuntimeError("map sibling failed")

    with pytest.raises(KediExecutionError, match="map sibling failed"):
        run(
            "> loop [n]: `range(2)`:\n  [value: int] = `n`\n"
            "> map:\n  `fail_on_one(value)`\n  = `value`\n",
            runtime_globals={"fail_on_one": fail_on_one},
        )


def test_concurrent_program_return_isolation():
    def execute(value):
        return run(f"> if: `True`:\n  = `{value}`\n= `99`\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(execute, [11, 22])) == [11, 22]


def test_procedure_map_returns_without_becoming_an_error():
    assert (
        run(
            "@pick() -> int:\n"
            "  > loop [n]: `range(3)`:\n    [value: int] = `n`\n"
            "  > map:\n    = `value`\n"
            "  = `99`\n"
            "= `pick()`\n"
        )
        == 0
    )


def test_manual_program_return_and_capture_validation():
    from kedi.lang.ast import (
        OutputSegment,
        ProcedureDef,
        Program,
        ReturnBlockStmt,
        ReturnStmt,
    )

    assert compile_program(Program(toplevel_return=ReturnBlockStmt(code="7"))).run_main() == 7
    with pytest.raises(KediParseError, match="cannot declare output captures"):
        compile_program(Program(toplevel_body=[ShowStmt(segments=[OutputSegment(name="value")])]))
    with pytest.raises(KediParseError, match="Unreachable statement after return"):
        compile_program(
            Program(
                procedures=[
                    ProcedureDef(
                        name="stop",
                        params=[],
                        body=[ReturnStmt(), ShowStmt()],
                    )
                ]
            )
        )


def test_show_hover():
    from lsprotocol.types import Position
    from kedi.lang.ast import Program, ReturnStmt, Span
    from kedi.lsp.features.hover import _generic_hover, compute_hover

    hover = compute_hover("> show: Hello.\n", "/tmp/show.kedi", Position(0, 3))
    assert hover is not None
    assert "Display" in str(hover)

    legacy_return = ReturnStmt(loc=Span(0, 6, 1, 1, 1, 7))
    legacy_hover = _generic_hover(
        Program(toplevel_return=legacy_return),
        Position(0, 1),
    )
    assert legacy_hover is not None
    assert "Return" in str(legacy_hover)


def test_interactive_diagnostics_use_the_same_return_boundary():
    from kedi.lsp.features.diagnostics import compute_diagnostics

    assert compute_diagnostics("= value\n", None) == []
    diagnostics = compute_diagnostics("= value\n", None, interactive=True)
    assert diagnostics[0].code == "kedi.interactive_return"
    assert (
        compute_diagnostics(
            "@value() -> int:\n  = `1`\n> show: `value()`\n", None, interactive=True
        )
        == []
    )
