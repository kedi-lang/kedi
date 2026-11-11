from __future__ import annotations

from textwrap import indent

import pytest

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize("left", ["<name>", "<`name`>", "`name`", "<echo()>"])
@pytest.mark.parametrize("right", ["<name>", "<`name`>", "`name`", "<echo()>"])
def test_adjacent_segment_kinds_preserve_their_separator(left, right) -> None:
    source = f"[name] = Ada\n@echo() -> str:\n  = Ada\n= {left} {right}\n"
    assert compile_program(parse_program(source)).run_main() == "Ada Ada"


@pytest.mark.parametrize("scope", ["top", "procedure", "profile"])
@pytest.mark.parametrize("multiline", [False, True])
@pytest.mark.parametrize("second", ["<last>", "<`last`>"])
def test_instructions_preserve_spacing_between_adjacent_substitutions(
    scope, multiline, second
) -> None:
    text = f"Use <first> {second} now."
    directive = f"> instructions:\n  {text}\n" if multiline else f"> instructions: {text}\n"
    prompt = ">> The name is [name].\n"
    source = "[first] = Ada\n[last] = Lovelace\n"
    if scope == "procedure":
        source += "@reply():\n" + indent(directive + prompt, "  ") + "= <reply()>\n"
    elif scope == "profile":
        source += "> profile: reviewer:\n" + indent(directive, "  ")
        source += "> use: reviewer\n" + prompt
    else:
        source += directive + prompt
    adapter = MockAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert "Use Ada Lovelace now." in adapter.instruction_calls[0]


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("<first> <last>", "Ada Lovelace"),
        ("<first>  <last>", "Ada  Lovelace"),
        ("<first>\t<last>", "Ada\tLovelace"),
        (r"<first> \t <last>", "Ada \t Lovelace"),
        (r"<first> \n <last>", "Ada \n Lovelace"),
        (r"<first> \s <last>", "Ada   Lovelace"),
        (r"\s<first>\s", " Ada "),
        (r"\t<first>\n", "\tAda\n"),
        ("`' Ada '`", " Ada "),
        ("<first>   ", "Ada"),
    ],
)
@pytest.mark.parametrize("context", ["return", "binding", "typed_procedure"])
def test_template_boundaries_preserve_meaningful_whitespace(expression, expected, context) -> None:
    source = "[first] = Ada\n[last] = Lovelace\n"
    if context == "return":
        source += f"= {expression}\n"
    elif context == "binding":
        source += f"[value: str] = {expression}\n= `value`\n"
    else:
        source += f"@format() -> str:\n  = {expression}\n= `format()`\n"
    assert compile_program(parse_program(source)).run_main() == expected


def test_mixed_call_arguments_keep_inter_segment_spaces() -> None:
    source = """@greet(name: str) -> str:
  = `name`
[first] = Ada
[last] = Lovelace
= <greet(Lady <first> <last>)>
"""
    assert compile_program(parse_program(source)).run_main() == "Lady Ada Lovelace"


@pytest.mark.parametrize(
    "expression, expected", [(r"\s`1`", " 1"), (r"`1`\s", "1 "), (r"\n<one()>", "\n1")]
)
def test_escaped_whitespace_prevents_native_value_shortcut(expression, expected) -> None:
    source = f"@one() -> int:\n  = `1`\n[value: str] = {expression}\n= `value`\n"
    result = compile_program(parse_program(source)).run_main()
    assert isinstance(result, str)
    assert result == expected


def test_interpolated_value_whitespace_is_not_trimmed_by_typed_return() -> None:
    source = "@echo(value: str) -> str:\n  = <value>\n= `echo(' padded ')`\n"
    assert compile_program(parse_program(source)).run_main() == " padded "


@pytest.mark.parametrize("raw", [False, True])
def test_model_prompts_preserve_adjacent_substitution_spacing(raw) -> None:
    adapter = MockAdapter()
    source = "[first] = Ada\n[last] = Lovelace\n"
    source += (
        "[reply] << <first> <last> is a programmer.\n"
        if raw
        else ">> <first> <last> is a [profession].\n"
    )
    compile_program(parse_program(source), adapter=adapter).run_main()
    assert len(adapter.request_calls) == 1
    assert "Ada Lovelace is a" in adapter.request_calls[0]


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("trailing", ["   ", "\t "])
def test_model_prompts_ignore_unescaped_trailing_source_whitespace(raw, trailing) -> None:
    requests = []
    for suffix in ("", trailing):
        adapter = MockAdapter()
        source = (
            f"[reply] << Say hello.{suffix}\n"
            if raw
            else f">> The greeting is [greeting].{suffix}\n"
        )
        compile_program(parse_program(source), adapter=adapter).run_main()
        requests.append(adapter.request_calls)
    assert requests[0] == requests[1]


def test_multiline_prompt_trims_each_source_line_without_losing_escaped_spaces() -> None:
    requests = []
    for suffix in ("", "   "):
        adapter = MockAdapter()
        source = f">> The person is [name].{suffix}\n   Their city is [city].\\s{suffix}\n"
        compile_program(parse_program(source), adapter=adapter).run_main()
        requests.append(adapter.request_calls)
    assert requests[0] == requests[1]
