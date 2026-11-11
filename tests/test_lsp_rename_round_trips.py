"""Renames that must leave a runnable program behind."""

from __future__ import annotations

import pytest
from lsprotocol import types as lsp

from kedi.lang import compile_program, parse_program
from tests.lsp_helpers import definition_at, references_at, rename_at

URI = "file:///tmp/rename.kedi"
PATH = "/tmp/rename.kedi"


def _apply(source: str, edit: lsp.WorkspaceEdit | None) -> str:
    assert edit is not None and edit.changes is not None
    lines = source.split("\n")
    for item in sorted(
        edit.changes[URI],
        key=lambda change: (change.range.start.line, change.range.start.character),
        reverse=True,
    ):
        line = lines[item.range.start.line]
        lines[item.range.start.line] = (
            line[: item.range.start.character] + item.new_text + line[item.range.end.character :]
        )
    return "\n".join(lines)


def _starts(locations: list[lsp.Location]) -> list[tuple[int, int]]:
    return sorted((item.range.start.line, item.range.start.character) for item in locations)


@pytest.mark.parametrize("capture", ["[ans]", "[ans: int]"])
@pytest.mark.parametrize("occurrence", [1, 2])
def test_raw_invoke_captures_are_declarations(capture: str, occurrence: int) -> None:
    source = f"{capture} << How many moons does Mars have?\n= <ans>!\n"

    renamed = _apply(source, rename_at(source, URI, PATH, "ans", "count", occurrence=occurrence))

    assert renamed == source.replace("ans", "count")
    assert _starts(references_at(source, URI, PATH, "ans", True, occurrence=occurrence)) == [
        (0, 1),
        (1, 3),
    ]
    definition = definition_at(source, URI, PATH, "ans", occurrence=2)
    assert definition is not None and definition.range.start.line == 0


@pytest.mark.parametrize("block", ["> if: `True`:", "> loop [n]: `[1]`:"])
@pytest.mark.parametrize("occurrence", [1, 2])
def test_bindings_inside_top_level_blocks_rename_from_either_end(
    block: str, occurrence: int, capsys: pytest.CaptureFixture[str]
) -> None:
    source = f"{block}\n  [greeting] = hello\n  > show: <greeting> world\n= done\n"

    renamed = _apply(
        source, rename_at(source, URI, PATH, "greeting", "message", occurrence=occurrence)
    )

    assert renamed == source.replace("greeting", "message")
    assert compile_program(parse_program(renamed)).run_main() == "done"
    assert capsys.readouterr().out == "hello world\n"


def test_rename_reaches_inline_python_uses() -> None:
    source = (
        "[count: int] = `2`\n> if: `count > 1`:\n  [count] := `count + 1`\n= <count> and `count`\n"
    )

    renamed = _apply(source, rename_at(source, URI, PATH, "count", "total"))

    assert renamed == source.replace("count", "total")
    assert compile_program(parse_program(renamed)).run_main() == "3 and 3"
    assert len(references_at(source, URI, PATH, "count", True)) == 6


def test_rename_follows_python_scopes_in_blocks() -> None:
    source = (
        "[n: int] = `1`\n"
        "[squares: Any] = `None`\n"
        "[shifted: Any] = `None`\n"
        "```\n"
        "squares = [n * n for n in range(3)]\n"
        "shifted = (lambda n: n + 1)(10)\n"
        "def bump():\n"
        "    global n\n"
        "    n = n + 1\n"
        "bump()\n"
        "```\n"
        "= `(n, squares, shifted)`\n"
    )
    expected = source.replace("global n", "global m").replace("n = n + 1", "m = m + 1")
    expected = expected.replace("[n: int]", "[m: int]").replace("`(n, ", "`(m, ")

    renamed = _apply(source, rename_at(source, URI, PATH, "n", "m"))

    assert renamed == expected
    assert compile_program(parse_program(renamed)).run_main() == (2, [0, 1, 4], 11)


def test_rename_keeps_procedure_locals_apart() -> None:
    source = (
        "[count: int] = `5`\n"
        "@twice(count: int) -> int:\n"
        "  = `count * 2`\n"
        "@shifted() -> int:\n"
        "  = `count + 1`\n"
        "= `(twice(1), shifted())`\n"
    )

    module_rename = _apply(source, rename_at(source, URI, PATH, "count", "total"))
    parameter_rename = _apply(source, rename_at(source, URI, PATH, "count", "k", occurrence=2))

    assert module_rename == source.replace("[count: int]", "[total: int]").replace(
        "`count + 1`", "`total + 1`"
    )
    assert parameter_rename == source.replace("@twice(count", "@twice(k").replace(
        "`count * 2`", "`k * 2`"
    )
    for renamed in (module_rename, parameter_rename):
        assert compile_program(parse_program(renamed)).run_main() == (2, 6)


def test_references_to_an_undeclared_name_skip_python() -> None:
    source = "= <ghost> and `ghost`\n"

    assert _starts(references_at(source, URI, PATH, "ghost", True)) == [(0, 3)]
