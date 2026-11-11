from __future__ import annotations

from pathlib import Path

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.diagnostics import compute_diagnostics


def test_comments_inside_a_selective_import_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "helpers.kedi").write_text(
        "[greeting] = hello\n@shout(text) -> str:\n  = <`text.upper()`>\n> export: *\n",
        encoding="utf-8",
    )
    source = "> import: helpers:\n  # the procedure we need\n  shout\n  greeting\n= <shout(<greeting>)>\n"
    path = str(tmp_path / "main.kedi")

    assert compile_program(parse_program(source, source_path=path)).run_main() == "HELLO"
    assert compute_diagnostics(source, path) == []


@pytest.mark.parametrize("string", ['"a\rb"', '"\\N{not a character}"'])
def test_type_strings_python_rejects_are_parse_diagnostics(string: str) -> None:
    source = f"~Choice(kind: Literal[{string}])\n= ok\n"

    with pytest.raises(KediParseError) as caught:
        parse_program(source)
    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["K-TYPE-STRING"]
    diagnostics = compute_diagnostics(source, "choice.kedi")
    assert [(item.code, item.range.start.line) for item in diagnostics] == [("K-TYPE-STRING", 0)]
