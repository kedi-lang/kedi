from __future__ import annotations

from types import SimpleNamespace

from lsprotocol import types as lsp

from kedi.lsp import kedi_docstrings
from tests.lsp_helpers import decode_semantic_tokens, position_of


PY_SOURCE = '''from pydantic import BaseModel

import kedi


class Person(BaseModel):
    name: str


@kedi.query
def extract_person(*, template: str, output_type: type[Person]) -> Person:
    """kedi
    >> Extract [output: `output_type`] from <template>.
    = `output`
    """
    ...
'''


def test_kedi_docstring_ranges_extract_cleaned_source() -> None:
    ranges = kedi_docstrings.compute_kedi_docstring_ranges(PY_SOURCE)

    assert len(ranges) == 1
    assert ranges[0]["text"] == (">> Extract [output: `output_type`] from <template>.\n= `output`")
    assert ranges[0]["range"]["start"] == {"line": 12, "character": 4}


def test_kedi_docstring_hover_maps_back_to_python_docstring() -> None:
    pos = position_of(PY_SOURCE, "template>")  # cursor inside <template>

    hover = kedi_docstrings.compute_kedi_docstring_hover(PY_SOURCE, "/tmp/api.py", pos)

    assert hover is not None
    assert "template" in hover["contents"]["value"]
    assert hover["range"]["start"]["line"] == 12


def test_kedi_docstring_definition_and_references_are_remapped() -> None:
    uri = "file:///tmp/api.py"
    usage = position_of(PY_SOURCE, "template>")

    definitions = kedi_docstrings.compute_kedi_docstring_definition(
        PY_SOURCE,
        uri,
        "/tmp/api.py",
        usage,
    )
    refs = kedi_docstrings.compute_kedi_docstring_references(
        PY_SOURCE,
        uri,
        "/tmp/api.py",
        usage,
        include_declaration=True,
    )

    assert definitions
    assert definitions[0]["uri"] == uri
    assert definitions[0]["range"]["start"]["line"] == 10
    assert refs
    assert all(ref["uri"] == uri for ref in refs)


def test_kedi_docstring_diagnostics_remap_parse_errors() -> None:
    source = '''import kedi

@kedi.query
def broken(name: str) -> str:
    """kedi
    > effort: impossible
    = <name>
    """
    ...
'''

    diagnostics = kedi_docstrings.compute_kedi_docstring_diagnostics(source, "/tmp/api.py")

    assert diagnostics
    assert diagnostics[0]["range"]["start"]["line"] == 5
    assert "effort" in diagnostics[0]["message"] or "Invalid" in diagnostics[0]["message"]


def test_kedi_docstring_semantic_tokens_are_remapped() -> None:
    tokens = kedi_docstrings.compute_kedi_docstring_semantic_tokens(PY_SOURCE)
    decoded = decode_semantic_tokens(tokens["data"])

    assert any(line == 12 for line, *_ in decoded)


def test_kedi_docstring_ignores_non_kedi_docstrings_and_invalid_python() -> None:
    assert kedi_docstrings.compute_kedi_docstring_ranges("def f():\n    '''plain'''\n") == []
    assert kedi_docstrings.compute_kedi_docstring_ranges("def broken(:\n") == []


def test_kedi_docstring_iter_skips_manually_empty_function_body(monkeypatch) -> None:
    import ast

    empty_function = ast.FunctionDef(
        name="f",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )
    monkeypatch.setattr(
        kedi_docstrings.ast,
        "parse",
        lambda source: ast.Module(body=[empty_function], type_ignores=[]),
    )

    assert list(kedi_docstrings._iter_kedi_docstrings("")) == []


def test_kedi_docstring_position_outside_returns_no_feature() -> None:
    pos = lsp.Position(line=0, character=0)

    assert kedi_docstrings.compute_kedi_docstring_hover(PY_SOURCE, "/tmp/api.py", pos) is None
    assert (
        kedi_docstrings.compute_kedi_docstring_definition(
            PY_SOURCE,
            "file:///tmp/api.py",
            "/tmp/api.py",
            pos,
        )
        == []
    )
    assert (
        kedi_docstrings.compute_kedi_docstring_references(
            PY_SOURCE,
            "file:///tmp/api.py",
            "/tmp/api.py",
            pos,
            include_declaration=True,
        )
        == []
    )


def test_kedi_docstring_blank_lines_single_quotes_and_empty_body_edges() -> None:
    source = """import kedi

@kedi.query
def ask(name: str) -> str:
    '''kedi

    >> Hello <name>. [answer]

    = <answer>
    '''
    ...
"""

    ranges = kedi_docstrings.compute_kedi_docstring_ranges(source)

    assert ranges[0]["text"] == ">> Hello <name>. [answer]\n\n= <answer>"
    assert (
        kedi_docstrings.compute_kedi_docstring_ranges("def empty():\n    '''kedi'''\n    ...\n")
        == []
    )
    assert (
        kedi_docstrings.compute_kedi_docstring_ranges("def plain():\n    '''not kedi'''\n    ...\n")
        == []
    )
    assert kedi_docstrings.compute_kedi_docstring_ranges("def numeric():\n    1\n    ...\n") == []
    assert kedi_docstrings.compute_kedi_docstring_ranges("def bare():\n    pass\n") == []


def test_kedi_docstring_external_symbols_cover_all_python_parameter_shapes() -> None:
    source = """import kedi

@kedi.query
def ask(first: str, /, second: str, *items: str, option: str, **rest: str) -> str:
    \"\"\"kedi
    >> <first> <second> <items> <option> <rest>. [answer]
    = <answer>
    \"\"\"
    ...
"""

    refs = kedi_docstrings.compute_kedi_docstring_references(
        source,
        "file:///tmp/api.py",
        "/tmp/api.py",
        position_of(source, "items>"),
        include_declaration=False,
    )
    definition = kedi_docstrings.compute_kedi_docstring_definition(
        source,
        "file:///tmp/api.py",
        "/tmp/api.py",
        position_of(source, "rest>"),
    )

    assert refs
    assert all(ref["range"]["start"]["line"] == 5 for ref in refs)
    assert definition[0]["range"]["start"]["line"] == 3


def test_kedi_docstring_internal_definition_paths(monkeypatch) -> None:
    uri = "file:///tmp/api.py"
    output_pos = position_of(PY_SOURCE, "= `output", offset=4)
    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_definition",
        lambda source, uri, source_path, pos: lsp.Location(
            uri=uri,
            range=lsp.Range(
                start=lsp.Position(line=0, character=12),
                end=lsp.Position(line=0, character=18),
            ),
        ),
    )

    mapped_definition = kedi_docstrings.compute_kedi_docstring_definition(
        PY_SOURCE,
        uri,
        "/tmp/api.py",
        output_pos,
    )
    assert mapped_definition
    assert mapped_definition[0]["range"]["start"]["line"] == 12

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_definition",
        lambda source, uri, source_path, pos: lsp.Location(
            uri="file:///tmp/other.kedi",
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=1),
            ),
        ),
    )
    unmapped_definition = kedi_docstrings.compute_kedi_docstring_definition(
        PY_SOURCE,
        uri,
        "/tmp/api.py",
        output_pos,
    )
    assert unmapped_definition[0]["uri"] == "file:///tmp/other.kedi"


def test_kedi_docstring_monkeypatched_feature_fallbacks(monkeypatch) -> None:
    bad_range = lsp.Range(
        start=lsp.Position(line=99, character=0),
        end=lsp.Position(line=99, character=1),
    )
    diagnostic = SimpleNamespace(
        range=bad_range,
        message="bad",
        severity=lsp.DiagnosticSeverity.Error,
    )
    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_diagnostics",
        lambda source, source_path: [diagnostic],
    )
    assert kedi_docstrings.compute_kedi_docstring_diagnostics(PY_SOURCE, "/tmp/api.py") == []

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_hover",
        lambda source, source_path, pos: None,
    )
    assert (
        kedi_docstrings.compute_kedi_docstring_hover(
            PY_SOURCE,
            "/tmp/api.py",
            position_of(PY_SOURCE, "template>"),
        )
        is None
    )

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_hover",
        lambda source, source_path, pos: lsp.Hover(contents="plain hover", range=bad_range),
    )
    hover = kedi_docstrings.compute_kedi_docstring_hover(
        PY_SOURCE,
        "/tmp/api.py",
        position_of(PY_SOURCE, "template>"),
    )
    assert hover == {"contents": {"kind": "markdown", "value": "plain hover"}}

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_hover",
        lambda source, source_path, pos: lsp.Hover(contents="plain hover", range=None),
    )
    hover_without_range = kedi_docstrings.compute_kedi_docstring_hover(
        PY_SOURCE,
        "/tmp/api.py",
        position_of(PY_SOURCE, "template>"),
    )
    assert hover_without_range == {"contents": {"kind": "markdown", "value": "plain hover"}}

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_references",
        lambda source, uri, source_path, pos, include_declaration: [
            lsp.Location(uri="file:///tmp/other.kedi", range=bad_range)
        ],
    )
    refs = kedi_docstrings.compute_kedi_docstring_references(
        PY_SOURCE,
        "file:///tmp/api.py",
        "/tmp/api.py",
        position_of(PY_SOURCE, "output`"),
        include_declaration=False,
    )
    assert refs[0]["uri"] == "file:///tmp/other.kedi"

    monkeypatch.setattr(
        kedi_docstrings.features,
        "compute_semantic_tokens_full",
        lambda source, source_path: lsp.SemanticTokens(data=[99, 0, 1, 1, 0]),
    )
    assert kedi_docstrings.compute_kedi_docstring_semantic_tokens(PY_SOURCE) == {"data": []}


def test_kedi_docstring_private_mapping_helpers_cover_boundaries(monkeypatch) -> None:
    doc = next(iter(kedi_docstrings._iter_kedi_docstrings(PY_SOURCE)))

    function = ast_parse_function("def f():\n    pass\n")
    assert (
        kedi_docstrings._extract_kedi_docstring(
            "",
            function,
            __import__("ast").Constant(1),
        )
        is None
    )
    assert (
        kedi_docstrings._extract_kedi_docstring(
            "",
            function,
            __import__("ast").Constant("kedi\n= ok"),
        )
        is None
    )
    single_quote = ast_parse_function("def f():\n    'kedi\\n= ok'\n")
    assert (
        kedi_docstrings._extract_kedi_docstring(
            "def f():\n    'kedi\\n= ok'\n",
            single_quote,
            single_quote.body[0].value,  # type: ignore[arg-type]
        )
        is None
    )
    triple_quote = ast_parse_function('def f():\n    """kedi\n    = ok\n    """\n')
    assert triple_quote.body
    monkeypatch.setattr(kedi_docstrings, "_find_header_line", lambda lines: None)
    assert (
        kedi_docstrings._extract_kedi_docstring(
            'def f():\n    """kedi\n    = ok\n    """\n',
            triple_quote,
            triple_quote.body[0].value,  # type: ignore[arg-type]
        )
        is None
    )
    monkeypatch.undo()
    monkeypatch.setattr(kedi_docstrings, "_dedent_line", lambda line, indent: "")
    assert (
        kedi_docstrings._extract_kedi_docstring(
            'def f():\n    """kedi\n    = ok\n    """\n',
            triple_quote,
            triple_quote.body[0].value,  # type: ignore[arg-type]
        )
        is None
    )
    assert kedi_docstrings._find_header_line(["not kedi", "kedi"]) is None
    assert kedi_docstrings._find_header_line(["", "  ", "kedi"]) == 2
    assert kedi_docstrings._find_header_line([]) is None
    assert kedi_docstrings._dedent_line("x", 4) == ""
    assert kedi_docstrings._map_position_to_kedi(doc, lsp.Position(line=12, character=0)) is None
    assert (
        kedi_docstrings._map_range_to_source(
            doc,
            lsp.Range(
                start=lsp.Position(line=99, character=0),
                end=lsp.Position(line=99, character=1),
            ),
        )
        is None
    )
    assert kedi_docstrings._map_position_to_source(doc, lsp.Position(line=99, character=0)) is None
    assert kedi_docstrings._map_position_to_source(doc, lsp.Position(line=0, character=999)) is None
    assert (
        kedi_docstrings._map_location_to_source(
            doc,
            lsp.Location(
                uri="file:///tmp/api.py",
                range=lsp.Range(
                    start=lsp.Position(line=99, character=0),
                    end=lsp.Position(line=99, character=1),
                ),
            ),
            "file:///tmp/api.py",
        )
        is None
    )
    assert (
        kedi_docstrings._external_symbol_location(
            doc,
            "file:///tmp/api.py",
            lsp.Position(line=0, character=0),
        )
        is None
    )
    assert (
        kedi_docstrings._external_symbol_location(
            doc,
            "file:///tmp/api.py",
            position_of(doc.kedi_source, "Extract"),
        )
        is None
    )
    assert (
        kedi_docstrings._external_symbol_references(
            doc,
            "file:///tmp/api.py",
            position_of(doc.kedi_source, "Extract"),
            include_declaration=True,
        )
        == []
    )
    synthetic = kedi_docstrings._KediDocstring(
        source_range=doc.source_range,
        kedi_source="<template>",
        line_maps=(),
        external_symbols=doc.external_symbols,
    )
    assert (
        kedi_docstrings._external_symbol_references(
            synthetic,
            "file:///tmp/api.py",
            lsp.Position(line=0, character=2),
            include_declaration=False,
        )
        == []
    )
    assert kedi_docstrings._word_at("", lsp.Position(line=0, character=0)) is None
    assert kedi_docstrings._word_at("abc", lsp.Position(line=4, character=0)) is None
    assert kedi_docstrings._word_at(" abc", lsp.Position(line=0, character=0)) is None
    assert kedi_docstrings._word_at("", lsp.Position(line=0, character=99)) is None
    assert kedi_docstrings._word_at("\n", lsp.Position(line=0, character=0)) is None
    assert kedi_docstrings._word_at("abc", lsp.Position(line=0, character=2)) == "abc"


def ast_parse_function(source: str):
    import ast

    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node
