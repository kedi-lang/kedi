"""Kedi #214 (6.14): parameter declaration spans in procedure headers.

The header is parsed into top-level comma segments, not substring
searched: a default may contain another parameter's name, and a type
annotation may contain commas. The declaration span must be the real
parameter position, so rename never rewrites a name inside a default.
"""

from __future__ import annotations

from kedi.lsp.coordinates import source_coordinates
from kedi.lsp.features.scope import _all_procedures, _param_decls, _split_header_param_segments
from kedi.lsp.parsing import parse_program
from tests.lsp_helpers import definition_at, references_at, rename_at

URI = "file:///tmp/x.kedi"
PATH = "/tmp/x.kedi"


def _decls(source: str) -> dict[str, list[tuple[str, int]]]:
    prog = parse_program(source, source_path=PATH)
    out: dict[str, list[tuple[str, int]]] = {}
    with source_coordinates(source, PATH):
        for proc in _all_procedures(prog):
            header = source[proc.loc.start_byte :].split("\n")[0]
            out[proc.name] = [(name, span.start_byte) for name, span in _param_decls(proc)]
            for name, span in _param_decls(proc):
                # The span decodes to the name and is followed by a real
                # parameter continuation (':', '=', ',', ')'), never by a
                # backtick (inside another parameter's default).
                assert source[span.start_byte : span.end_byte] == name
                assert header[span.end_col - 1] in (":", "=", ",", ")")
    return out


def test_defaults_containing_parameter_names() -> None:
    src = "@p(x=`y`, y=`1`):\n  = <y>\n"
    decls = _decls(src)
    assert [name for name, _ in decls["p"]] == ["x", "y"]
    assert src[decls["p"][1][1] :].startswith("y=`1`")


def test_nested_procedure_defaults_containing_parameter_names() -> None:
    src = "@p(x=`y`, y=`1`):\n  @inner(a=`b`, b=`2`):\n    = <`a`>\n  = <p(1, 2)>\n"
    decls = _decls(src)
    assert [name for name, _ in decls["inner"]] == ["a", "b"]
    assert src[decls["inner"][1][1] :].startswith("b=`2`")


def test_bracketed_annotations_with_commas() -> None:
    src = "@p(a: dict[str, int], b: list[int]):\n  = <`a`>\n"
    decls = _decls(src)
    assert [name for name, _ in decls["p"]] == ["a", "b"]


def test_default_call_with_commas() -> None:
    src = "@p(x=`f(1, 2)`, y=`1`):\n  = <y>\n"
    decls = _decls(src)
    assert [name for name, _ in decls["p"]] == ["x", "y"]
    assert src[decls["p"][1][1] :].startswith("y=`1`")


def test_rename_never_rewrites_a_name_inside_a_default() -> None:
    src = "@p(x=`y`, y=`1`):\n  = <y>\n"
    edit = rename_at(src, URI, PATH, "y", "z", occurrence=2)
    assert edit is not None
    changes = edit.changes or {}
    ranges = sorted((e.range.start.line, e.range.start.character) for e in changes[URI])
    # The declaration and the return use only; the `` `y` `` default
    # (line 0, character 6) is untouched.
    assert ranges == [(0, 10), (1, 5)]


def test_definition_of_use_resolves_to_the_declaration_not_the_default() -> None:
    src = "@p(x=`y`, y=`1`):\n  = <y>\n"
    loc = definition_at(src, URI, PATH, "y", occurrence=3)
    assert loc is not None
    assert loc.range.start.line == 0
    assert loc.range.start.character == src.split("\n")[0].index("y=`1`")


def test_default_reading_top_level_binding_still_resolves() -> None:
    src = "[outer] = `1`\n\n@p(x=`outer`):\n  = <x>\n"
    refs = references_at(src, URI, PATH, "outer", True, occurrence=1)
    assert sorted((r.range.start.line, r.range.start.character) for r in refs) == [
        (0, 1),
        (2, 6),
    ]


def test_nested_default_reading_outer_parameter_still_resolves() -> None:
    src = "@outer(y):\n  @inner(x=`y`):\n    = <x>\n  = <inner()>\n"
    refs = references_at(src, URI, PATH, "y", True, occurrence=1)
    assert sorted((r.range.start.line, r.range.start.character) for r in refs) == [
        (0, 7),
        (1, 12),
    ]


def test_header_segment_splitter() -> None:
    header = "@p(a: dict[str, int], b=`f(1, 2)`, c):"
    segments = _split_header_param_segments(header, header.find("(") + 1)
    assert [header[s:e] for s, e in segments] == [
        "a: dict[str, int]",
        " b=`f(1, 2)`",
        " c",
    ]
