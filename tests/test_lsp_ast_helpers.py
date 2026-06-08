from __future__ import annotations

from kedi.lang.ast import AssignStmt, Span
from kedi.lsp import ast_helpers


def test_ast_helpers_cover_assignment_and_docstring_edges() -> None:
    assert ast_helpers.assignment_name_span(AssignStmt(name="missing")) is None

    loc = Span(
        start_byte=0,
        end_byte=10,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=11,
        source_path="mod.kedi",
    )
    span = ast_helpers.assignment_name_span(AssignStmt(name="value", loc=loc))
    assert span is not None
    assert span.start_col == 2

    assert ast_helpers.is_public_export_name("public")
    assert not ast_helpers.is_public_export_name("_private")

    assert ast_helpers.extract_module_docstring("") is None
    assert ast_helpers.extract_module_docstring("###\n###\n") is None
    assert ast_helpers.extract_module_docstring("# one\n# two\n= ok\n") == "one\ntwo"
