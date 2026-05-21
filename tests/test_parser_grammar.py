"""Tests for the tree-sitter-backed parser pipeline (Phase 6).

These are NEW tests added during the refactor — they verify the parts
of the public API that the pre-refactor parser did not expose:

  - ``Span`` is populated on every AST node.
  - Structured ``TypeExpr`` is produced (``TypeRef`` / ``TypeApply`` /
    ``TypeUnion`` / ``TypePython``), not just opaque strings.
  - ``parse_program`` returns ``ProgramWithValidation`` whenever
    ``@test:`` or ``@eval:`` blocks are present.
  - ``KediParseError`` is a ``ValueError`` subclass and carries
    ``KediDiagnostic`` records with source locations.

The pre-existing ``test_kedi_lang_*`` suites still cover the
source-in / behavior-out side; this file complements them by asserting
on the structured AST shape itself.
"""

from __future__ import annotations

import pytest

from kedi.lang import (
    KediDiagnostic,
    KediParseError,
    Program,
    ProgramWithValidation,
    Span,
    TypeApply,
    TypePython,
    TypeRef,
    TypeUnion,
    parse_program,
)
from kedi.lang.ast import (
    AssignStmt,
    OutputSegment,
    ReturnBlockStmt,
    TextSegment,
)

# ---------------------------------------------------------------------
# Span propagation
# ---------------------------------------------------------------------


def test_span_populated_on_root():
    prog = parse_program("= ok\n", source_path="example.kedi")
    assert isinstance(prog.loc, Span)
    assert prog.loc.start_line == 1
    assert prog.loc.start_col == 1
    assert prog.loc.source_path == "example.kedi"


def test_span_propagated_to_procedures_and_body():
    src = "@hello(name):\n  Hi, [reply: str]\n  = <reply>\n"
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.loc is not None
    assert proc.loc.start_line == 1
    # Procedure body is the indented block, starting at line 2.
    assert proc.body[0].loc is not None
    assert proc.body[0].loc.start_line == 2
    assert proc.body[1].loc is not None
    assert proc.body[1].loc.start_line == 3


def test_span_source_path_forwarded_to_all_nodes():
    prog = parse_program("@p():\n  = ok\n", source_path="/tmp/example.kedi")
    proc = prog.procedures[0]
    assert proc.loc.source_path == "/tmp/example.kedi"
    assert proc.body[0].loc.source_path == "/tmp/example.kedi"


# ---------------------------------------------------------------------
# Structured TypeExpr
# ---------------------------------------------------------------------


def test_type_ref_for_bare_identifier():
    prog = parse_program("@p(x: int) -> str:\n  = ok\n")
    proc = prog.procedures[0]
    assert isinstance(proc.return_type_expr, TypeRef)
    assert proc.return_type_expr.name == "str"
    assert isinstance(proc.param_type_exprs["x"], TypeRef)
    assert proc.param_type_exprs["x"].name == "int"


def test_type_apply_for_generics():
    prog = parse_program("@xs(items: list[int]) -> dict[str, int]:\n  = ok\n")
    proc = prog.procedures[0]
    params = proc.param_type_exprs
    assert isinstance(params["items"], TypeApply)
    assert params["items"].name == "list"
    assert isinstance(params["items"].args[0], TypeRef)
    assert params["items"].args[0].name == "int"

    ret = proc.return_type_expr
    assert isinstance(ret, TypeApply)
    assert ret.name == "dict"
    assert len(ret.args) == 2
    assert isinstance(ret.args[0], TypeRef) and ret.args[0].name == "str"
    assert isinstance(ret.args[1], TypeRef) and ret.args[1].name == "int"


def test_type_apply_nested():
    prog = parse_program("@p(g: list[list[int]]):\n  = ok\n")
    proc = prog.procedures[0]
    g = proc.param_type_exprs["g"]
    assert isinstance(g, TypeApply) and g.name == "list"
    inner = g.args[0]
    assert isinstance(inner, TypeApply) and inner.name == "list"
    assert isinstance(inner.args[0], TypeRef) and inner.args[0].name == "int"


def test_type_union():
    prog = parse_program("@p(v: int | str | float):\n  = ok\n")
    proc = prog.procedures[0]
    v = proc.param_type_exprs["v"]
    # The union is left-associative: ((int | str) | float).
    assert isinstance(v, TypeUnion)
    assert isinstance(v.right, TypeRef) and v.right.name == "float"
    assert isinstance(v.left, TypeUnion)
    assert isinstance(v.left.left, TypeRef) and v.left.left.name == "int"
    assert isinstance(v.left.right, TypeRef) and v.left.right.name == "str"


def test_type_python_carries_raw_code():
    prog = parse_program("[x: `Optional[int]`] = `42`\n")
    stmt = prog.toplevel_body[0]
    assert isinstance(stmt, AssignStmt)
    assert isinstance(stmt.type_expr, TypePython)
    assert stmt.type_expr.code == "Optional[int]"


def test_legacy_string_annotation_still_populated():
    """Backward-compat: the string ``type_annotation`` field is filled
    so unmigrated consumers (codegen, optimizer utilities) keep working.
    """
    prog = parse_program("@p(x: list[int]) -> str:\n  = ok\n")
    proc = prog.procedures[0]
    assert proc.return_type_annotation == "str"
    assert proc.param_type_annotations["x"] == "list[int]"


def test_output_segment_carries_type_expr():
    prog = parse_program("@p():\n  Pick a [x: int].\n  = <x>\n")
    template_line = prog.procedures[0].body[0]
    out = next(s for s in template_line.segments if isinstance(s, OutputSegment))
    assert out.name == "x"
    assert isinstance(out.type_expr, TypeRef)
    assert out.type_expr.name == "int"
    # Legacy string still set.
    assert out.type_annotation == "int"


# ---------------------------------------------------------------------
# ProgramWithValidation discrimination
# ---------------------------------------------------------------------


def test_program_returns_plain_program_without_validation():
    prog = parse_program("@p():\n  = ok\n")
    assert isinstance(prog, Program)
    assert not isinstance(prog, ProgramWithValidation)


def test_program_returns_with_validation_when_test_block_present():
    src = '@p():\n  = ok\n\n@test: p:\n  > case: c1:\n    `assert p() == "ok"`\n'
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    assert len(prog.tests) == 1
    assert prog.tests[0].procedure_name == "p"
    assert len(prog.tests[0].cases) == 1


def test_program_returns_with_validation_when_eval_block_present():
    src = (
        "@p(x: int) -> int:\n"
        "  = `x * 2`\n"
        "\n"
        "@eval: p:\n"
        "  > data: cases:\n"
        "    = `[(1, 2), (3, 6)]`\n"
        "  > metric: acc(cases):\n"
        '    = `(1.0, "ok")`\n'
    )
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    assert len(prog.evals) == 1
    suite = prog.evals[0]
    assert suite.procedure_name == "p"
    assert "cases" in suite.datasets
    assert len(suite.metrics) == 1
    assert suite.metrics[0].dataset_name == "cases"


def test_validation_block_kw_is_validated():
    """Walker discriminates ``test`` vs ``eval`` from the kw text and
    rejects everything else (the kw is matched as a plain identifier
    in the grammar to avoid keyword-extraction conflicts with
    procedure names ``@test(...)`` / ``@eval(...)``).
    """
    with pytest.raises(KediParseError):
        parse_program("@bogus: foo:\n  > case: c:\n    `pass`\n")


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_kedi_parse_error_is_value_error():
    """Backward compatibility for ``pytest.raises(ValueError)`` callers."""
    with pytest.raises(ValueError):
        parse_program("[42]\n")  # invalid identifier


def test_diagnostic_has_span_with_line_col():
    try:
        parse_program("~Item(a, a)\n", source_path="dup.kedi")
    except KediParseError as exc:
        assert exc.diagnostics
        d = exc.diagnostics[0]
        assert isinstance(d, KediDiagnostic)
        assert d.span is not None
        assert d.span.start_line >= 1
        assert d.span.start_col >= 1
        assert "duplicate field" in d.message.lower()
    else:
        pytest.fail("expected KediParseError")


def test_unterminated_block_comment_raises():
    with pytest.raises(KediParseError) as info:
        parse_program("###\nbody never closed\n")
    assert any("unterminated" in d.message.lower() for d in info.value.diagnostics)


def test_optimize_empty_body_raises_with_legacy_substring():
    """The diagnostic message must contain the legacy substring so
    callers using ``pytest.raises(ValueError, match="must have an
    indented block")`` keep working.
    """
    src = "@p():\n  > optimize: span:\n\n  = done\n"
    with pytest.raises(KediParseError, match="must have an indented block"):
        parse_program(src)


# ---------------------------------------------------------------------
# Return-statement classification
# ---------------------------------------------------------------------


def test_sole_backtick_return_produces_return_block_stmt():
    """``= \\`code\\``` is a ReturnBlockStmt — preserves the native
    Python value at runtime, so non-string returns without an explicit
    annotation are rejected (legacy invariant the runtime relies on).
    """
    prog = parse_program("@p() -> int:\n  = `42`\n")
    body = prog.procedures[0].body
    assert len(body) == 1
    assert isinstance(body[0], ReturnBlockStmt)
    assert body[0].code == "42"


def test_mixed_return_is_returnstmt_not_block():
    """A return value that mixes text and inline-python segments is a
    ReturnStmt (rendered to string at runtime), not a ReturnBlockStmt.
    The distinction lets the runtime preserve native values for the
    ``= \\`code\\``` form while stringifying everything else.
    """
    from kedi.lang.ast import ReturnStmt

    prog = parse_program("@p(x: int) -> str:\n  = result is <`x`> here\n")
    body = prog.procedures[0].body
    assert isinstance(body[0], ReturnStmt)
    # Segments: text + python_expr_segment + text.
    assert len(body[0].segments) >= 2


# ---------------------------------------------------------------------
# Escape handling
# ---------------------------------------------------------------------


def test_template_text_escapes_decoded():
    prog = parse_program("@p():\n  Use \\<angle\\> \\[brackets\\] \\@at \\= eq\n  = ok\n")
    template_line = prog.procedures[0].body[0]
    text = "".join(s.text for s in template_line.segments if isinstance(s, TextSegment))
    assert "<angle>" in text
    assert "[brackets]" in text
    assert "@at" in text
    assert "= eq" in text


def test_tilde_double_escape_decoded():
    prog = parse_program("= literal ~~Person\n")
    seg = prog.toplevel_return.segments[0]
    assert isinstance(seg, TextSegment)
    assert "~Person" in seg.text


def test_line_continuation_collapses_to_single_space():
    src = "= Start \\\n  middle \\\n  end\n"
    prog = parse_program(src)
    seg = prog.toplevel_return.segments[0]
    assert isinstance(seg, TextSegment)
    # The trailing-WS-rstrip + `\<newline>` + leading-WS-strip rule
    # collapses to single spaces.
    assert seg.text == "Start middle end"
