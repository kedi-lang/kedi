"""LSP type rendering and AST description helper edge-case tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kedi.lsp import features
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_at_line


def test_resolve_in_type_def_rejects_missing_or_out_of_range_location() -> None:
    from kedi.lang.ast import Span, TypeDef

    pos = position_at_line(0, 4)
    assert features._resolve_in_type_def(TypeDef("Person", [], loc=None), pos, {}) is None
    assert (
        features._resolve_in_type_def(
            TypeDef("Person", [], loc=Span(0, 6, 1, 1, 1, 7)),
            position_at_line(5, 0),
            {},
        )
        is None
    )


def test_type_expr_to_str_covers_ref_apply_union_and_fallback(tmp_path: Path) -> None:
    from kedi.lang.ast import TypeApply, TypeExpr, TypeRef, TypeUnion

    type_ref = TypeRef(name="Person", loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=7))
    type_apply = TypeApply(
        name="list", args=[type_ref], loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=13)
    )
    type_union = TypeUnion(
        left=type_apply,
        right=TypeRef(name="str", loc=make_edge_span(tmp_path)),
        loc=make_edge_span(tmp_path),
    )
    assert features._type_expr_to_str(None) == "None"
    assert features._type_expr_to_str(type_ref) == "Person"
    assert features._type_expr_to_str(type_apply) == "list[Person]"
    assert features._type_expr_to_str(type_union) == "list[Person] | str"
    assert "TypeExpr" in features._type_expr_to_str(TypeExpr())


def test_prelude_defined_names_skips_broken_and_collects_symbols() -> None:
    assert features._prelude_defined_names(None) == set()
    assert features._prelude_defined_names("class Broken(:\n") == set()
    assert {
        "Model",
        "fn",
        "async_fn",
        "path",
        "alias",
        "x",
        "y",
    } <= features._prelude_defined_names(
        "import os.path as path\n"
        "from typing import Any as alias\n"
        "class Model: pass\n"
        "def fn(): pass\n"
        "async def async_fn(): pass\n"
        "x = y = 1\n"
    )


def test_proc_signature_markdown_variants(tmp_path: Path) -> None:
    from kedi.lang.ast import AutoSpec, OptimizeSpan, ProcedureDef

    proc = ProcedureDef(
        name="p",
        params=["x"],
        body=[],
        param_type_annotations={"x": "int"},
        return_type_annotation="str",
        loc=make_edge_span(tmp_path),
    )
    assert "@p(x: int)" in features._proc_signature_markdown(proc)
    assert "AI-generated" in features._proc_signature_markdown(
        ProcedureDef(
            name="auto",
            params=[],
            auto_spec=AutoSpec("make it", loc=make_edge_span(tmp_path)),
            loc=make_edge_span(tmp_path),
        )
    )
    assert "optimization spans" in features._proc_signature_markdown(
        ProcedureDef(
            name="opt",
            params=[],
            optimize_spans=[OptimizeSpan("prompt", loc=make_edge_span(tmp_path))],
            loc=make_edge_span(tmp_path),
        )
    )
    assert "doc" in features._proc_signature_markdown(
        ProcedureDef(name="doc", params=[], docstring="doc", loc=make_edge_span(tmp_path))
    )


def test_describe_ast_node_covers_supported_nodes(tmp_path: Path) -> None:
    from kedi.lang.ast import (
        AssignBlockStmt,
        AssignStmt,
        AutoSpec,
        BacktickLineStmt,
        CallSegment,
        EvalDataset,
        EvalMetric,
        EvalSuite,
        InputSegment,
        OptimizeSpan,
        OutputSegment,
        ProcedureDef,
        PythonBlock,
        PythonExprSegment,
        ReturnBlockStmt,
        ReturnStmt,
        Stmt,
        TemplateLine,
        TestSuite,
        TextSegment,
        TypeDef,
        TypeField,
    )

    proc = ProcedureDef(name="p", params=["x"], body=[], loc=make_edge_span(tmp_path))
    described_nodes = [
        proc,
        TypeDef(
            "Person",
            [TypeField("name", loc=make_edge_span(tmp_path))],
            loc=make_edge_span(tmp_path),
        ),
        TypeField("name", default_expr="'Ada'", loc=make_edge_span(tmp_path)),
        TemplateLine(
            segments=[TextSegment(text="hello", loc=make_edge_span(tmp_path))],
            loc=make_edge_span(tmp_path),
        ),
        AssignStmt(name="x", type_annotation="int", loc=make_edge_span(tmp_path)),
        AssignBlockStmt(name="x", type_annotation="int", loc=make_edge_span(tmp_path)),
        ReturnStmt(loc=make_edge_span(tmp_path)),
        ReturnBlockStmt(loc=make_edge_span(tmp_path)),
        BacktickLineStmt(code="x = 1", loc=make_edge_span(tmp_path)),
        PythonBlock(code="x = 1", loc=make_edge_span(tmp_path)),
        AutoSpec("make it", loc=make_edge_span(tmp_path)),
        OptimizeSpan("prompt", loc=make_edge_span(tmp_path)),
        TextSegment(text="x" * 50, loc=make_edge_span(tmp_path)),
        InputSegment(name="x", loc=make_edge_span(tmp_path)),
        OutputSegment(name="x", type_annotation="int", loc=make_edge_span(tmp_path)),
        CallSegment(name="p", loc=make_edge_span(tmp_path)),
        PythonExprSegment(code="x", loc=make_edge_span(tmp_path)),
        EvalSuite("p", loc=make_edge_span(tmp_path)),
        TestSuite("p", loc=make_edge_span(tmp_path)),
        EvalDataset("train", "[]", loc=make_edge_span(tmp_path)),
        EvalMetric("score", "return 1", loc=make_edge_span(tmp_path)),
    ]
    assert all(features._describe_ast_node(node) for node in described_nodes)
    assert features._describe_ast_node(Stmt(loc=make_edge_span(tmp_path))) is None


def test_generic_hover_returns_none_without_actionable_nodes(tmp_path: Path) -> None:
    from kedi.lang.ast import Program, Stmt

    pos = position_at_line(0, 4)
    assert features._generic_hover(Program(), pos) is None
    assert (
        features._generic_hover(
            SimpleNamespace(procedures=[], type_defs=[], top_level_statements=[]), pos
        )
        is None
    )
    assert (
        features._generic_hover(
            SimpleNamespace(
                procedures=[],
                type_defs=[],
                top_level_statements=[Stmt(loc=make_edge_span(tmp_path))],
            ),
            pos,
        )
        is None
    )
