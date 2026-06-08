"""LSP feature helper edge-case tests."""

from __future__ import annotations

from typing import Any, cast

from kedi.lsp import features
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_at_line


def test_lsp_symbol_resolution_helper_edges(tmp_path, monkeypatch):
    from kedi.lang.ast import (
        AssignBlockStmt,
        AssignStmt,
        AutoSpec,
        CallArg,
        CallSegment,
        EvalDataset,
        EvalMetric,
        EvalSuite,
        InputSegment,
        OptimizeSpan,
        OutputSegment,
        ProcDefStmt,
        ProcedureDef,
        Program,
        ProgramWithValidation,
        ReturnStmt,
        Span,
        TemplateLine,
        TestCase,
        TestSuite,
        TextSegment,
        TypeApply,
        TypeDef,
        TypeField,
        TypeRef,
        TypeUnion,
    )

    type_ref = TypeRef(
        name="Person",
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=7),
    )
    type_apply = TypeApply(
        name="list",
        args=[type_ref],
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=13),
    )
    type_union = TypeUnion(
        left=type_apply,
        right=TypeRef(name="str", loc=make_edge_span(tmp_path)),
        loc=make_edge_span(tmp_path),
    )
    pos = position_at_line(line=0, character=4)
    proc = ProcedureDef(
        name="p",
        params=["x"],
        body=[],
        param_type_annotations={"x": "int"},
        return_type_annotation="str",
        loc=make_edge_span(tmp_path),
    )

    nested_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        nested_arg,
        "__call_segment__",
        CallSegment(name="inner", args=[], loc=make_edge_span(tmp_path)),
    )
    rendered_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="rendered", args=[], loc=make_edge_span(tmp_path))],
    )
    outer_call = CallSegment(
        name="outer", args=[nested_arg, rendered_arg], loc=make_edge_span(tmp_path)
    )
    calls: list[Span] = []
    features._collect_call_refs(outer_call, "inner", calls)
    features._collect_call_refs(outer_call, "rendered", calls)
    assert len(calls) == 2
    inputs: list[Span] = []
    nested_input_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        nested_input_arg,
        "__python_segment__",
        InputSegment(name="needle", loc=make_edge_span(tmp_path)),
    )
    rendered_input_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        rendered_input_arg,
        "__render_segments__",
        [InputSegment(name="needle", loc=make_edge_span(tmp_path))],
    )
    features._collect_input_refs(
        CallSegment(
            name="outer", args=[nested_input_arg, rendered_input_arg], loc=make_edge_span(tmp_path)
        ),
        "needle",
        inputs,
    )
    assert len(inputs) == 2

    refs: list[Span] = []
    features._collect_type_refs_in_expr(type_union, "Person", refs)
    features._collect_type_refs_in_expr(type_apply, "list", refs)
    features._collect_type_refs_in_stmt(
        OutputSegment(name="x", type_expr=type_ref, loc=make_edge_span(tmp_path)), "Person", refs
    )
    features._collect_type_refs_in_type_def(
        TypeDef("Box", [TypeField("item", type_expr=type_ref)]), "Person", refs
    )
    assert refs

    assert features._position_in_proc_name(proc, position_at_line(line=1, character=1)) is False
    assert features._find_local_declaration(None, "x") is None
    local_proc = ProcedureDef(
        "local",
        [],
        body=[
            AssignStmt(name="x", loc=make_edge_span(tmp_path)),
            TemplateLine(
                segments=[OutputSegment(name="y", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            ),
        ],
    )
    assert features._find_local_declaration(local_proc, "x") is not None
    assert features._find_local_declaration(local_proc, "y") is not None
    assert features._find_local_declaration(local_proc, "z") is None

    monkeypatch.setattr(
        features,
        "_param_decls",
        lambda proc: [
            ("miss", make_edge_span(tmp_path, line=1, start_col=20, end_col=25)),
            ("x", make_edge_span(tmp_path)),
        ],
    )
    assert features._resolve_in_procedure(proc, pos, {}, {}, {}) is not None
    monkeypatch.undo()

    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=type_ref,
                right=TypeRef(name="Other", loc=make_edge_span(tmp_path)),
                loc=make_edge_span(tmp_path),
            ),
            pos,
            {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))},
        )
        is not None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(name="list", args=[type_ref], loc=make_edge_span(tmp_path)),
            position_at_line(line=0, character=4),
            {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))},
        )
        is not None
    )
    assert (
        features._resolve_in_stmt(
            AssignBlockStmt(name="x", type_expr=type_ref, loc=make_edge_span(tmp_path)),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))},
            {},
        )
        is not None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(name="x", type_expr=type_ref, loc=make_edge_span(tmp_path)),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))},
            {},
        )
        is not None
    )
    field_ref = features._resolve_in_type_def(
        TypeDef(
            "Person",
            [TypeField("name", loc=make_edge_span(tmp_path, line=1, start_col=9, end_col=13))],
            loc=make_edge_span(tmp_path),
        ),
        position_at_line(line=0, character=8),
        {},
    )
    assert field_ref is not None and field_ref.kind == "type_field"
    assert (
        features._resolve_in_type_def(
            TypeDef(
                "Person",
                [TypeField("name", loc=make_edge_span(tmp_path, line=1, start_col=9, end_col=13))],
                loc=make_edge_span(tmp_path),
            ),
            position_at_line(line=0, character=20),
            {},
        )
        is None
    )
    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=TypeRef(
                    name="Nope", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=24)
                ),
                right=TypeRef(
                    name="Other", loc=make_edge_span(tmp_path, line=1, start_col=30, end_col=35)
                ),
                loc=make_edge_span(tmp_path),
            ),
            pos,
            {},
        )
        is None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(
                name="list",
                args=[
                    TypeRef(
                        name="Other", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25)
                    )
                ],
                loc=make_edge_span(tmp_path),
            ),
            pos,
            {},
        )
        is None
    )
    assert (
        features._resolve_in_stmt(
            AssignBlockStmt(
                name="x",
                type_expr=TypeRef(
                    name="Other", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25)
                ),
                loc=make_edge_span(tmp_path),
            ),
            pos,
            None,
            {},
            {},
            {},
        )
        is None
    )

    resolve_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        resolve_arg, "__call_segment__", InputSegment(name="resolved", loc=make_edge_span(tmp_path))
    )
    resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[resolve_arg], loc=make_edge_span(tmp_path)),
        position_at_line(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert resolved is not None and resolved.name == "resolved"
    rendered_resolve_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        rendered_resolve_arg,
        "__render_segments__",
        [InputSegment(name="rendered", loc=make_edge_span(tmp_path))],
    )
    rendered_resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[rendered_resolve_arg], loc=make_edge_span(tmp_path)),
        position_at_line(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert rendered_resolved is not None and rendered_resolved.name == "rendered"
    missing_inner_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        missing_inner_arg,
        "__call_segment__",
        InputSegment(name="miss", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25)),
    )
    missing_render_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        missing_render_arg,
        "__render_segments__",
        [InputSegment(name="miss", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25))],
    )
    assert (
        features._resolve_in_segment(
            CallSegment(
                name="outer",
                args=[missing_inner_arg, missing_render_arg],
                loc=make_edge_span(tmp_path),
            ),
            position_at_line(line=0, character=10),
            None,
            {},
            {},
            {},
        )
        is None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(
                name="x",
                type_expr=TypeRef(
                    name="Other", loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25)
                ),
                loc=make_edge_span(tmp_path),
            ),
            pos,
            None,
            {},
            {},
            {},
        )
        is None
    )

    program = ProgramWithValidation(
        procedures=[
            ProcedureDef(
                "outer",
                [],
                body=cast(
                    Any,
                    [
                        None,
                        ProcDefStmt(
                            proc=ProcedureDef("inner", [], body=[], loc=make_edge_span(tmp_path)),
                            loc=make_edge_span(tmp_path),
                        ),
                        TemplateLine(segments=[outer_call], loc=make_edge_span(tmp_path)),
                    ],
                ),
                auto_spec=AutoSpec("auto", loc=make_edge_span(tmp_path)),
                optimize_spans=[
                    OptimizeSpan(
                        "prompt",
                        [TemplateLine(loc=make_edge_span(tmp_path))],
                        loc=make_edge_span(tmp_path),
                    )
                ],
                loc=make_edge_span(tmp_path),
            )
        ],
        type_defs=[
            TypeDef(
                "Box",
                [TypeField("item", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            )
        ],
        tests=[
            TestSuite(
                "outer",
                [TestCase("basic", "assert True", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            )
        ],
        evals=[
            EvalSuite(
                "outer",
                datasets={"train": EvalDataset("train", "[]", loc=make_edge_span(tmp_path))},
                test_datasets={
                    "holdout": EvalDataset("holdout", "[]", loc=make_edge_span(tmp_path))
                },
                metrics=[EvalMetric("score", "return 1", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            )
        ],
    )
    setattr(
        program,
        "top_level_statements",
        [
            TemplateLine(
                segments=[InputSegment(name="x", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            ),
            AssignStmt(
                name="x",
                expr=[TextSegment(text="value", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            ),
        ],
    )
    assert features._smallest_enclosing(program, pos) is not None

    ref_prog = Program(
        procedures=[
            ProcedureDef(
                name="target",
                params=[],
                body=[
                    TemplateLine(
                        segments=[CallSegment(name="target", loc=make_edge_span(tmp_path))],
                        loc=make_edge_span(tmp_path),
                    )
                ],
                loc=make_edge_span(tmp_path),
            ),
            ProcedureDef(
                name="typed",
                params=["x"],
                body=[AssignStmt(name="v", type_expr=type_ref, loc=make_edge_span(tmp_path))],
                param_type_exprs={"x": type_ref, "skip": None},
                return_type_expr=type_ref,
                loc=make_edge_span(tmp_path),
            ),
        ],
        toplevel_return=ReturnStmt(
            segments=[CallSegment(name="target", loc=make_edge_span(tmp_path))],
            loc=make_edge_span(tmp_path),
        ),
        toplevel_body=[
            TemplateLine(
                segments=[CallSegment(name="target", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            ),
            AssignStmt(name="v", type_expr=type_ref, loc=make_edge_span(tmp_path)),
        ],
        type_defs=[
            TypeDef("Person", [TypeField("item", type_expr=type_ref)], loc=make_edge_span(tmp_path))
        ],
    )
    duplicate_span = make_edge_span(tmp_path)
    proc_refs = features._find_all_references(
        ref_prog,
        features.SymbolRef("procedure", "target", duplicate_span, None, duplicate_span),
        include_declaration=True,
    )
    type_refs = features._find_all_references(
        ref_prog,
        features.SymbolRef("type", "Person", duplicate_span, None, duplicate_span),
        include_declaration=True,
    )
    assert len(proc_refs) >= 2
    assert len({(s.start_line, s.start_col) for s in type_refs}) == 1
    assert (
        features._find_all_references(
            ref_prog,
            features.SymbolRef("local", "x", None, None, duplicate_span),
            include_declaration=False,
        )
        == []
    )

    proc_symbol = features._proc_symbol(program.procedures[0])
    assert proc_symbol.children
    assert any(child.name == "inner" for child in proc_symbol.children)
    assert any(child.name == "> optimize: prompt" for child in proc_symbol.children)
    no_loc_opt_symbol = features._proc_symbol(
        ProcedureDef(
            name="no_loc_opt",
            params=[],
            optimize_spans=[OptimizeSpan("prompt")],
            loc=make_edge_span(tmp_path),
        )
    )
    assert no_loc_opt_symbol.children == []
    eval_symbol = features._eval_suite_symbol(program.evals[0])
    assert eval_symbol.children is not None
    assert [child.name for child in eval_symbol.children] == [
        "> data: train",
        "> test_data: holdout",
        "> metric: score",
    ]
    test_symbol = features._test_suite_symbol(program.tests[0])
    assert test_symbol.children is not None
    assert test_symbol.children[0].name == "basic"

    assert features._empty_range().start.line == 0
    assert features._resolve_overlaps([]) == []
    assert features._resolve_overlaps([(0, -2, 1, 1, 0), (0, 0, 1, 2, 0)])
    assert (
        features._position_in_range(
            position_at_line(line=2, character=10),
            {"start": {"line": 1, "character": 0}, "end": {"line": 2, "character": 5}},
        )
        is False
    )
