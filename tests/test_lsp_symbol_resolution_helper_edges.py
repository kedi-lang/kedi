"""LSP symbol resolution and reference-collection helper edge-case tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from lsprotocol import types as lsp

from kedi.lang.ast import ProcedureDef, TypeApply, TypeRef, TypeUnion
from kedi.lsp import features
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_at_line


@pytest.fixture
def resolution_pos() -> lsp.Position:
    return position_at_line(line=0, character=4)


@pytest.fixture
def person_type_ref(tmp_path: Path) -> TypeRef:
    return TypeRef(
        name="Person",
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=7),
    )


@pytest.fixture
def list_person_apply(tmp_path: Path, person_type_ref: TypeRef) -> TypeApply:
    return TypeApply(
        name="list",
        args=[person_type_ref],
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=13),
    )


@pytest.fixture
def list_person_union(tmp_path: Path, list_person_apply: TypeApply) -> TypeUnion:
    return TypeUnion(
        left=list_person_apply,
        right=TypeRef(name="str", loc=make_edge_span(tmp_path)),
        loc=make_edge_span(tmp_path),
    )


@pytest.fixture
def sample_proc(tmp_path: Path) -> ProcedureDef:
    return ProcedureDef(
        name="p",
        params=["x"],
        body=[],
        param_type_annotations={"x": "int"},
        return_type_annotation="str",
        loc=make_edge_span(tmp_path),
    )


def test_collect_call_and_input_refs_follow_nested_args(tmp_path: Path) -> None:
    from kedi.lang.ast import CallArg, CallSegment, InputSegment, Span, TextSegment

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
    inputs: list[Span] = []
    features._collect_input_refs(
        CallSegment(
            name="outer",
            args=[nested_input_arg, rendered_input_arg],
            loc=make_edge_span(tmp_path),
        ),
        "needle",
        inputs,
    )
    features._collect_input_refs(InputSegment(name="other"), "needle", inputs)
    features._collect_input_refs(TextSegment(text="plain"), "needle", inputs)
    assert len(inputs) == 2


def test_collect_type_refs_in_expr_stmt_and_type_def(
    tmp_path: Path,
    person_type_ref: TypeRef,
    list_person_apply: TypeApply,
    list_person_union: TypeUnion,
) -> None:
    from kedi.lang.ast import OutputSegment, Span, TypeDef, TypeField

    refs: list[Span] = []
    features._collect_type_refs_in_expr(list_person_union, "Person", refs)
    features._collect_type_refs_in_expr(list_person_apply, "list", refs)
    features._collect_type_refs_in_stmt(
        OutputSegment(name="x", type_expr=person_type_ref, loc=make_edge_span(tmp_path)),
        "Person",
        refs,
    )
    features._collect_type_refs_in_type_def(
        TypeDef("Box", [TypeField("item", type_expr=person_type_ref)]), "Person", refs
    )
    assert refs


def test_find_local_declaration_in_procedure_body(tmp_path: Path) -> None:
    from kedi.lang.ast import AssignStmt, OutputSegment, ProcedureDef, TemplateLine

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


def test_resolve_in_procedure_uses_param_decls(
    tmp_path: Path,
    sample_proc: ProcedureDef,
    resolution_pos: lsp.Position,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        features._position_in_proc_name(sample_proc, position_at_line(line=1, character=1)) is False
    )
    monkeypatch.setattr(
        features,
        "_param_decls",
        lambda proc: [
            ("miss", make_edge_span(tmp_path, line=1, start_col=20, end_col=25)),
            ("x", make_edge_span(tmp_path)),
        ],
    )
    from kedi.lang.ast import Program

    assert (
        features._resolve_in_procedure(Program(), sample_proc, resolution_pos, {}, {}, {})
        is not None
    )


def test_resolve_in_type_expr_and_stmt(
    tmp_path: Path,
    person_type_ref: TypeRef,
    resolution_pos: lsp.Position,
) -> None:
    from kedi.lang.ast import (
        AssignBlockStmt,
        OutputSegment,
        TypeApply,
        TypeDef,
        TypeRef,
        TypeUnion,
    )

    person_def = {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))}
    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=person_type_ref,
                right=TypeRef(name="Other", loc=make_edge_span(tmp_path)),
                loc=make_edge_span(tmp_path),
            ),
            resolution_pos,
            person_def,
        )
        is not None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(name="list", args=[person_type_ref], loc=make_edge_span(tmp_path)),
            position_at_line(line=0, character=4),
            person_def,
        )
        is not None
    )
    from kedi.lang.ast import Program

    assert (
        features._resolve_in_stmt(
            Program(),
            AssignBlockStmt(name="x", type_expr=person_type_ref, loc=make_edge_span(tmp_path)),
            resolution_pos,
            None,
            {},
            person_def,
            {},
        )
        is not None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(name="x", type_expr=person_type_ref, loc=make_edge_span(tmp_path)),
            resolution_pos,
            None,
            {},
            person_def,
            {},
        )
        is not None
    )


def test_resolve_in_type_def_field(tmp_path: Path) -> None:
    from kedi.lang.ast import TypeDef, TypeField

    field_ref = features._resolve_in_type_def(
        TypeDef(
            "Person",
            [
                TypeField(
                    "name",
                    loc=make_edge_span(tmp_path, line=1, start_col=9, end_col=13),
                )
            ],
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
                [
                    TypeField(
                        "name",
                        loc=make_edge_span(tmp_path, line=1, start_col=9, end_col=13),
                    )
                ],
                loc=make_edge_span(tmp_path),
            ),
            position_at_line(line=0, character=20),
            {},
        )
        is None
    )


def test_resolve_in_type_expr_and_stmt_miss_unknown_types(
    tmp_path: Path,
    resolution_pos: lsp.Position,
) -> None:
    from kedi.lang.ast import AssignBlockStmt, TypeApply, TypeRef, TypeUnion

    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=TypeRef(
                    name="Nope",
                    loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=24),
                ),
                right=TypeRef(
                    name="Other",
                    loc=make_edge_span(tmp_path, line=1, start_col=30, end_col=35),
                ),
                loc=make_edge_span(tmp_path),
            ),
            resolution_pos,
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
                        name="Other",
                        loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25),
                    )
                ],
                loc=make_edge_span(tmp_path),
            ),
            resolution_pos,
            {},
        )
        is None
    )
    from kedi.lang.ast import Program

    assert (
        features._resolve_in_stmt(
            Program(),
            AssignBlockStmt(
                name="x",
                type_expr=TypeRef(
                    name="Other",
                    loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25),
                ),
                loc=make_edge_span(tmp_path),
            ),
            resolution_pos,
            None,
            {},
            {},
            {},
        )
        is None
    )


def test_resolve_in_segment_nested_inputs(tmp_path: Path) -> None:
    from kedi.lang.ast import CallArg, CallSegment, InputSegment, OutputSegment, TypeRef

    resolve_arg = CallArg(loc=make_edge_span(tmp_path))
    setattr(
        resolve_arg,
        "__call_segment__",
        InputSegment(name="resolved", loc=make_edge_span(tmp_path)),
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
        [
            InputSegment(
                name="miss",
                loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25),
            )
        ],
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
                    name="Other",
                    loc=make_edge_span(tmp_path, line=1, start_col=20, end_col=25),
                ),
                loc=make_edge_span(tmp_path),
            ),
            position_at_line(line=0, character=4),
            None,
            {},
            {},
            {},
        )
        is None
    )


def test_smallest_enclosing_validation_program(
    tmp_path: Path, resolution_pos: lsp.Position
) -> None:
    from kedi.lang.ast import (
        AssignStmt,
        AutoSpec,
        CallArg,
        CallSegment,
        EvalDataset,
        EvalMetric,
        EvalSuite,
        InputSegment,
        OptimizeSpan,
        ProcDefStmt,
        ProcedureDef,
        ProgramWithValidation,
        TemplateLine,
        TestCase,
        TestSuite,
        TextSegment,
        TypeDef,
        TypeField,
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
    assert features._smallest_enclosing(program, resolution_pos) is not None


def test_find_all_references_for_procedures_and_types(
    tmp_path: Path,
    person_type_ref: TypeRef,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.lang.ast import (
        AssignStmt,
        CallSegment,
        ProcedureDef,
        ProfileDefStmt,
        Program,
        ReturnStmt,
        TemplateLine,
        TestSuite,
        TypeDef,
        TypeField,
    )

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
                body=[
                    AssignStmt(
                        name="v",
                        type_expr=person_type_ref,
                        loc=make_edge_span(tmp_path),
                    )
                ],
                param_type_exprs={"x": person_type_ref, "skip": None},
                return_type_expr=person_type_ref,
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
            AssignStmt(name="v", type_expr=person_type_ref, loc=make_edge_span(tmp_path)),
        ],
        type_defs=[
            TypeDef(
                "Person",
                [TypeField("item", type_expr=person_type_ref)],
                loc=make_edge_span(tmp_path),
            )
        ],
    )
    duplicate_span = make_edge_span(tmp_path)
    monkeypatch.setattr(
        features,
        "_profile_use_entry_spans",
        lambda prog, profile: [("other", duplicate_span), ("target", duplicate_span)],
    )
    monkeypatch.setattr(
        features,
        "_validation_suites",
        lambda prog: [
            TestSuite("other"),
            TestSuite("target"),
            TestSuite("target", loc=duplicate_span),
        ],
    )
    monkeypatch.setattr(
        features,
        "_validation_procedure_span",
        lambda prog, suite: suite.loc,
    )
    ref_prog.profile_defs["profile"] = ProfileDefStmt(name="profile")
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


def test_collect_lexical_input_refs_recurses_through_call_arguments(tmp_path: Path) -> None:
    from kedi.lang.ast import (
        AssignStmt,
        CallSegment,
        InputSegment,
        LiteralArg,
        ProcedureDef,
        TextSegment,
    )

    declaration = make_edge_span(tmp_path)
    owner = ProcedureDef(
        name="owner",
        params=[],
        body=[AssignStmt(name="value", loc=declaration)],
        loc=make_edge_span(tmp_path),
    )
    segment = CallSegment(
        name="outer",
        args=[
            LiteralArg(render_segments=[InputSegment(name="value", loc=make_edge_span(tmp_path))])
        ],
        loc=make_edge_span(tmp_path),
    )
    refs = []
    declaration_span = features._find_local_declaration(owner, "value")
    assert declaration_span is not None
    features._collect_lexical_input_refs(
        segment,
        "value",
        owner,
        (),
        owner,
        declaration_span,
        refs,
    )
    assert refs
    features._collect_lexical_input_refs(
        InputSegment(name="value", loc=make_edge_span(tmp_path)),
        "value",
        owner,
        (),
        ProcedureDef(name="other", params=[]),
        declaration_span,
        refs,
    )
    assert len(refs) == 1
    features._collect_lexical_input_refs(
        TextSegment(text="literal", loc=make_edge_span(tmp_path)),
        "value",
        owner,
        (),
        owner,
        declaration_span,
        refs,
    )
    assert len(refs) == 1


def test_scope_path_and_nested_capture_resolution_edges() -> None:
    from kedi.lang import parse_program
    from kedi.lang.ast import AssignStmt, InputSegment, OutputSegment, ReassignStmt, Stmt

    source = (
        "[outer] = root\n"
        "> if: `True`:\n"
        "  [branch] = yes\n"
        "> loop [item]: `[1]`:\n"
        "  [loop_local] = yes\n"
        "> loop: `False`:\n"
        "  [conditional] = yes\n"
        "@parent(param: str) -> str:\n"
        "  [captured] = local\n"
        "  @child() -> str:\n"
        "    = <captured>-<param>\n"
        "  = <child()>\n"
    )
    program = parse_program(source, source_path="/tmp/scope-path.kedi")

    assert features._direct_statement_declarations(Stmt(loc=None), "x", None) == []
    assert features._direct_statement_declarations(AssignStmt(name="x", loc=None), "x", None) == []
    assert (
        features._direct_statement_declarations(
            AssignStmt(expr=[OutputSegment(name="x", loc=None)]), "x", None
        )
        == []
    )
    assert features._declaration_in_scope_path([Stmt(loc=None)], "x", 1, None) is None
    ended = Stmt(loc=make_edge_span(Path("/tmp"), line=1, start_col=1, end_col=2))
    another_ended = Stmt(loc=make_edge_span(Path("/tmp"), line=2, start_col=1, end_col=2))
    assert features._declaration_in_scope_path([ended, another_ended], "x", 100, None) is None

    missing_target = parse_program(
        "[missing] := value\n",
        source_path="/tmp/missing-reassign.kedi",
    ).toplevel_body[0]
    missing_span = features.assignment_name_span(missing_target)
    assert missing_span is not None
    missing_ref = features._resolve_in_stmt(
        parse_program("[missing] := value\n"),
        missing_target,
        lsp.Position(line=0, character=2),
        None,
        {},
        {},
        {},
    )
    assert missing_ref is not None and missing_ref.kind == "local"
    candidates = []
    features._collect_value_reference_candidates(
        [ReassignStmt(name="missing", loc=None)],
        "missing",
        candidates,
    )
    assert candidates == []

    branch = program.toplevel_body[1]
    loop = program.toplevel_body[2]
    conditional = program.toplevel_body[3]
    assert branch.loc is not None and loop.loc is not None and conditional.loc is not None
    assert (
        features._declaration_in_scope_path(
            program.toplevel_body,
            "outer",
            branch.loc.start_byte,
            None,
        )
        is not None
    )
    assert (
        features._declaration_in_scope_path(
            [branch],
            "branch",
            branch.loc.start_byte + 1,
            None,
        )
        is None
    )
    assert (
        features._declaration_in_scope_path(
            [loop],
            "loop_local",
            loop.loc.start_byte + 1,
            None,
        )
        is None
    )
    assert (
        features._declaration_in_scope_path(
            [conditional],
            "conditional",
            conditional.loc.start_byte + 1,
            None,
        )
        is None
    )

    parent = program.procedures[0]
    child = features._nested_procedures(parent)[0]
    return_stmt = child.body[0]
    captured_segment, param_segment = return_stmt.segments[0], return_stmt.segments[2]
    captured_span = features._input_name_span(captured_segment)
    param_span = features._input_name_span(param_segment)
    assert captured_span is not None and param_span is not None
    from kedi.lsp.coordinates import source_coordinates

    with source_coordinates(source, "/tmp/scope-path.kedi"):
        assert features._resolve_lexical_name(parent, (), "param") is not None
        captured = features._resolve_value_declaration(
            program,
            child,
            (parent,),
            "captured",
            captured_span,
        )
        parameter = features._resolve_value_declaration(
            program,
            child,
            (parent,),
            "param",
            param_span,
        )
        legacy_param = features._resolve_in_segment(
            InputSegment(name="param", loc=param_segment.loc),
            lsp.Position(
                line=param_span.start_line - 1,
                character=param_span.start_col,
            ),
            parent,
            {},
            {},
            {},
        )
    assert captured is not None and captured.kind == "local"
    assert parameter is not None and parameter.kind == "parameter"
    assert legacy_param is not None and legacy_param.kind == "parameter"

    nested_source = (
        "@grand(grand_param: str) -> str:\n"
        "  @middle() -> str:\n"
        "    @leaf() -> str:\n"
        "      = <grand_param>\n"
        "    = <leaf()>\n"
        "  = <middle()>\n"
    )
    nested_program = parse_program(
        nested_source,
        source_path="/tmp/deep-capture.kedi",
    )
    grand = nested_program.procedures[0]
    middle = features._nested_procedures(grand)[0]
    leaf = features._nested_procedures(middle)[0]
    grand_param_segment = leaf.body[0].segments[0]
    grand_param_span = features._input_name_span(grand_param_segment)
    assert grand_param_span is not None
    with source_coordinates(nested_source, "/tmp/deep-capture.kedi"):
        deep_parameter = features._resolve_value_declaration(
            nested_program,
            leaf,
            (grand, middle),
            "grand_param",
            grand_param_span,
        )
    assert deep_parameter is not None and deep_parameter.scope_proc is grand


def test_collect_loop_binder_refs_respects_nested_shadowing_and_map_scope() -> None:
    from kedi.lang import parse_program

    source = (
        "[outside] = value\n"
        "> loop [item]: `[1]`:\n"
        "  = <item>\n"
        "  > if: `True`:\n"
        "    = <item>\n"
        "  > loop [item]: `[2]`:\n"
        "    = <item>\n"
        "  > loop [other]: `[3]`:\n"
        "    = <item>\n"
        "  > loop: `False`:\n"
        "    = <item>\n"
        "> map:\n"
        "  = <item>\n"
        "@worker() -> str:\n"
        "  > loop [worker_item]: `[1]`:\n"
        "    = <worker_item>\n"
        "  = done\n"
    )
    program = parse_program(source)
    loop = program.toplevel_body[1]
    target = features.SymbolRef(
        "loop_binder",
        "item",
        loop.binder_loc,
        None,
        loop.binder_loc,
    )
    refs = []

    features._collect_loop_binder_refs(program, target, refs)

    assert [(span.start_line, span.start_col) for span in refs] == [
        (3, 6),
        (5, 8),
        (9, 8),
        (11, 8),
        (13, 6),
    ]

    worker_loop = program.procedures[0].body[0]
    worker_target = features.SymbolRef(
        "loop_binder",
        "worker_item",
        worker_loop.binder_loc,
        program.procedures[0],
        worker_loop.binder_loc,
    )
    worker_refs = []
    features._collect_loop_binder_refs(program, worker_target, worker_refs)
    assert [(span.start_line, span.start_col) for span in worker_refs] == [(16, 8)]


def test_document_symbol_helpers(tmp_path: Path) -> None:
    from kedi.lang.ast import (
        EvalDataset,
        EvalMetric,
        EvalSuite,
        OptimizeSpan,
        ProcDefStmt,
        ProcedureDef,
        ProgramWithValidation,
        TestCase,
        TestSuite,
    )

    program = ProgramWithValidation(
        procedures=[
            ProcedureDef(
                "outer",
                [],
                body=cast(
                    Any,
                    [
                        ProcDefStmt(
                            proc=ProcedureDef("inner", [], body=[], loc=make_edge_span(tmp_path)),
                            loc=make_edge_span(tmp_path),
                        ),
                    ],
                ),
                optimize_spans=[OptimizeSpan("prompt", loc=make_edge_span(tmp_path))],
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
        tests=[
            TestSuite(
                "outer",
                [TestCase("basic", "assert True", loc=make_edge_span(tmp_path))],
                loc=make_edge_span(tmp_path),
            )
        ],
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


def test_range_and_overlap_helpers() -> None:
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
