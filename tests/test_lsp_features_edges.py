"""LSP feature tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from lsprotocol import types as lsp

from kedi.lsp import features
from tests.lsp_helpers import hover_value


def test_lsp_feature_helper_edges(tmp_path, monkeypatch):
    from kedi.lang.ast import (
        AssignBlockStmt,
        AssignStmt,
        AutoSpec,
        BacktickLineStmt,
        CallArg,
        CallSegment,
        EvalDataset,
        EvalMetric,
        EvalSuite,
        InputSegment,
        ModuleExport,
        ModuleImport,
        OptimizeSpan,
        OutputSegment,
        ProcDefStmt,
        ProcedureDef,
        Program,
        ProgramWithValidation,
        PythonBlock,
        PythonExprSegment,
        ReturnBlockStmt,
        ReturnStmt,
        Span,
        Stmt,
        TemplateLine,
        TestCase,
        TestSuite,
        TextSegment,
        TypeApply,
        TypeDef,
        TypeExpr,
        TypeField,
        TypeRef,
        TypeUnion,
    )
    from kedi.lang.parser import parse_program

    def span(line: int = 1, start_col: int = 1, end_col: int = 80) -> Span:
        return Span(
            start_byte=0,
            end_byte=max(end_col - start_col, 0),
            start_line=line,
            start_col=start_col,
            end_line=line,
            end_col=end_col,
            source_path=str(tmp_path / "x.kedi"),
        )

    pos = lsp.Position(line=0, character=4)
    assert "builtins" in features._build_lsp_type_registry()["str"]
    monkeypatch.setattr(features, "_typing_extensions", None)
    assert "str" in features._build_lsp_type_registry()
    monkeypatch.undo()
    assert features._program_path(Program(source_map=SimpleNamespace(documents=[]))) is None
    mapped_prog = Program(
        source_map=SimpleNamespace(documents=[SimpleNamespace(path=str(tmp_path / "m.kedi"))])
    )
    assert features._program_path(mapped_prog) == (tmp_path / "m.kedi").resolve()
    assert features._assignment_name_span(AssignStmt(name="x")) is None
    assert features._is_public_export_name("public")
    assert not features._is_public_export_name("")
    assert not features._is_public_export_name("_private")

    source_map = SimpleNamespace(get_line=lambda path, line: f"{path}:{line}")
    mapped_line_prog = Program(loc=span(), source_map=source_map)
    assert features._program_line_text(mapped_line_prog, 7) == f"{tmp_path / 'x.kedi'}:7"
    assert features._program_line_text(Program(source_text="one"), 2) is None
    assert features._program_line_text(Program(source_map=source_map), 1) is None
    assert features._program_line_text(Program(), 1) is None
    assert features._output_name_span(OutputSegment(name="x")) is None
    assert features._input_name_span(InputSegment(name="x")) is None
    assert features._extract_module_docstring("") is None
    assert features._extract_module_docstring("###\n###\n") is None
    assert features._extract_module_docstring("# one\n# two\n= ok\n") == "one\ntwo"
    assert features._extract_module_docstring("# one\n# two\n") == "one\ntwo"

    class FakeNode:
        def __init__(self, type_: str, start: int = 0, end: int = 3):
            self.type = type_
            self.start_byte = start
            self.end_byte = end
            self.is_named = True

    from kedi.lsp import module_surface

    monkeypatch.setattr(
        module_surface,
        "_ts_parse",
        lambda source: SimpleNamespace(
            root_node=SimpleNamespace(children=[FakeNode("block_comment")])
        ),
    )
    assert features._extract_module_docstring("###") is None
    monkeypatch.undo()

    assert features._import_name_span(Program(), ModuleImport("profiles")) is None
    no_line_prog = Program(imports=[ModuleImport("profiles", loc=span())])
    assert features._import_name_span(no_line_prog, no_line_prog.imports[0]) is None
    wrong_line_prog = Program(
        source_text="> import: other\n",
        imports=[ModuleImport("profiles", loc=span())],
    )
    assert features._import_name_span(wrong_line_prog, wrong_line_prog.imports[0]) is None

    export_prog = Program(
        source_text="> export:\n  *\n  !bad\n  Name\n",
        exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))],
    )
    assert (
        features._exported_name_at_position(
            Program(exports=[ModuleExport(["Name"], loc=None)]),
            lsp.Position(line=0, character=0),
        )
        is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=10, character=0)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=0, character=3)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=1, character=3)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=2, character=3)) is None
    )
    hit = features._exported_name_at_position(export_prog, lsp.Position(line=3, character=3))
    assert hit is not None and hit[0] == "Name"
    no_source_export = Program(exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))])
    assert (
        features._exported_name_at_position(no_source_export, lsp.Position(line=3, character=3))
        is None
    )

    class FindlessLine(str):
        def find(self, sub, start=0, end=None):  # noqa: ANN001
            return (
                -1 if sub == "Name" else super().find(sub, start, len(self) if end is None else end)
            )

    monkeypatch.setattr(
        features, "_program_line_text", lambda *args, **kwargs: FindlessLine("  Name")
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=3, character=3)) is None
    )
    monkeypatch.undo()

    assert features._load_imported_surface(tmp_path / "missing.kedi", set()).modules == {}
    assert (
        features._load_imported_surface(
            tmp_path / "missing.kedi", {tmp_path / "missing.kedi"}
        ).modules
        == {}
    )
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text("[bad\n", encoding="utf-8")
    bad_surface = features._load_imported_surface(bad_module, set(), "bad")
    assert bad_surface.modules == {}
    assert bad_surface.failures
    assert bad_surface.failures[0].module_name == "bad"
    bad_import_source = "> import: bad\n= ok\n"
    bad_import_diagnostics = features.compute_diagnostics(
        bad_import_source, str(tmp_path / "main_bad.kedi")
    )
    assert not any(d.code == "K-IMPORT-LOAD-FAILED" for d in bad_import_diagnostics)
    exported_module = tmp_path / "exported.kedi"
    exported_module.write_text(
        "> import: missing_nested\n"
        "~Public(name)\n"
        "~_Private(name)\n"
        "[value: int] = `1`\n"
        "[_hidden: int] = `2`\n"
        "@proc() -> str:\n"
        "  = ok\n"
        "> export: *\n",
        encoding="utf-8",
    )
    surface = features._load_imported_surface(exported_module, set(), "exported")
    assert "exported" in surface.modules
    assert "Public" in surface.types
    assert "_Private" not in surface.types
    assert "value" in surface.values
    assert "_hidden" not in surface.values
    assert "proc" in surface.procedures
    no_module_surface = features._load_imported_surface(exported_module, set(), None)
    assert no_module_surface.modules == {}
    nested_module = tmp_path / "nested.kedi"
    nested_module.write_text(
        "@nested_proc() -> str:\n  = ok\n> export:\n  nested_proc\n", encoding="utf-8"
    )
    parent_module = tmp_path / "parent.kedi"
    parent_module.write_text("> import: nested\n> export:\n  nested_proc\n", encoding="utf-8")
    nested_surface = features._load_imported_surface(parent_module, set(), "parent")
    assert "nested_proc" in nested_surface.procedures
    missing_export_module = tmp_path / "missing_export.kedi"
    missing_export_module.write_text("> export:\n  absent\n", encoding="utf-8")
    assert features._load_imported_surface(missing_export_module, set()).procedures == {}

    missing_import_prog = Program(imports=[ModuleImport("missing", loc=None)])
    assert features._missing_import_diagnostics(missing_import_prog) == []
    unresolved_prog = parse_program(
        "> import: no_such_module\n= ok\n", source_path=str(tmp_path / "main.kedi")
    )
    assert features._imported_surface_for_program(unresolved_prog).modules == {}

    type_ref = TypeRef(name="Person", loc=Span(0, 6, 1, 1, 1, 7))
    type_apply = TypeApply(name="list", args=[type_ref], loc=Span(0, 12, 1, 1, 1, 13))
    type_union = TypeUnion(left=type_apply, right=TypeRef(name="str", loc=span()), loc=span())
    assert features._type_expr_to_str(None) == "None"
    assert features._type_expr_to_str(type_ref) == "Person"
    assert features._type_expr_to_str(type_apply) == "list[Person]"
    assert features._type_expr_to_str(type_union) == "list[Person] | str"
    assert "TypeExpr" in features._type_expr_to_str(TypeExpr())

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

    proc = ProcedureDef(
        name="p",
        params=["x"],
        body=[],
        param_type_annotations={"x": "int"},
        return_type_annotation="str",
        loc=span(),
    )
    assert "@p(x: int)" in features._proc_signature_markdown(proc)
    assert "AI-generated" in features._proc_signature_markdown(
        ProcedureDef(
            name="auto",
            params=[],
            auto_spec=AutoSpec("make it", loc=span()),
            loc=span(),
        )
    )
    assert "optimization spans" in features._proc_signature_markdown(
        ProcedureDef(
            name="opt",
            params=[],
            optimize_spans=[OptimizeSpan("prompt", loc=span())],
            loc=span(),
        )
    )
    assert "doc" in features._proc_signature_markdown(
        ProcedureDef(name="doc", params=[], docstring="doc", loc=span())
    )

    described_nodes = [
        proc,
        TypeDef("Person", [TypeField("name", loc=span())], loc=span()),
        TypeField("name", default_expr="'Ada'", loc=span()),
        TemplateLine(segments=[TextSegment(text="hello", loc=span())], loc=span()),
        AssignStmt(name="x", type_annotation="int", loc=span()),
        AssignBlockStmt(name="x", type_annotation="int", loc=span()),
        ReturnStmt(loc=span()),
        ReturnBlockStmt(loc=span()),
        BacktickLineStmt(code="x = 1", loc=span()),
        PythonBlock(code="x = 1", loc=span()),
        AutoSpec("make it", loc=span()),
        OptimizeSpan("prompt", loc=span()),
        TextSegment(text="x" * 50, loc=span()),
        InputSegment(name="x", loc=span()),
        OutputSegment(name="x", type_annotation="int", loc=span()),
        CallSegment(name="p", loc=span()),
        PythonExprSegment(code="x", loc=span()),
        EvalSuite("p", loc=span()),
        TestSuite("p", loc=span()),
        EvalDataset("train", "[]", loc=span()),
        EvalMetric("score", "return 1", loc=span()),
    ]
    assert all(features._describe_ast_node(node) for node in described_nodes)
    assert features._describe_ast_node(Stmt(loc=span())) is None

    assert features._generic_hover(Program(), pos) is None
    assert (
        features._generic_hover(
            SimpleNamespace(procedures=[], type_defs=[], top_level_statements=[]), pos
        )
        is None
    )
    assert (
        features._generic_hover(
            SimpleNamespace(procedures=[], type_defs=[], top_level_statements=[Stmt(loc=span())]),
            pos,
        )
        is None
    )

    nested_arg = CallArg(loc=span())
    setattr(nested_arg, "__call_segment__", CallSegment(name="inner", args=[], loc=span()))
    rendered_arg = CallArg(loc=span())
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="rendered", args=[], loc=span())],
    )
    outer_call = CallSegment(name="outer", args=[nested_arg, rendered_arg], loc=span())
    calls: list[Span] = []
    features._collect_call_refs(outer_call, "inner", calls)
    features._collect_call_refs(outer_call, "rendered", calls)
    assert len(calls) == 2
    inputs: list[Span] = []
    nested_input_arg = CallArg(loc=span())
    setattr(nested_input_arg, "__python_segment__", InputSegment(name="needle", loc=span()))
    rendered_input_arg = CallArg(loc=span())
    setattr(
        rendered_input_arg,
        "__render_segments__",
        [InputSegment(name="needle", loc=span())],
    )
    features._collect_input_refs(
        CallSegment(name="outer", args=[nested_input_arg, rendered_input_arg], loc=span()),
        "needle",
        inputs,
    )
    assert len(inputs) == 2

    refs: list[Span] = []
    features._collect_type_refs_in_expr(type_union, "Person", refs)
    features._collect_type_refs_in_expr(type_apply, "list", refs)
    features._collect_type_refs_in_stmt(
        OutputSegment(name="x", type_expr=type_ref, loc=span()), "Person", refs
    )
    features._collect_type_refs_in_type_def(
        TypeDef("Box", [TypeField("item", type_expr=type_ref)]), "Person", refs
    )
    assert refs

    assert features._position_in_proc_name(proc, lsp.Position(line=1, character=1)) is False
    assert features._find_local_declaration(None, "x") is None
    local_proc = ProcedureDef(
        "local",
        [],
        body=[
            AssignStmt(name="x", loc=span()),
            TemplateLine(segments=[OutputSegment(name="y", loc=span())], loc=span()),
        ],
    )
    assert features._find_local_declaration(local_proc, "x") is not None
    assert features._find_local_declaration(local_proc, "y") is not None
    assert features._find_local_declaration(local_proc, "z") is None

    monkeypatch.setattr(
        features,
        "_param_decls",
        lambda proc: [("miss", span(1, 20, 25)), ("x", span())],
    )
    assert features._resolve_in_procedure(proc, pos, {}, {}, {}) is not None
    monkeypatch.undo()

    assert (
        features._resolve_in_type_expr(
            TypeUnion(left=type_ref, right=TypeRef(name="Other", loc=span()), loc=span()),
            pos,
            {"Person": TypeDef("Person", [], loc=span())},
        )
        is not None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(name="list", args=[type_ref], loc=span()),
            lsp.Position(line=0, character=4),
            {"Person": TypeDef("Person", [], loc=span())},
        )
        is not None
    )
    assert (
        features._resolve_in_stmt(
            AssignBlockStmt(name="x", type_expr=type_ref, loc=span()),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=span())},
            {},
        )
        is not None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(name="x", type_expr=type_ref, loc=span()),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=span())},
            {},
        )
        is not None
    )
    field_ref = features._resolve_in_type_def(
        TypeDef("Person", [TypeField("name", loc=span(1, 9, 13))], loc=span()),
        lsp.Position(line=0, character=8),
        {},
    )
    assert field_ref is not None and field_ref.kind == "type_field"
    assert (
        features._resolve_in_type_def(
            TypeDef("Person", [TypeField("name", loc=span(1, 9, 13))], loc=span()),
            lsp.Position(line=0, character=20),
            {},
        )
        is None
    )
    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=TypeRef(name="Nope", loc=span(1, 20, 24)),
                right=TypeRef(name="Other", loc=span(1, 30, 35)),
                loc=span(),
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
                args=[TypeRef(name="Other", loc=span(1, 20, 25))],
                loc=span(),
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
                type_expr=TypeRef(name="Other", loc=span(1, 20, 25)),
                loc=span(),
            ),
            pos,
            None,
            {},
            {},
            {},
        )
        is None
    )

    resolve_arg = CallArg(loc=span())
    setattr(resolve_arg, "__call_segment__", InputSegment(name="resolved", loc=span()))
    resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[resolve_arg], loc=span()),
        lsp.Position(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert resolved is not None and resolved.name == "resolved"
    rendered_resolve_arg = CallArg(loc=span())
    setattr(
        rendered_resolve_arg,
        "__render_segments__",
        [InputSegment(name="rendered", loc=span())],
    )
    rendered_resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[rendered_resolve_arg], loc=span()),
        lsp.Position(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert rendered_resolved is not None and rendered_resolved.name == "rendered"
    missing_inner_arg = CallArg(loc=span())
    setattr(
        missing_inner_arg,
        "__call_segment__",
        InputSegment(name="miss", loc=span(1, 20, 25)),
    )
    missing_render_arg = CallArg(loc=span())
    setattr(
        missing_render_arg,
        "__render_segments__",
        [InputSegment(name="miss", loc=span(1, 20, 25))],
    )
    assert (
        features._resolve_in_segment(
            CallSegment(name="outer", args=[missing_inner_arg, missing_render_arg], loc=span()),
            lsp.Position(line=0, character=10),
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
                type_expr=TypeRef(name="Other", loc=span(1, 20, 25)),
                loc=span(),
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
                            proc=ProcedureDef("inner", [], body=[], loc=span()),
                            loc=span(),
                        ),
                        TemplateLine(segments=[outer_call], loc=span()),
                    ],
                ),
                auto_spec=AutoSpec("auto", loc=span()),
                optimize_spans=[OptimizeSpan("prompt", [TemplateLine(loc=span())], loc=span())],
                loc=span(),
            )
        ],
        type_defs=[TypeDef("Box", [TypeField("item", loc=span())], loc=span())],
        tests=[TestSuite("outer", [TestCase("basic", "assert True", loc=span())], loc=span())],
        evals=[
            EvalSuite(
                "outer",
                datasets={"train": EvalDataset("train", "[]", loc=span())},
                test_datasets={"holdout": EvalDataset("holdout", "[]", loc=span())},
                metrics=[EvalMetric("score", "return 1", loc=span())],
                loc=span(),
            )
        ],
    )
    setattr(
        program,
        "top_level_statements",
        [
            TemplateLine(segments=[InputSegment(name="x", loc=span())], loc=span()),
            AssignStmt(name="x", expr=[TextSegment(text="value", loc=span())], loc=span()),
        ],
    )
    assert features._smallest_enclosing(program, pos) is not None

    ref_prog = Program(
        procedures=[
            ProcedureDef(
                name="target",
                params=[],
                body=[TemplateLine(segments=[CallSegment(name="target", loc=span())], loc=span())],
                loc=span(),
            ),
            ProcedureDef(
                name="typed",
                params=["x"],
                body=[AssignStmt(name="v", type_expr=type_ref, loc=span())],
                param_type_exprs={"x": type_ref, "skip": None},
                return_type_expr=type_ref,
                loc=span(),
            ),
        ],
        toplevel_return=ReturnStmt(segments=[CallSegment(name="target", loc=span())], loc=span()),
        toplevel_body=[
            TemplateLine(segments=[CallSegment(name="target", loc=span())], loc=span()),
            AssignStmt(name="v", type_expr=type_ref, loc=span()),
        ],
        type_defs=[TypeDef("Person", [TypeField("item", type_expr=type_ref)], loc=span())],
    )
    duplicate_span = span()
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
            loc=span(),
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
            lsp.Position(line=2, character=10),
            {"start": {"line": 1, "character": 0}, "end": {"line": 2, "character": 5}},
        )
        is False
    )


def test_lsp_feature_public_edge_paths(tmp_path, monkeypatch):
    from kedi.lang.ast import CallArg, CallSegment, EvalDataset, EvalMetric, EvalSuite, Stmt

    bad = "@broken(:\n"
    assert features.compute_hover(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1)) is None
    assert (
        features.compute_definition(
            bad,
            "file:///tmp/bad.kedi",
            "/tmp/bad.kedi",
            lsp.Position(line=0, character=1),
        )
        is None
    )
    assert (
        features.compute_references(
            bad,
            "file:///tmp/bad.kedi",
            "/tmp/bad.kedi",
            lsp.Position(line=0, character=1),
            True,
        )
        == []
    )
    assert (
        features.compute_prepare_rename(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_rename(
            bad,
            "file:///tmp/bad.kedi",
            "/tmp/bad.kedi",
            lsp.Position(line=0, character=1),
            "x",
        )
        is None
    )
    assert (
        features.compute_signature_help(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_inlay_hints(
            bad,
            "/tmp/bad.kedi",
            lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=1, character=0),
            ),
        )
        == []
    )
    assert features.compute_document_symbols(bad, "/tmp/bad.kedi") == []

    main = tmp_path / "main.kedi"
    missing_src = "> import: missing\n= ok\n"
    main.write_text(missing_src, encoding="utf-8")
    hover = features.compute_hover(missing_src, str(main), lsp.Position(line=0, character=11))
    assert hover is not None and "import target not found" in hover_value(hover)

    module_src = "~Person(name)\n[x: int] = `1`\n> export:\n  Person\n  x\n"
    assert "Kedi type definition" in hover_value(
        features.compute_hover(module_src, "/tmp/mod.kedi", lsp.Position(line=3, character=3))
    )
    assert "Module value" in hover_value(
        features.compute_hover(module_src, "/tmp/mod.kedi", lsp.Position(line=4, character=2))
    )

    producer = tmp_path / "producer.kedi"
    producer.write_text("[value] = ok\n> export:\n  value\n", encoding="utf-8")
    consumer = tmp_path / "consumer.kedi"
    consumer_src = "> import: producer\n> export:\n  value\n= <`value`>\n"
    consumer.write_text(consumer_src, encoding="utf-8")
    hover = features.compute_hover(consumer_src, str(consumer), lsp.Position(line=2, character=3))
    assert hover is not None and "Imported value" in hover_value(hover)
    same_file_definition = features.compute_definition(
        module_src,
        "file:///tmp/mod.kedi",
        "/tmp/mod.kedi",
        lsp.Position(line=3, character=3),
    )
    assert same_file_definition is not None and same_file_definition.uri == "file:///tmp/mod.kedi"

    unknown_call_src = "= <missing()>\n"
    assert "procedure not found" in hover_value(
        features.compute_hover(unknown_call_src, "/tmp/x.kedi", lsp.Position(line=0, character=4))
    )
    assert "Local" in hover_value(
        features.compute_hover(
            "@p():\n  = <unknown>\n", "/tmp/x.kedi", lsp.Position(line=1, character=6)
        )
    )
    assert "Field" in hover_value(
        features.compute_hover("~Person(name)\n", "/tmp/x.kedi", lsp.Position(line=0, character=9))
    )

    generic = features.compute_hover(
        "@p():\n  plain prompt text\n", "/tmp/x.kedi", lsp.Position(line=1, character=4)
    )
    assert generic is not None and "Text segment" in hover_value(generic)

    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="parameter",
            name="x",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 1, 1, 1, 1, 2),
        ),
    )
    assert "Parameter" in hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="type_field",
            name="field",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "Field" in hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="custom",
            name="thing",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "custom" in hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.undo()

    value_span = features.Span(0, 5, 1, 1, 1, 6)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="imported_value",
            name="loose",
            declaration_span=value_span,
            scope_proc=None,
            cursor_span=value_span,
        ),
    )
    monkeypatch.setattr(
        features,
        "_imported_surface_for_program",
        lambda prog: features.ImportedSurface(
            values={"loose": features.ImportedValueInfo("loose", value_span)}
        ),
    )
    assert "another Kedi module" in hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.undo()

    assert (
        features.compute_prepare_rename("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_rename(
            "= ok\n",
            "file:///tmp/x.kedi",
            "/tmp/x.kedi",
            lsp.Position(line=0, character=1),
            "x",
        )
        is None
    )
    assert (
        features.compute_prepare_rename(missing_src, str(main), lsp.Position(line=0, character=11))
        is None
    )
    assert (
        features.compute_prepare_rename(
            "@p(x: int):\n  = <x>\n", "/tmp/x.kedi", lsp.Position(line=0, character=6)
        )
        is None
    )
    assert (
        features.compute_rename(
            "@p(x: int):\n  = <x>\n",
            "file:///tmp/x.kedi",
            "/tmp/x.kedi",
            lsp.Position(line=0, character=6),
            "y",
        )
        is None
    )

    external_span = features.Span(0, 1, 1, 1, 1, 2, source_path=str(tmp_path / "other.kedi"))
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            "imported_value", "v", external_span, None, external_span
        ),
    )
    assert (
        features.compute_prepare_rename("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
        is None
    )
    assert (
        features.compute_rename(
            "= ok\n",
            "file:///tmp/x.kedi",
            "/tmp/x.kedi",
            lsp.Position(line=0, character=0),
            "v2",
        )
        is None
    )
    monkeypatch.undo()

    rename_span = features.Span(0, 1, 1, 1, 1, 2)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", rename_span, None, rename_span),
    )
    monkeypatch.setattr(
        features,
        "_find_all_references",
        lambda prog, sym, *, include_declaration: [rename_span],
    )
    rename_edit = features.compute_rename(
        "= ok\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=0),
        "y",
    )
    assert rename_edit is not None and rename_edit.changes is not None
    assert len(rename_edit.changes["file:///tmp/x.kedi"]) == 1
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", None, None, cast(Any, None)),
    )
    rename_without_cursor = features.compute_rename(
        "= ok\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=0),
        "y",
    )
    assert rename_without_cursor is not None and rename_without_cursor.changes is not None
    assert len(rename_without_cursor.changes["file:///tmp/x.kedi"]) == 1
    monkeypatch.undo()

    type_src = "~Person(name)\n@p(x: Person) -> list[Person]:\n  [y: Person] = <x>\n  Value [z: Person]\n  = <x>\n"
    refs = features.compute_references(
        type_src,
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=2),
        True,
    )
    assert len(refs) >= 4
    local_refs = features.compute_references(
        "@p(x):\n  Value [answer]\n  = <answer>\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=2, character=6),
        True,
    )
    assert local_refs

    nested_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  = <inner(<inner(1)>)>\n"
    sig = features.compute_signature_help(
        nested_sig_src, "/tmp/x.kedi", lsp.Position(line=4, character=15)
    )
    assert sig is not None and sig.active_parameter == 0
    top_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n[x] = <inner(1)>\n"
    top_sig = features.compute_signature_help(
        top_sig_src, "/tmp/x.kedi", lsp.Position(line=3, character=13)
    )
    assert top_sig is not None and top_sig.signatures
    no_return_sig_src = "@plain(x):\n  = <x>\n\n= <plain(ok)>\n"
    no_return_sig = features.compute_signature_help(
        no_return_sig_src, "/tmp/x.kedi", lsp.Position(line=3, character=8)
    )
    assert no_return_sig is not None and "->" not in no_return_sig.signatures[0].label
    assert (
        features.compute_signature_help(
            "= <missing(1)>\n", "/tmp/x.kedi", lsp.Position(line=0, character=10)
        )
        is None
    )

    hints_src = "@inner(x, y):\n  = ok\n\n= <inner(<inner(1, 2)>, 3)>\n"
    hints = features.compute_inlay_hints(
        hints_src,
        "/tmp/x.kedi",
        lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=10, character=0),
        ),
    )
    assert len(hints) >= 3
    extra_hints = features.compute_inlay_hints(
        "@one(x):\n  = ok\n\n= <one(1, 2)>\n",
        "/tmp/x.kedi",
        lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=10, character=0),
        ),
    )
    assert extra_hints
    assert (
        features.compute_inlay_hints(
            "@one(x):\n  = ok\n\n= <one(1)>\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=10, character=0),
                end=lsp.Position(line=11, character=0),
            ),
        )
        == []
    )
    assert (
        features.compute_inlay_hints(
            "Value [answer]\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=10, character=0),
                end=lsp.Position(line=11, character=0),
            ),
        )
        == []
    )
    assert (
        features.compute_inlay_hints(
            "Value [answer: int]\n= <answer>\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=3, character=0),
            ),
        )
        == []
    )

    assert (
        features._in_range(
            features.Span(0, 1, 1, 1, 1, 2),
            lsp.Range(
                start=lsp.Position(line=2, character=0),
                end=lsp.Position(line=3, character=0),
            ),
        )
        is False
    )
    assert (
        features._in_range(
            features.Span(0, 1, 5, 1, 5, 2),
            lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=3, character=0),
            ),
        )
        is False
    )
    assert (
        features._active_param_index(
            CallSegment(name="p", args=cast(Any, None)),
            lsp.Position(line=0, character=0),
        )
        == 0
    )
    assert features._innermost_call_at(Stmt(), lsp.Position(line=0, character=0)) is None
    rendered_arg = CallArg(loc=features.Span(0, 10, 1, 1, 1, 20))
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="deep", args=[], loc=features.Span(0, 10, 1, 5, 1, 12))],
    )
    innermost = features._innermost_call_in_segment(
        CallSegment(name="outer", args=[rendered_arg], loc=features.Span(0, 20, 1, 1, 1, 21)),
        lsp.Position(line=0, character=6),
    )
    assert innermost is not None and innermost.name == "deep"

    eval_symbols = features.compute_document_symbols(
        "@eval: p:\n"
        "  > data: train:\n"
        "    = `[]`\n"
        "  > test_data: holdout:\n"
        "    = `[]`\n"
        "  > metric: score(train):\n"
        "    = `1.0`\n",
        "/tmp/x.kedi",
    )
    assert any(symbol.name == "@eval: p" for symbol in eval_symbols)
    direct_eval = features._eval_suite_symbol(
        EvalSuite(
            "p",
            datasets={"train": EvalDataset("train", "[]")},
            test_datasets={"holdout": EvalDataset("holdout", "[]")},
            metrics=[EvalMetric("score", "return 1")],
        )
    )
    assert direct_eval.range.start.line == 0

    monkeypatch.setitem(features._CAPTURE_TO_LSP, "custom.deep", ("variable", 0))
    assert features._capture_to_token("custom.deep.extra") == (
        features.SEMANTIC_TOKEN_TYPES.index("variable"),
        0,
    )
    assert features._capture_to_token("missing.capture") is None

    original_py_parser = features._PY_PARSER
    original_py_query = features._PY_HIGHLIGHTS_QUERY
    features._PY_PARSER = original_py_parser
    features._PY_HIGHLIGHTS_QUERY = original_py_query
    py_tokens = features.compute_semantic_tokens_full("= <`len([1, 2])`>\n", "/tmp/x.kedi")
    assert len(py_tokens.data) % 5 == 0
    monkeypatch.setattr(features, "_KEDI_HIGHLIGHTS_QUERY", None)
    monkeypatch.setattr(features, "_KEDI_INJECTIONS_QUERY", None)
    assert features.compute_semantic_tokens_full("@p():\n  = ok\n", "/tmp/x.kedi").data == []
    monkeypatch.undo()

    all_ranges = features.compute_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    header_ranges = features.compute_header_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    assert any(r.get("ownerKind") == "param" for r in header_ranges)
    assert all(r["text"] == "1" for r in header_ranges)
    assert any(r["text"] == "x" for r in all_ranges)
