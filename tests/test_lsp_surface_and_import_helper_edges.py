"""LSP feature helper edge-case tests."""

from __future__ import annotations

from types import SimpleNamespace

from kedi.lsp import ast_helpers, features, module_surface
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_at_line, position_of


def test_lsp_surface_and_import_helper_edges(tmp_path, monkeypatch):
    from kedi.lang.ast import (
        AssignStmt,
        InputSegment,
        ModuleExport,
        ModuleImport,
        OutputSegment,
        Program,
        Span,
    )
    from kedi.lang.parser import parse_program

    assert "builtins" in features._build_lsp_type_registry()["str"]
    monkeypatch.setattr(features, "_typing_extensions", None)
    assert "str" in features._build_lsp_type_registry()
    monkeypatch.undo()
    assert module_surface.program_path(Program(source_map=SimpleNamespace(documents=[]))) is None
    mapped_prog = Program(
        source_map=SimpleNamespace(documents=[SimpleNamespace(path=str(tmp_path / "m.kedi"))])
    )
    assert module_surface.program_path(mapped_prog) == (tmp_path / "m.kedi").resolve()
    assert module_surface.assignment_name_span(AssignStmt(name="x")) is None
    assert module_surface.is_public_export_name("public")
    assert not module_surface.is_public_export_name("")
    assert not module_surface.is_public_export_name("_private")

    source_map = SimpleNamespace(get_line=lambda path, line: f"{path}:{line}")
    mapped_line_prog = Program(loc=make_edge_span(tmp_path), source_map=source_map)
    assert features._program_line_text(mapped_line_prog, 7) == f"{tmp_path / 'x.kedi'}:7"
    assert features._program_line_text(Program(source_text="one"), 2) is None
    assert features._program_line_text(Program(source_map=source_map), 1) is None
    assert features._program_line_text(Program(), 1) is None
    assert features._output_name_span(OutputSegment(name="x")) is None
    assert features._input_name_span(InputSegment(name="x")) is None
    assert module_surface.extract_module_docstring("") is None
    assert module_surface.extract_module_docstring("###\n###\n") is None
    assert module_surface.extract_module_docstring("# one\n# two\n= ok\n") == "one\ntwo"
    assert module_surface.extract_module_docstring("# one\n# two\n") == "one\ntwo"

    class FakeNode:
        def __init__(self, type_: str, start: int = 0, end: int = 3):
            self.type = type_
            self.start_byte = start
            self.end_byte = end
            self.is_named = True

    monkeypatch.setattr(
        ast_helpers,
        "_ts_parse",
        lambda source: SimpleNamespace(
            root_node=SimpleNamespace(children=[FakeNode("block_comment")])
        ),
    )
    assert module_surface.extract_module_docstring("###") is None
    monkeypatch.undo()

    assert features._import_name_span(Program(), ModuleImport("profiles")) is None
    no_line_prog = Program(imports=[ModuleImport("profiles", loc=make_edge_span(tmp_path))])
    assert features._import_name_span(no_line_prog, no_line_prog.imports[0]) is None
    wrong_line_prog = Program(
        source_text="> import: other\n",
        imports=[ModuleImport("profiles", loc=make_edge_span(tmp_path))],
    )
    assert features._import_name_span(wrong_line_prog, wrong_line_prog.imports[0]) is None

    export_src = "> export:\n  *\n  !bad\n  Name\n"
    export_prog = Program(
        source_text=export_src,
        exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))],
    )
    assert (
        features._exported_name_at_position(
            Program(exports=[ModuleExport(["Name"], loc=None)]),
            position_of("> export:\n", ">"),
        )
        is None
    )
    assert features._exported_name_at_position(export_prog, position_at_line(10, 0)) is None
    assert (
        features._exported_name_at_position(export_prog, position_of(export_src, "export")) is None
    )
    assert features._exported_name_at_position(export_prog, position_of(export_src, "*")) is None
    assert features._exported_name_at_position(export_prog, position_of(export_src, "!bad")) is None
    hit = features._exported_name_at_position(export_prog, position_of(export_src, "Name"))
    assert hit is not None and hit[0] == "Name"
    no_source_export = Program(exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))])
    assert (
        features._exported_name_at_position(no_source_export, position_of(export_src, "Name"))
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
    assert features._exported_name_at_position(export_prog, position_of(export_src, "Name")) is None
    monkeypatch.undo()

    assert module_surface.load_imported_surface(tmp_path / "missing.kedi", set()).modules == {}
    assert (
        module_surface.load_imported_surface(
            tmp_path / "missing.kedi", {tmp_path / "missing.kedi"}
        ).modules
        == {}
    )
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text("[bad\n", encoding="utf-8")
    bad_surface = module_surface.load_imported_surface(bad_module, set(), "bad")
    assert bad_surface.modules == {}
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
    surface = module_surface.load_imported_surface(exported_module, set(), "exported")
    assert "exported" in surface.modules
    assert "Public" in surface.types
    assert "_Private" not in surface.types
    assert "value" in surface.values
    assert "_hidden" not in surface.values
    assert "proc" in surface.procedures
    no_module_surface = module_surface.load_imported_surface(exported_module, set(), None)
    assert no_module_surface.modules == {}
    nested_module = tmp_path / "nested.kedi"
    nested_module.write_text(
        "@nested_proc() -> str:\n  = ok\n> export:\n  nested_proc\n", encoding="utf-8"
    )
    parent_module = tmp_path / "parent.kedi"
    parent_module.write_text("> import: nested\n> export:\n  nested_proc\n", encoding="utf-8")
    nested_surface = module_surface.load_imported_surface(parent_module, set(), "parent")
    assert "nested_proc" in nested_surface.procedures
    missing_export_module = tmp_path / "missing_export.kedi"
    missing_export_module.write_text("> export:\n  absent\n", encoding="utf-8")
    assert module_surface.load_imported_surface(missing_export_module, set()).procedures == {}

    missing_import_prog = Program(imports=[ModuleImport("missing", loc=None)])
    assert features._missing_import_diagnostics(missing_import_prog) == []
    unresolved_prog = parse_program(
        "> import: no_such_module\n= ok\n", source_path=str(tmp_path / "main.kedi")
    )
    assert module_surface.imported_surface_for_program(unresolved_prog).modules == {}
