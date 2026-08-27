"""LSP surface, import, and module helper edge-case tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.lsp import ast_helpers, features, module_surface
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_at_line, position_of


def test_lsp_type_registry_includes_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "builtins" in features._build_lsp_type_registry()["str"]
    monkeypatch.setattr(features, "_typing_extensions", None)
    assert "str" in features._build_lsp_type_registry()


def test_program_path_and_export_name_rules(tmp_path: Path) -> None:
    from kedi.lang.ast import VariableInitStmt, Program

    assert module_surface.program_path(Program(source_map=SimpleNamespace(documents=[]))) is None
    mapped_prog = Program(
        source_map=SimpleNamespace(documents=[SimpleNamespace(path=str(tmp_path / "m.kedi"))])
    )
    assert module_surface.program_path(mapped_prog) == (tmp_path / "m.kedi").resolve()
    assert module_surface.binding_name_span(VariableInitStmt(name="x")) is None
    assert module_surface.is_public_export_name("public")
    assert not module_surface.is_public_export_name("")
    assert not module_surface.is_public_export_name("_private")


def test_program_line_text_edges(tmp_path: Path) -> None:
    from kedi.lang.ast import Program

    source_map = SimpleNamespace(get_line=lambda path, line: f"{path}:{line}")
    mapped_line_prog = Program(loc=make_edge_span(tmp_path), source_map=source_map)
    assert features._program_line_text(mapped_line_prog, 7) == f"{tmp_path / 'x.kedi'}:7"
    assert features._program_line_text(Program(source_text="one"), 2) is None
    assert features._program_line_text(Program(source_map=source_map), 1) is None
    assert features._program_line_text(Program(), 1) is None


def test_input_output_name_span_without_location() -> None:
    from kedi.lang.ast import InputSegment, OutputSegment

    assert features._output_name_span(OutputSegment(name="x")) is None
    assert features._input_name_span(InputSegment(name="x")) is None


def test_module_docstring_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    assert module_surface.extract_module_docstring("") is None
    assert module_surface.extract_module_docstring("###\n###\n") is None
    assert module_surface.extract_module_docstring("# one\n# two\n= ok\n") == "one\ntwo"
    assert module_surface.extract_module_docstring("# one\n# two\n") == "one\ntwo"

    class FakeNode:
        def __init__(self, type_: str, start: int = 0, end: int = 3) -> None:
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


def test_import_name_span_edges(tmp_path: Path) -> None:
    from kedi.lang.ast import ModuleImport, Program

    assert features._import_name_span(Program(), ModuleImport("profiles")) is None
    no_line_prog = Program(imports=[ModuleImport("profiles", loc=make_edge_span(tmp_path))])
    assert features._import_name_span(no_line_prog, no_line_prog.imports[0]) is None
    wrong_line_prog = Program(
        source_text="> import: other\n",
        imports=[ModuleImport("profiles", loc=make_edge_span(tmp_path))],
    )
    assert features._import_name_span(wrong_line_prog, wrong_line_prog.imports[0]) is None


def test_name_span_helpers_handle_missing_line_byte_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    from kedi.lang.parser import parse_program

    use_prog = parse_program("> use: foo\n")
    use = use_prog.toplevel_body[0]
    import_prog = parse_program("> import: profiles\n")
    module_import = import_prog.imports[0]
    export_prog = parse_program("> export:\n  Name\n")

    monkeypatch.setattr(features, "_line_start_byte_for_program", lambda *_args: None)

    assert use.loc is not None
    assert module_import.loc is not None
    assert features._directive_name_span(use_prog, use.loc, "foo") is None
    assert features._import_name_span(import_prog, module_import) is None
    assert (
        features._exported_name_at_position(
            export_prog,
            position_of("> export:\n  Name\n", "Name"),
        )
        is None
    )


def test_exported_name_at_position_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.lang.ast import ModuleExport, Program, Span

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


def test_load_imported_surface_edges(tmp_path: Path) -> None:
    assert module_surface.load_imported_surface(tmp_path / "missing.kedi", set()).modules == {}
    assert (
        module_surface.load_imported_surface(
            tmp_path / "missing.kedi", {tmp_path / "missing.kedi"}
        ).modules
        == {}
    )
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text(">> [bad\n", encoding="utf-8")
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


def test_load_imported_surface_reads_python_prelude_exports(tmp_path: Path) -> None:
    module = tmp_path / "helpers.kedi"
    module.write_text(
        "```\n"
        "def require(package_names: str | list[str], required_by: str | None = None) -> bool:\n"
        "    return True\n"
        "\n"
        "async def fetch_one(url: str) -> str:\n"
        "    return url\n"
        "\n"
        "class Helper:\n"
        "    name: str\n"
        "    @property\n"
        "    def display_name(self) -> str:\n"
        "        return self.name\n"
        "    @classmethod\n"
        "    def create(cls, name: str) -> Helper:\n"
        "        return cls()\n"
        "    @staticmethod\n"
        "    def normalize(value: str) -> str:\n"
        "        return value.strip()\n"
        "    @custom\n"
        "    def decorated(self) -> str:\n"
        "        return self.name\n"
        "    def format(self, prefix: str) -> str:\n"
        "        return f'{prefix}{self.name}'\n"
        "\n"
        "answer: int = 42\n"
        "left, right = 1, 2\n"
        "nested, holder.attr = 3, 4\n"
        "\n"
        "def no_return(value):\n"
        "    return value\n"
        "```\n"
        "\n"
        "> export:\n"
        "  require\n"
        "  fetch_one\n"
        "  Helper\n"
        "  answer\n"
        "  left\n"
        "  right\n"
        "  nested\n"
        "  no_return\n",
        encoding="utf-8",
    )

    surface = module_surface.load_imported_surface(module, set(), "helpers")

    assert surface.values["require"].python_stub == (
        "def require(package_names: str | list[str], required_by: str | None=None) -> bool: ..."
    )
    assert surface.values["fetch_one"].python_stub == ("async def fetch_one(url: str) -> str: ...")
    assert surface.values["Helper"].python_stub == (
        "class Helper:\n"
        "    name: str\n"
        "    @property\n"
        "    def display_name(self) -> str: ...\n"
        "    @classmethod\n"
        "    def create(cls, name: str) -> Helper: ...\n"
        "    @staticmethod\n"
        "    def normalize(value: str) -> str: ...\n"
        "    def decorated(self) -> str: ...\n"
        "    def format(self, prefix: str) -> str: ..."
    )
    assert surface.values["answer"].python_stub == "answer: int = ..."
    assert surface.values["left"].python_stub == "left: object = ..."
    assert surface.values["right"].python_stub == "right: object = ..."
    assert surface.values["nested"].python_stub == "nested: object = ..."
    assert surface.values["no_return"].python_stub == "def no_return(value): ..."
    assert module_surface._unparse_or_any(None) == "Any"
    assert surface.values["require"].declaration_span.start_line == 2
    assert surface.values["answer"].declaration_span.start_line == 25


def test_python_name_span_falls_back_when_source_is_unreadable_or_line_is_missing(
    tmp_path: Path,
) -> None:
    from kedi.lang.ast import Span

    unreadable = Span(10, 20, 4, 1, 4, 10, source_path=str(tmp_path / "missing.py"))
    unreadable_span = module_surface._python_name_span(unreadable, "value", 1, 3)
    assert (unreadable_span.start_byte, unreadable_span.end_byte) == (10, 20)

    source = tmp_path / "short.py"
    source.write_text("value = 1\n", encoding="utf-8")
    outside = Span(10, 20, 20, 1, 20, 10, source_path=str(source))
    outside_span = module_surface._python_name_span(outside, "value", 5, 3)
    assert (outside_span.start_byte, outside_span.end_byte) == (10, 20)

    no_path = Span(10, 20, 4, 1, 4, 10)
    no_path_span = module_surface._python_name_span(no_path, "value", 1, 3)
    assert (no_path_span.start_byte, no_path_span.end_byte) == (10, 20)


def test_load_imported_surface_ignores_invalid_python_prelude_exports(
    tmp_path: Path,
) -> None:
    module = tmp_path / "broken_python.kedi"
    module.write_text(
        "```\ndef broken(\n```\n\n> export:\n  broken\n",
        encoding="utf-8",
    )

    assert module_surface.load_imported_surface(module, set(), "broken_python").values == {}


def test_missing_import_diagnostics_and_unresolved_surface(tmp_path: Path) -> None:
    from kedi.lang.ast import ModuleImport, Program
    from kedi.lang.parser import parse_program

    missing_import_prog = Program(imports=[ModuleImport("missing", loc=None)])
    assert features._missing_import_diagnostics(missing_import_prog) == []
    unresolved_prog = parse_program(
        "> import: no_such_module\n= ok\n", source_path=str(tmp_path / "main.kedi")
    )
    assert module_surface.imported_surface_for_program(unresolved_prog).modules == {}


def test_profile_validation_and_imported_hover_helper_edges(tmp_path: Path) -> None:
    from kedi.lang.ast import ProcedureDef, ProfileDefStmt, Program, Span, TestSuite

    profile_source = "> profile: analyst:\n    > use:\n        search_docs\n\n        search_docs\n"
    profile = ProfileDefStmt(
        name="analyst",
        tool_names=["search_docs", "search_docs"],
        loc=Span(0, len(profile_source), 1, 1, 5, 20),
    )
    profile_spans = features._profile_use_entry_spans(
        Program(source_text=profile_source),
        profile,
    )
    assert [(name, span.start_line) for name, span in profile_spans] == [
        ("search_docs", 3),
        ("search_docs", 5),
    ]

    assert features._validation_procedure_span(Program(), TestSuite("p")) is None
    suite = TestSuite("p", loc=Span(0, 5, 1, 1, 1, 5))
    assert features._validation_procedure_span(Program(), suite) is None

    malformed_proc = ProcedureDef(
        name="p",
        params=["missing"],
        loc=Span(0, 5, 1, 1, 1, 5),
    )
    from kedi.lsp.coordinates import source_coordinates

    with source_coordinates("@p(other):\n  = ok\n", None):
        assert features._param_decls(malformed_proc) == []
        assert features._resolve_lexical_name(malformed_proc, (), "missing") is None

    assert features._imported_value_markdown("missing", None) == "**Imported value** `missing`"
    async_value = module_surface.ImportedValueInfo(
        name="fetch",
        declaration_span=Span(0, 5, 1, 1, 1, 6),
        python_stub="async def fetch(url: str) -> str: ...",
    )
    assert "Python async function" in features._imported_value_markdown("fetch", async_value)

    helpers = tmp_path / "helpers.kedi"
    helpers.write_text(
        "```\nclass External:\n    value: str\n```\n\n> export:\n  External\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    source = "> import: helpers\n@consume(value: External[str]):\n  = ok\n"
    main.write_text(source, encoding="utf-8")
    hover = features.compute_hover(
        source,
        str(main),
        position_of(source, "External"),
    )
    assert hover is not None
    assert "Python class" in hover.contents.value
