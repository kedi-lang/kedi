from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.lang.ast import AssignStmt, Program, Span
from kedi.lang.parser import parse_program
from kedi.lsp import module_surface


def test_module_surface_returns_empty_surface_for_invalid_module(tmp_path) -> None:
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text("[bad\n", encoding="utf-8")

    bad_surface = module_surface.load_imported_surface(bad_module, set(), "bad")

    assert bad_surface.modules == {}

    exported_module = tmp_path / "profiles.kedi"
    exported_module.write_text(
        "###\nProfiles module.\n###\n"
        "~Profile(name)\n"
        "~_Private(name)\n"
        "[value: int] = `1`\n"
        "[_hidden: int] = `2`\n"
        "@get_profile() -> str:\n"
        "  = ok\n"
        "> export: *\n",
        encoding="utf-8",
    )

    surface = module_surface.load_imported_surface(exported_module, set(), "profiles")

    assert surface.modules["profiles"].docstring == "Profiles module."
    assert "Profile" in surface.types
    assert "_Private" not in surface.types
    assert "value" in surface.values
    assert "_hidden" not in surface.values
    assert "get_profile" in surface.procedures


def test_module_surface_program_path_and_value_helpers(tmp_path) -> None:
    assert module_surface.program_path(Program(source_map=SimpleNamespace(documents=[]))) is None

    source_path = tmp_path / "main.kedi"
    prog = Program(source_map=SimpleNamespace(documents=[SimpleNamespace(path=str(source_path))]))
    assert module_surface.program_path(prog) == source_path.resolve()

    stmt = AssignStmt(
        name="value",
        loc=Span(
            start_byte=0,
            end_byte=14,
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=15,
            source_path=str(source_path),
        ),
    )
    value_prog = Program(toplevel_body=[stmt])
    values = module_surface.top_level_value_infos(value_prog)

    assert values["value"].declaration_span.start_col == 2
    assert module_surface.assignment_name_span(AssignStmt(name="missing")) is None
    assert module_surface.is_public_export_name("public")
    assert not module_surface.is_public_export_name("_private")


def test_module_surface_lookup_import_path_handles_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing(*_args: object, **_kwargs: object) -> Path:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(module_surface, "find_module_path", raise_missing)

    assert module_surface._lookup_import_path("bad", None) is None


def test_module_surface_resolves_imported_program(tmp_path) -> None:
    (tmp_path / "child.kedi").write_text("@child() -> str:\n  = ok\n> export:\n  child\n")
    prog = parse_program("> import: child\n= <child()>\n", source_path=str(tmp_path / "main.kedi"))

    surface = module_surface.imported_surface_for_program(prog)

    assert "child" in surface.modules
    assert "child" in surface.procedures
