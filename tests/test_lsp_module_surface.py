from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.lang.ast import AssignStmt, Program, Span
from kedi.lang.parser import parse_program
from kedi.lsp import module_surface


def test_module_surface_returns_empty_surface_for_invalid_module(tmp_path) -> None:
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text(">> [bad\n", encoding="utf-8")

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


def test_module_surface_exports_agent_profiles(tmp_path: Path) -> None:
    exported_module = tmp_path / "models.kedi"
    exported_module.write_text(
        "> profile: advanced:\n"
        "    > model: opus\n"
        "> profile: _hidden:\n"
        "    > model: secret\n"
        "\n"
        "> export:\n"
        "  advanced\n",
        encoding="utf-8",
    )

    surface = module_surface.load_imported_surface(exported_module, set(), "models")

    assert "advanced" in surface.profiles
    assert surface.profiles["advanced"].name == "advanced"
    assert "_hidden" not in surface.profiles


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
    second_stmt = AssignStmt(name="later", loc=stmt.loc)
    value_prog = Program(toplevel_body=[AssignStmt(name="missing"), stmt, second_stmt])
    values = module_surface.top_level_value_infos(value_prog)
    bindings = module_surface._binding_surface_for_program(
        value_prog,
        source_path=None,
        seen=set(),
        module_name=None,
    )

    assert values["value"].declaration_span.start_col == 2
    assert "value" in bindings.surface.values
    assert "later" in bindings.surface.values
    assert module_surface.assignment_name_span(AssignStmt(name="missing")) is None
    assert (
        module_surface._top_level_value_info(AssignStmt(name="missing"), module_name=None) is None
    )
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


def test_module_surface_filters_selective_imports_and_preserves_last_write(tmp_path) -> None:
    (tmp_path / "first.kedi").write_text(
        "@shared() -> str:\n  = first\n"
        "@only_first() -> str:\n  = first\n"
        "> export:\n  shared\n  only_first\n"
    )
    (tmp_path / "second.kedi").write_text("[shared: str] = second\n> export:\n  shared\n")
    prog = parse_program(
        "> import: first:\n  shared\n> import: second\n= <shared>\n",
        source_path=str(tmp_path / "main.kedi"),
    )

    surface = module_surface.imported_surface_for_program(prog)

    assert "only_first" not in surface.procedures
    assert "shared" not in surface.procedures
    assert surface.values["shared"].module_name == "second"


@pytest.mark.parametrize(
    ("facade_source", "expects_imported"),
    [
        (
            "@shared() -> str:\n  = local\n~Record(value: str)\n> import: values\n> export: *\n",
            True,
        ),
        (
            "> import: values\n@shared() -> str:\n  = local\n~Record(value: str)\n> export: *\n",
            False,
        ),
    ],
)
def test_module_surface_uses_runtime_write_order_for_local_and_imported_bindings(
    tmp_path: Path,
    facade_source: str,
    expects_imported: bool,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "@shared() -> str:\n  = imported\n"
        "~Record(value: str, source: str)\n"
        "> export:\n  shared\n  Record\n",
        encoding="utf-8",
    )
    facade = tmp_path / "facade.kedi"
    facade.write_text(facade_source, encoding="utf-8")

    surface = module_surface.load_imported_surface(facade, set(), "facade")

    assert (
        any(field.name == "source" for field in surface.types["Record"].fields) is expects_imported
    )
    procedure_text = surface.procedures["shared"].body[0].segments[0].text
    assert ("imported" in procedure_text) is expects_imported


@pytest.mark.parametrize(
    ("main_source", "expects_imported"),
    [
        (
            "> import: values\n@shared() -> str:\n  = local\n= <shared()>\n",
            False,
        ),
        (
            "@shared() -> str:\n  = local\n> import: values\n= <shared()>\n",
            True,
        ),
    ],
)
def test_program_import_surface_only_keeps_imported_final_writers(
    tmp_path: Path,
    main_source: str,
    expects_imported: bool,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "@shared() -> str:\n  = imported\n> export:\n  shared\n",
        encoding="utf-8",
    )
    program = parse_program(main_source, source_path=str(tmp_path / "main.kedi"))

    surface = module_surface.imported_surface_for_program(program)

    assert ("shared" in surface.procedures) is expects_imported


def test_module_surface_uses_cache_local_definition_offsets(tmp_path: Path) -> None:
    module = tmp_path / "generated.kedi"
    module.write_text("@root() -> str:\n  = ok\n", encoding="utf-8")
    cache = module.with_suffix(".cache.kedi")
    cache.write_text(
        "@cached() -> str:\n  = generated\n> export:\n  cached\n",
        encoding="utf-8",
    )

    surface = module_surface.load_imported_surface(module, set(), "generated")
    span = surface.procedures["cached"].loc

    assert span is not None
    assert span.source_path == str(cache.resolve())
    assert span.start_line == 1
    assert span.start_byte == 0
    assert span.end_byte == len("@cached() -> str:\n  = generated\n".encode("utf-8"))
