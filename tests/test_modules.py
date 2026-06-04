from __future__ import annotations

from pathlib import Path

import pytest

from kedi.lang import compile_program, parse_program
from kedi.module_registry import (
    describe_module_search_paths,
    internal_module_path,
    module_search_paths,
    resolve_module_path,
)
from kedi import module_registry
from tests.mock_adapter import MockAdapter


def _compile_file(path: Path, *, adapter=None):
    source = path.read_text(encoding="utf-8")
    return compile_program(parse_program(source, source_path=str(path.resolve())), adapter=adapter)


def test_parser_reads_import_and_export_directives() -> None:
    prog = parse_program(
        "> import: profiles (file name without .kedi)\n\n> export:\n  xd\n  get_proc\n"
    )

    assert [item.module for item in prog.imports] == ["profiles"]
    assert prog.exports[0].names == ["xd", "get_proc"]


def test_parser_reads_star_export_directive() -> None:
    prog = parse_program("> export: *\n")

    assert prog.exports[0].export_all is True
    assert prog.exports[0].names == []


def test_import_exposes_only_exported_procedure_value_and_type(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "~Profile(name: str, id: int)\n"
        "\n"
        "[profile_id: int] = `41`\n"
        "\n"
        "@get_profile(name: str) -> Profile:\n"
        "  = `Profile(name=name, id=profile_id)`\n"
        "\n"
        "> export:\n"
        "  Profile\n"
        "  get_profile\n"
        "  profile_id\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        '[profile: Profile] = <get_profile(`"Ada"`)>\n'
        "= <`profile.name`>:<`profile.id`>:<`profile_id + 1`>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "Ada:41:42"


def test_imported_values_are_visible_inside_local_procedures(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "[xd: int] = `10`\n\n> export:\n  xd\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n\n@plus_one() -> int:\n  = `xd + 1`\n\n= <plus_one()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "11"


def test_imported_procedure_runs_with_its_own_runtime_context(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "@ask() -> str:\n  Say [answer].\n  = <answer>\n\n> export:\n  ask\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n\n= <ask()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main, adapter=MockAdapter())

    assert rt.run_main() == "answer-value"


def test_star_export_exposes_public_names_only(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "~Profile(name: str)\n"
        "~_Private(value: str)\n"
        "\n"
        "[xd: int] = `10`\n"
        "[_hidden_value: int] = `20`\n"
        "\n"
        "@get_proc() -> str:\n"
        "  = ok\n"
        "\n"
        "@_hidden_proc() -> str:\n"
        "  = nope\n"
        "\n"
        "> export: *\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        '[profile: Profile] = `Profile(name="Ada")`\n'
        "= <get_proc()>:<`xd`>:<`profile.name`>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "ok:10:Ada"
    assert "_hidden_proc" not in rt._procedures
    assert "_hidden_value" not in rt._import_env


def test_unexported_procedure_is_not_imported(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "@hidden() -> str:\n  = nope\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n\n= <hidden()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(KeyError, match="Unknown procedure: hidden"):
        rt.run_main()


def test_missing_exported_name_raises_during_compile(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "> export:\n  missing\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n\n= ok\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Kedi export 'missing' is not defined"):
        _compile_file(main)


def test_cyclic_imports_raise_with_full_import_chain(tmp_path: Path) -> None:
    (tmp_path / "a.kedi").write_text(
        "> import: b\n\n> export:\n  a\n\n@a() -> str:\n  = a\n", encoding="utf-8"
    )
    (tmp_path / "b.kedi").write_text(
        "> import: a\n\n> export:\n  b\n\n@b() -> str:\n  = b\n", encoding="utf-8"
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: a\n\n= ok\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as info:
        _compile_file(main)

    message = str(info.value)
    assert "Cyclic Kedi import detected" in message
    assert str((tmp_path / "a.kedi").resolve()) in message
    assert str((tmp_path / "b.kedi").resolve()) in message


def test_import_without_source_path_cannot_resolve_module() -> None:
    prog = parse_program("> import: profiles\n= ok\n")

    with pytest.raises(RuntimeError, match="registered internal module"):
        compile_program(prog)


def test_internal_registry_module_resolves_from_default_store(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text("> import: this\n\n= ok\n", encoding="utf-8")

    rt = _compile_file(main)

    assert rt.run_main() == "ok"


def test_internal_registry_module_resolves_without_source_path() -> None:
    prog = parse_program("> import: this\n= ok\n")

    rt = compile_program(prog)

    assert rt.run_main() == "ok"


def test_module_registry_handles_internal_resource_lookup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_module_path.cache_clear()

    def raise_files(_package: str):
        raise RuntimeError("resource system unavailable")

    monkeypatch.setattr(module_registry.importlib.resources, "files", raise_files)

    assert internal_module_path("this") is None

    internal_module_path.cache_clear()

    class BrokenResource:
        def joinpath(self, _name: str) -> "BrokenResource":
            return self

        def is_file(self) -> bool:
            raise OSError("cannot inspect resource")

    monkeypatch.setattr(
        module_registry.importlib.resources, "files", lambda _package: BrokenResource()
    )

    assert internal_module_path("this") is None


def test_module_registry_search_path_rendering_and_dedup(tmp_path: Path) -> None:
    internal_module_path.cache_clear()
    internal = internal_module_path("this")
    assert internal is not None

    source_path = internal.parent / "main.kedi"
    assert module_search_paths("this", source_path=source_path) == (internal,)

    assert describe_module_search_paths(()) == "registered internal modules"
    assert describe_module_search_paths((tmp_path / "a.kedi",)) == str(tmp_path / "a.kedi")
    assert describe_module_search_paths((tmp_path / "a.kedi", tmp_path / "b.kedi")) == (
        f"{tmp_path / 'a.kedi'} or {tmp_path / 'b.kedi'}"
    )


def test_module_registry_missing_sibling_reports_file_path(tmp_path: Path) -> None:
    source = tmp_path / "main.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as info:
        resolve_module_path("profiles", source_path=source)

    assert str(tmp_path / "profiles.kedi") in str(info.value)
