from __future__ import annotations

from pathlib import Path

import pytest

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _compile_file(path: Path, *, adapter=None):
    source = path.read_text(encoding="utf-8")
    return compile_program(parse_program(source, source_path=str(path.resolve())), adapter=adapter)


def test_parser_reads_import_and_export_directives() -> None:
    prog = parse_program(
        "> import: profiles (file name without .kedi)\n\n"
        "> export:\n"
        "  xd\n"
        "  get_proc\n"
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
        "[profile: Profile] = <get_profile(`\"Ada\"`)>\n"
        "= <`profile.name`>:<`profile.id`>:<`profile_id + 1`>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "Ada:41:42"


def test_imported_values_are_visible_inside_local_procedures(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "[xd: int] = `10`\n"
        "\n"
        "> export:\n"
        "  xd\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        "@plus_one() -> int:\n"
        "  = `xd + 1`\n"
        "\n"
        "= <plus_one()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "11"


def test_imported_procedure_runs_with_its_own_runtime_context(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "@ask() -> str:\n"
        "  Say [answer].\n"
        "  = <answer>\n"
        "\n"
        "> export:\n"
        "  ask\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        "= <ask()>\n",
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
        "[profile: Profile] = `Profile(name=\"Ada\")`\n"
        "= <get_proc()>:<`xd`>:<`profile.name`>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "ok:10:Ada"
    assert "_hidden_proc" not in rt._procedures
    assert "_hidden_value" not in rt._import_env


def test_unexported_procedure_is_not_imported(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "@hidden() -> str:\n"
        "  = nope\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        "= <hidden()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(KeyError, match="Unknown procedure: hidden"):
        rt.run_main()


def test_missing_exported_name_raises_during_compile(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        "> export:\n"
        "  missing\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: profiles\n"
        "\n"
        "= ok\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Kedi export 'missing' is not defined"):
        _compile_file(main)


def test_import_without_source_path_cannot_resolve_module() -> None:
    prog = parse_program("> import: profiles\n= ok\n")

    with pytest.raises(RuntimeError, match="without a source path"):
        compile_program(prog)
