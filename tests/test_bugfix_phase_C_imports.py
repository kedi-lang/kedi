from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kedi.lang import compile_program, parse_program


def _compile_file(path: Path, *, runtime_globals=None):
    source = path.read_text(encoding="utf-8")
    return compile_program(
        parse_program(source, source_path=str(path.resolve())),
        runtime_globals=runtime_globals,
    )


def test_imported_procedure_sees_importer_runtime_globals(tmp_path: Path) -> None:
    """BUG-009: an imported procedure must run under the importing program's
    runtime globals so program-level context such as ``args`` is available."""
    (tmp_path / "lib.kedi").write_text(
        "@show_arg() -> str:\n  = <`args.name`>\n\n> export:\n  show_arg\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: lib\n\n= <show_arg()>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main, runtime_globals={"args": SimpleNamespace(name="Ada")})

    assert rt.run_main() == "Ada"


def test_imported_procedure_called_directly_sees_runtime_globals(tmp_path: Path) -> None:
    """The propagated globals must be visible even when the wrapped imported
    procedure is invoked directly (not just through ``run_main``)."""
    (tmp_path / "lib.kedi").write_text(
        "@show_arg() -> str:\n  = <`args.name`>\n\n> export:\n  show_arg\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: lib\n\n= ok\n",
        encoding="utf-8",
    )

    rt = _compile_file(main, runtime_globals={"args": SimpleNamespace(name="Grace")})

    assert rt._procedures["show_arg"]() == "Grace"


def test_module_top_level_export_materializes_but_return_does_not_leak(
    tmp_path: Path,
) -> None:
    """BUG-008 (design lock): importing a module runs its top-level code to
    materialize exported values, but the module's top-level RETURN value must
    not leak into the importing program."""
    (tmp_path / "lib.kedi").write_text(
        "[shared: int] = `7`\n= module-top-level-return\n\n> export:\n  shared\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: lib\n\n= <`shared`>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    # The exported top-level assignment is materialized into the importer.
    assert rt.run_main() == "7"
    assert rt._import_env["shared"] == 7
    # The module's top-level return value did not leak into the importer.
    assert "module-top-level-return" not in rt._import_env.values()
