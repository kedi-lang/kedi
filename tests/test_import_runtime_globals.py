from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize("identifier", ["args", "cli_args"])
@pytest.mark.parametrize("nested", [False, True])
def test_star_export_preserves_cli_binding_without_reexporting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: str, nested: bool
) -> None:
    from kedi.lang.compiler import environment

    monkeypatch.setattr(environment, "CLI_ARGS_IDENTIFIER", identifier)
    (tmp_path / "lib.kedi").write_text(
        f"@show_arg() -> str:\n  = <`{identifier}.name`>\n"
        "[public_value: int] = `7`\n"
        "[_private: int] = `9`\n"
        "> export: *\n",
        encoding="utf-8",
    )
    module = "lib"
    if nested:
        (tmp_path / "bridge.kedi").write_text("> import: lib\n> export: *\n", encoding="utf-8")
        module = "bridge"
    main = tmp_path / "main.kedi"
    main.write_text(
        f"> import: {module}\n= <show_arg()>:<`{identifier}.name`>:<public_value>\n",
        encoding="utf-8",
    )
    cli_args = SimpleNamespace(name="reader")
    runtime = _compile_file(main, runtime_globals={identifier: cli_args})
    assert runtime.run_main() == "reader:reader:7"
    assert runtime._initial_globals_env[identifier] is cli_args
    assert identifier not in runtime._import_env
    assert "_private" not in runtime._import_env
    assert runtime._import_env["public_value"] == 7


def test_cli_binding_cannot_be_selectively_imported_from_star_export(tmp_path: Path) -> None:
    (tmp_path / "lib.kedi").write_text("[value: int] = `1`\n> export: *\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text("> import: lib:\n  args\n", encoding="utf-8")
    with pytest.raises(KediExecutionError, match="does not export 'args'"):
        _compile_file(main, runtime_globals={"args": SimpleNamespace(name="reader")})


def test_explicit_cli_binding_export_still_rejects_overwriting_importer(tmp_path: Path) -> None:
    (tmp_path / "lib.kedi").write_text("> export:\n  args\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text("> import: lib\n", encoding="utf-8")
    with pytest.raises(KediExecutionError, match="Reserved identifier 'args'"):
        _compile_file(main, runtime_globals={"args": SimpleNamespace(name="reader")})


def _compile_file(path: Path, *, runtime_globals=None):
    source = path.read_text(encoding="utf-8")
    return compile_program(
        parse_program(source, source_path=str(path.resolve())),
        runtime_globals=runtime_globals,
    )


def test_imported_procedure_sees_importer_runtime_globals(tmp_path: Path) -> None:
    """an imported procedure must run under the importing program's
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


def test_imported_procedure_called_directly_sees_runtime_globals(
    tmp_path: Path,
) -> None:
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
    """Importing a module runs its top-level code to materialize exported values,
    but the module's top-level RETURN value must not leak into the importer."""
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
