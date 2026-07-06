from __future__ import annotations

from pathlib import Path

import pytest

from kedi import module_registry
from kedi.lang import compile_program, parse_program
from kedi.module_registry import (
    describe_module_search_paths,
    internal_module_path,
    module_search_paths,
    resolve_module_path,
)
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
        "@ask() -> str:\n  >> Say [answer].\n  = <answer>\n\n> export:\n  ask\n",
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


def test_import_exposes_exported_agent_profile(tmp_path: Path) -> None:
    (tmp_path / "models.kedi").write_text(
        "> profile: advanced:\n    > model: opus\n\n> export:\n  advanced\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: models\n\n> use: advanced\n@ask():\n    >> answer [x]\n\n= <ask()>\n",
        encoding="utf-8",
    )

    adapter = MockAdapter(default_model="default-model")
    rt = _compile_file(main, adapter=adapter)
    rt.run_main()

    assert adapter.model_calls == ["opus"]
    assert rt._profile_registry["advanced"].members[0].plain_value == "opus"


def test_star_export_includes_agent_profiles(tmp_path: Path) -> None:
    (tmp_path / "models.kedi").write_text(
        "> profile: advanced:\n"
        "    > model: opus\n"
        "> profile: _hidden:\n"
        "    > model: secret\n"
        "\n"
        "> export: *\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: models\n\n> use: advanced\n@ask():\n    >> answer [x]\n\n= <ask()>\n",
        encoding="utf-8",
    )

    adapter = MockAdapter(default_model="default-model")
    rt = _compile_file(main, adapter=adapter)
    rt.run_main()

    assert adapter.model_calls == ["opus"]
    assert "advanced" in rt._profile_registry
    assert "_hidden" not in rt._profile_registry


def test_internal_filesystem_module_exports_profile_and_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello filesystem", encoding="utf-8")

    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: filesystem\n\n= <read_text_file(`"data.txt"`)>\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "hello filesystem"
    assert "filesystem" in rt._profile_registry
    assert "read_text_file" in rt._procedures


def test_internal_filesystem_apply_patch_tool_updates_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello\nold value\nbye\n", encoding="utf-8")

    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: filesystem\n\n"
        '= <apply_patch(`"update"`, `"data.txt"`, `""`, `"old value"`, `"new value"`)>\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == f"updated {data_file.resolve()}"
    assert data_file.read_text(encoding="utf-8") == "hello\nnew value\nbye\n"


def test_internal_filesystem_apply_patch_add_rejects_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "data.txt"
    data_file.write_text("existing", encoding="utf-8")

    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: filesystem\n\n= <apply_patch(`"add"`, `"data.txt"`, `"new"`)>\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(Exception, match="file already exists"):
        rt.run_main()


def test_internal_filesystem_apply_patch_rejects_ambiguous_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "data.txt"
    data_file.write_text("same\nsame\n", encoding="utf-8")

    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: filesystem\n\n= <apply_patch(`"update"`, `"data.txt"`, `""`, `"same"`, `"new"`)>\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(Exception, match="ambiguous"):
        rt.run_main()


def test_internal_filesystem_rejects_out_of_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    main = tmp_path / "main.kedi"
    main.write_text(
        f'> import: filesystem\n\n= <write_text_file(`{str(outside)!r}`, `"owned"`)>\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(Exception, match="path escapes filesystem root"):
        rt.run_main()


def _compile_fs_module_with_call(tmp_path: Path, call: str) -> Path:
    # remove_directory is intentionally not exported / not in the default profile, so exercise the
    # real module code by compiling the actual filesystem.kedi source with an appended return.
    src = Path(internal_module_path("filesystem")).read_text(encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text(f"{src}\n{call}\n", encoding="utf-8")
    return main


def test_internal_filesystem_remove_directory_refuses_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # B15: remove_directory(".") resolves to the sandbox root; deleting it would wipe the tree.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keep.txt").write_text("important", encoding="utf-8")
    main = _compile_fs_module_with_call(tmp_path, '= <remove_directory(`"."`)>')
    rt = _compile_file(main)

    with pytest.raises(Exception, match="protected root directory"):
        rt.run_main()
    assert (tmp_path / "keep.txt").exists()  # the root tree is untouched


def test_internal_filesystem_remove_directory_allows_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A genuine subdirectory removal still works — the guard only protects the roots.
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "scratch"
    victim.mkdir()
    (victim / "f.txt").write_text("x", encoding="utf-8")
    main = _compile_fs_module_with_call(tmp_path, '= <remove_directory(`"scratch"`)>')
    rt = _compile_file(main)

    assert rt.run_main() == str(victim.resolve())
    assert not victim.exists()


def test_internal_filesystem_remove_file_refuses_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review #4: the destructive guard now lives at the _safe_path chokepoint, so remove_file is
    # protected symmetrically — remove_file(".") refuses the sandbox root with a clear error.
    monkeypatch.chdir(tmp_path)
    main = _compile_fs_module_with_call(tmp_path, '= <remove_file(`"."`)>')
    rt = _compile_file(main)

    with pytest.raises(Exception, match="protected root directory"):
        rt.run_main()


def test_internal_filesystem_remove_file_deletes_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A genuine file removal still works through the for_delete chokepoint.
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "f.txt"
    victim.write_text("x", encoding="utf-8")
    main = _compile_fs_module_with_call(tmp_path, '= <remove_file(`"f.txt"`)>')
    rt = _compile_file(main)

    assert rt.run_main() == str(victim.resolve())
    assert not victim.exists()


def test_internal_sandbox_dependency_check_does_not_call_kedi_from_prelude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    find_spec = importlib.util.find_spec
    checked_names: list[str] = []

    def fake_find_spec(name: str):
        checked_names.append(name)
        return None if name == "pydantic_monty" else find_spec(name)

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        fake_find_spec,
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: sandbox\n\n= ok\n", encoding="utf-8")

    with pytest.raises(Exception, match="requires package.*pydantic_monty"):
        _compile_file(main)
    assert "pydantic_monty" in checked_names
    assert '"pydantic_monty"' not in checked_names


def test_internal_filesystem_profile_registers_tool_names(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: filesystem\n\n"
        "> use: filesystem\n\n"
        ">> Read pyproject.toml and provide [content]\n\n"
        "= <content>\n",
        encoding="utf-8",
    )

    adapter = MockAdapter(default_model="mock")
    rt = _compile_file(main, adapter=adapter)

    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [
        (
            "read_text_file",
            "write_text_file",
            "apply_patch",
            "path_exists",
            "list_directory",
            "create_directory",
            "get_file_info",
        )
    ]


def test_internal_errors_module_formats_single_package_name(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: errors\n\n= ```\nraise ModuleNotInstalledError("abc")\n```\n',
        encoding="utf-8",
    )

    rt = _compile_file(main)

    with pytest.raises(Exception, match="abc is not installed"):
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
