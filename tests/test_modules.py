from __future__ import annotations

import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi import module_registry, package_registry
from kedi.agent_profile import AgentProfile
from kedi.errors import KediExecutionError
from kedi.lang import KediParseError, PackageMetadata, compile_program, parse_program
from kedi.lang.ast import ModelDirectiveStmt, OutputSegment, Program, TemplateLine
from kedi.lang.compiler.program_compiler import ProgramCompiler
from kedi.module_registry import (
    describe_module_search_paths,
    installed_module_path,
    internal_module_path,
    module_search_paths,
    resolve_module_path,
)
from kedi.package_manifest import (
    load_package_manifest,
    parse_python_requirement,
    validate_package_metadata,
)
from kedi.package_registry import (
    CorruptInstalledPackageError,
    MockPackageRegistryClient,
    ResolvedPackageSource,
    add_package,
    install_git_package,
    install_package,
    installed_package_root,
    kedi_home,
)
from tests.mock_adapter import MockAdapter


def _compile_file(path: Path, *, adapter=None, runtime_globals: dict | None = None):
    source = path.read_text(encoding="utf-8")
    return compile_program(
        parse_program(source, source_path=str(path.resolve())),
        adapter=adapter,
        runtime_globals=runtime_globals,
    )


def test_parser_reads_import_and_export_directives() -> None:
    prog = parse_program("> import: profiles\n\n> export:\n  xd\n  get_proc\n")

    assert [item.module for item in prog.imports] == ["profiles"]
    assert prog.imports[0].names == []
    assert prog.exports[0].names == ["xd", "get_proc"]


def test_parser_reads_submodule_and_selective_import() -> None:
    prog = parse_program("> import: package/services:\n  fetch\n  Response\n")

    assert prog.imports[0].module == "package/services"
    assert prog.imports[0].names == ["fetch", "Response"]
    assert set(prog.imports[0].name_locs) == {"fetch", "Response"}


def test_module_registry_rejects_invalid_module_paths() -> None:
    with pytest.raises(ValueError, match="Invalid Kedi module path"):
        module_registry.module_search_paths("../private", source_path=Path("/tmp/main.kedi"))


def test_parser_rejects_text_after_import_module() -> None:
    with pytest.raises(KediParseError):
        parse_program("> import: profiles (file name without .kedi)\n")


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


def test_selective_import_exposes_only_selected_exports(tmp_path: Path) -> None:
    (tmp_path / "values.kedi").write_text(
        "[first: str] = one\n[second: str] = two\n\n> export:\n  first\n  second\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: values:\n  second\n\n= <second>\n", encoding="utf-8")

    rt = _compile_file(main)

    assert rt.run_main() == "two"
    assert "first" not in rt._import_env
    assert rt._import_env["second"] == "two"


def test_selective_import_rejects_name_outside_export_surface(tmp_path: Path) -> None:
    (tmp_path / "values.kedi").write_text(
        "[public: str] = yes\n[private: str] = no\n> export:\n  public\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: values:\n  private\n\n= ok\n", encoding="utf-8")

    with pytest.raises(KediExecutionError, match="does not export 'private'"):
        _compile_file(main)


def test_invalid_selective_import_does_not_execute_module_prelude(tmp_path: Path) -> None:
    (tmp_path / "values.kedi").write_text(
        "```\nevents.append('values prelude')\n```\n[public: str] = yes\n> export:\n  public\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: values:\n  absent\n", encoding="utf-8")
    events: list[str] = []

    with pytest.raises(KediExecutionError, match="does not export 'absent'"):
        _compile_file(main, runtime_globals={"events": events})

    assert events == []


def test_invalid_selective_import_from_star_export_does_not_execute_module_prelude(
    tmp_path: Path,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "```\nevents.append('values prelude')\n```\n[public: str] = yes\n> export: *\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: values:\n  absent\n", encoding="utf-8")
    events: list[str] = []

    with pytest.raises(KediExecutionError, match="does not export 'absent'"):
        _compile_file(main, runtime_globals={"events": events})

    assert events == []


def test_star_export_static_surface_includes_reexported_names_without_initializing_modules(
    tmp_path: Path,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "```\nevents.append('values prelude')\n```\n[public: str] = yes\n> export: *\n",
        encoding="utf-8",
    )
    (tmp_path / "facade.kedi").write_text(
        "> import: values:\n  public\n> export: *\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: facade:\n  absent\n", encoding="utf-8")
    events: list[str] = []

    with pytest.raises(KediExecutionError, match="does not export 'absent'"):
        _compile_file(main, runtime_globals={"events": events})

    assert events == []


@pytest.mark.parametrize(
    ("main_source", "expected"),
    [
        (
            "[value: str] = local\n> import: values\n= <value>\n",
            "imported",
        ),
        (
            "> import: values\n[value: str] = local\n= <value>\n",
            "local",
        ),
    ],
)
def test_import_and_local_value_collisions_follow_source_order(
    tmp_path: Path,
    main_source: str,
    expected: str,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "[value: str] = imported\n> export:\n  value\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(main_source, encoding="utf-8")

    assert _compile_file(main).run_main() == expected


@pytest.mark.parametrize(
    ("main_source", "expected"),
    [
        (
            "@source() -> str:\n  = local\n> import: values\n= <source()>\n",
            "imported",
        ),
        (
            "> import: values\n@source() -> str:\n  = local\n= <source()>\n",
            "local",
        ),
    ],
)
def test_import_and_local_procedure_collisions_follow_source_order(
    tmp_path: Path,
    main_source: str,
    expected: str,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "@source() -> str:\n  = imported\n> export:\n  source\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(main_source, encoding="utf-8")

    assert _compile_file(main).run_main() == expected


@pytest.mark.parametrize(
    ("main_source", "expects_imported_type"),
    [
        (
            "~Record(value: str)\n> import: values\n= done\n",
            True,
        ),
        (
            "> import: values\n~Record(value: str)\n= done\n",
            False,
        ),
    ],
)
def test_import_and_local_type_collisions_follow_source_order(
    tmp_path: Path,
    main_source: str,
    expects_imported_type: bool,
) -> None:
    (tmp_path / "values.kedi").write_text(
        "~Record(value: str, source: str)\n> export:\n  Record\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(main_source, encoding="utf-8")

    runtime = _compile_file(main)
    runtime.run_main()
    result_type = runtime._type_env_stack[0]["Record"]

    assert ("source" in result_type.model_fields) is expects_imported_type


def test_compiler_rejects_package_metadata_without_a_package_manifest_path() -> None:
    program = parse_program("> package: demo:\n  source: src/demo\n")

    with pytest.raises(ValueError, match="only allowed in package.kedi"):
        compile_program(program)


def test_repeated_import_runs_module_prelude_once_per_root_compilation(tmp_path: Path) -> None:
    (tmp_path / "common.kedi").write_text(
        "```\nevents.append('common')\n```\n[value: str] = ready\n> export:\n  value\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: common\n> import: common\n= <value>\n", encoding="utf-8")
    events: list[str] = []

    runtime = _compile_file(main, runtime_globals={"events": events})

    assert runtime.run_main() == "ready"
    assert events == ["common"]


def test_diamond_import_runs_shared_module_prelude_once(tmp_path: Path) -> None:
    (tmp_path / "common.kedi").write_text(
        "```\n"
        "events.append('common')\n"
        "```\n"
        "[common_value: str] = shared\n"
        "> export:\n"
        "  common_value\n",
        encoding="utf-8",
    )
    (tmp_path / "left.kedi").write_text(
        "> import: common\n> export:\n  common_value\n",
        encoding="utf-8",
    )
    (tmp_path / "right.kedi").write_text(
        "> import: common\n> export:\n  common_value\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: left\n> import: right\n= <common_value>\n", encoding="utf-8")
    events: list[str] = []

    runtime = _compile_file(main, runtime_globals={"events": events})

    assert runtime.run_main() == "shared"
    assert events == ["common"]


def test_submodule_imports_resolve_relative_to_importing_module(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "helper.kedi").write_text(
        "[suffix: str] = relative\n> export:\n  suffix\n",
        encoding="utf-8",
    )
    (package / "service.kedi").write_text(
        "> import: helper\n\n@result() -> str:\n  = <suffix>\n\n> export:\n  result\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: package/service:\n  result\n\n= <result()>\n", encoding="utf-8")

    rt = _compile_file(main)

    assert rt.run_main() == "relative"


def test_later_import_wins_when_binding_kind_changes(tmp_path: Path) -> None:
    (tmp_path / "procedures.kedi").write_text(
        "@shared() -> str:\n  = procedure\n> export:\n  shared\n",
        encoding="utf-8",
    )
    (tmp_path / "values.kedi").write_text(
        "[shared: str] = value\n> export:\n  shared\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: procedures\n> import: values\n\n= <shared>\n",
        encoding="utf-8",
    )

    rt = _compile_file(main)

    assert rt.run_main() == "value"
    assert "shared" not in rt._procedures


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
    assert "remove_file" in rt._procedures
    assert "remove_directory" in rt._procedures

    from kedi.artifacts import ArtifactStream
    from kedi.tool_context import kedi_tool_call_context

    with kedi_tool_call_context():
        streamed = rt._procedures["read_text_file"](file_path="data.txt")
    assert isinstance(streamed, ArtifactStream)
    assert list(streamed.iter_chunks()) == ["hello filesystem"]


def test_internal_filesystem_large_reads_require_explicit_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "large.txt"
    data_file.write_text("abcdef" + "x" * 1_000_000, encoding="utf-8")
    unbounded = tmp_path / "unbounded.kedi"
    unbounded.write_text(
        '> import: filesystem\n\n= <read_text_file(`"large.txt"`)>\n',
        encoding="utf-8",
    )
    bounded = tmp_path / "bounded.kedi"
    bounded.write_text(
        '> import: filesystem\n\n= <read_text_file(`"large.txt"`, `False`, `2`, `4`)>\n',
        encoding="utf-8",
    )
    bounded_large = tmp_path / "bounded_large.kedi"
    bounded_large.write_text(
        '> import: filesystem\n\n= <read_text_file(`"large.txt"`, `False`, `0`, `70000`)>\n',
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="use offset and max_chars"):
        _compile_file(unbounded).run_main()
    assert _compile_file(bounded).run_main() == "cdef"

    runtime = _compile_file(bounded_large)
    result = runtime.run_main()
    assert len(result) == 70_000
    assert result.startswith("abcdef")

    from kedi.artifacts import ArtifactStream
    from kedi.tool_context import kedi_tool_call_context

    with kedi_tool_call_context():
        streamed = runtime._procedures["read_text_file"](
            file_path="large.txt",
            max_chars=70_000,
        )
    assert isinstance(streamed, ArtifactStream)
    chunks = list(streamed.iter_chunks())
    assert [len(chunk) for chunk in chunks] == [65_536, 4_464]
    assert "".join(chunks) == result


def test_internal_filesystem_directory_listing_stops_at_bounded_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "many"
    directory.mkdir()
    for index in range(1_001):
        (directory / f"entry-{index}").touch()
    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: filesystem\n\n= <list_directory(`"many"`)>\n',
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="directory has too many entries"):
        _compile_file(main).run_main()


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


@pytest.mark.parametrize(
    "operation,old_text,expected",
    [
        ("update", "missing", "not found"),
        ("update", "same", "2 matches"),
        ("update", "", "must not be empty"),
        ("add", "", "already exists"),
        ("replace", "same", "one of: add, update, append"),
    ],
)
def test_agent_patch_rejections_preserve_file_and_do_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    old_text: str,
    expected: str,
) -> None:
    from kedi.tool_context import kedi_tool_call_context

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.txt"
    data.write_text("same\nsame\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: filesystem\n\n"
        f'= <apply_patch(`{operation!r}`, `"data.txt"`, `"new"`, `{old_text!r}`, `"new"`)>\n',
        encoding="utf-8",
    )
    with kedi_tool_call_context():
        result = _compile_file(main).run_main()
    assert result.startswith("Patch rejected; no changes made:")
    assert expected in result
    assert data.read_text(encoding="utf-8") == "same\nsame\n"


def test_internal_filesystem_rejects_out_of_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
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


def test_internal_filesystem_strict_policy_rejects_out_of_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "strict")
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    main = _compile_fs_module_with_call(
        tmp_path,
        f"= <read_text_file(`{str(outside)!r}`)>",
    )

    with pytest.raises(Exception, match="path escapes filesystem root"):
        _compile_file(main).run_main()


def test_internal_filesystem_none_policy_allows_out_of_root_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "none")
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    main = _compile_fs_module_with_call(
        tmp_path,
        f"= <read_text_file(`{str(outside)!r}`)>",
    )

    assert _compile_file(main).run_main() == "outside"


def test_internal_filesystem_none_policy_keeps_secret_file_protection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "none")
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / ".env.outside"
    outside.write_text("SECRET=value", encoding="utf-8")
    main = _compile_fs_module_with_call(
        tmp_path,
        f"= <read_text_file(`{str(outside)!r}`)>",
    )

    with pytest.raises(Exception, match="refusing to read secret file"):
        _compile_file(main).run_main()


def test_internal_filesystem_rejects_invalid_workspace_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "relaxed")
    monkeypatch.chdir(tmp_path)
    main = _compile_fs_module_with_call(tmp_path, '= <path_exists(`"."`)>')

    with pytest.raises(Exception, match="Invalid KEDI_WORKSPACE_POLICY.*none\\|strict"):
        _compile_file(main).run_main()


def _compile_fs_module_with_call(tmp_path: Path, call: str) -> Path:
    main = tmp_path / "main.kedi"
    main.write_text(f"> import: filesystem\n\n{call}\n", encoding="utf-8")
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


def test_internal_sandbox_executes_code_with_inputs(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: sandbox\n\n"
        '[result: Any] = `execute_code("sum(values)", {"values": [1, 2, 3]})`\n\n'
        "= `result`\n",
        encoding="utf-8",
    )

    assert _compile_file(main).run_main() == 6


def test_internal_sandbox_defaults_to_empty_inputs(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: sandbox\n\n[result: int] = `execute_code("1 + 2")`\n\n= `result`\n',
        encoding="utf-8",
    )

    assert _compile_file(main).run_main() == 3


def test_internal_sandbox_can_return_execution_errors(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        '> import: sandbox\n\n[result: str] = `execute_code("missing_name")`\n\n= `result`\n',
        encoding="utf-8",
    )

    assert "missing_name" in _compile_file(main).run_main()


def test_internal_sandbox_can_fail_fast_explicitly(tmp_path: Path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: sandbox\n\n"
        '[result: str] = `execute_code("missing_name", {}, True)`\n\n'
        "= `result`\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="Sandbox execution failed"):
        _compile_file(main).run_main()


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
            "remove_file",
            "remove_directory",
            "get_file_info",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
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


def test_module_registry_builds_relative_submodule_path(tmp_path: Path) -> None:
    source = tmp_path / "main.kedi"

    assert (
        module_search_paths("package/service", source_path=source)[0]
        == (tmp_path / "package" / "service.kedi").resolve()
    )


def test_package_manifest_reads_fixed_python_version_and_dependencies(tmp_path: Path) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n"
        "  version: 1.0\n"
        "  source: src/kedi_http\n"
        "  python: python@3.11\n"
        "  python_dependencies:\n"
        "    httpx>=0.27\n"
        "    pydantic>=2\n",
        encoding="utf-8",
    )

    metadata = load_package_manifest(manifest)

    assert metadata.name == "kedi_http"
    assert metadata.python_requirement == "python@3.11"
    assert metadata.python_dependencies == ["httpx>=0.27", "pydantic>=2"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "not a version", "invalid package version"),
        ("python_dependencies", "not a valid requirement @", "invalid Python dependency"),
    ],
)
def test_package_manifest_rejects_semantically_invalid_metadata(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    if field == "python_dependencies":
        field_source = f"  {field}:\n    {value}\n"
    else:
        field_source = f"  {field}: {value}\n"
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n" + field_source,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_package_manifest(manifest)


def test_install_package_rejects_python_incompatible_manifest(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    (source / "main.kedi").write_text("= done\n", encoding="utf-8")
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n  python: python@3.99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires Python"):
        install_package(package / "package.kedi", registry_root=tmp_path / "registry")


def test_parser_rejects_duplicate_selective_import_names() -> None:
    source = "> import: profiles:\n  get_profile\n  get_profile\n"

    with pytest.raises(KediParseError, match="duplicate selective import name"):
        parse_program(source, source_path="/tmp/main.kedi")


def test_package_directive_is_only_legal_in_package_manifest() -> None:
    source = "> package: demo:\n  source: src/demo\n"

    with pytest.raises(KediParseError, match="only allowed in package.kedi"):
        parse_program(source, source_path="/tmp/main.kedi")

    assert parse_program(source, source_path="/tmp/package.kedi").package is not None


def test_package_manifest_rejects_type_definitions(tmp_path: Path) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        "> package: demo:\n  source: src/demo\n~Config(name: str)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="package metadata only"):
        load_package_manifest(manifest)


@pytest.mark.parametrize(
    ("filename", "source", "message"),
    [
        ("manifest.kedi", "> package: demo:\n", "must be named"),
        ("package.kedi", "# metadata is required\n", "does not define a package"),
    ],
)
def test_package_manifest_requires_its_name_and_package_directive(
    tmp_path: Path, filename: str, source: str, message: str
) -> None:
    manifest = tmp_path / filename
    manifest.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_package_manifest(manifest)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> package: first:\n  version: 1.0\n> package: second:\n  version: 1.0\n",
            "only one package",
        ),
        (
            "> package: demo:\n  unknown: value\n",
            "unknown package metadata field",
        ),
        (
            "> package: demo:\n  version: 1.0\n  version: 2.0\n",
            "duplicate package metadata field",
        ),
        (
            "> package: demo:\n  python_dependencies:\n    httpx\n  python_dependencies:\n    pydantic\n",
            "only one python_dependencies",
        ),
    ],
)
def test_package_parser_rejects_invalid_manifest_metadata(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source, source_path="/tmp/package.kedi")


@pytest.mark.parametrize("value", ["python@3", ">=3.11", "python@3.14-3.11"])
def test_package_manifest_rejects_invalid_python_requirements(tmp_path: Path, value: str) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        f"> package: kedi_http:\n  source: src/kedi_http\n  python: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(KediParseError, match="python"):
        load_package_manifest(manifest)


def test_installed_package_resolves_root_and_submodule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package-source"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  version: 1.0\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text(
        "@root() -> str:\n  = root\n> export:\n  root\n",
        encoding="utf-8",
    )
    (source / "client.kedi").write_text(
        "@request() -> str:\n  = client\n> export:\n  request\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))
    install_package(package / "package.kedi")
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: kedi_http\n> import: kedi_http/client\n= <root()>:<request()>\n",
        encoding="utf-8",
    )

    assert _compile_file(main).run_main() == "root:client"


def test_package_registry_defaults_to_the_user_kedi_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEDI_HOME", raising=False)

    assert kedi_home() == Path.home() / ".kedi"
    assert installed_package_root("kedi_http") == Path.home() / ".kedi" / "registry" / "kedi_http"


def test_installed_module_path_rejects_invalid_or_incomplete_package_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "kedi-home" / "registry"
    invalid = registry / "invalid"
    incomplete = registry / "incomplete"
    invalid.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (invalid / "package.kedi").write_text("# no package\n", encoding="utf-8")
    (incomplete / "package.kedi").write_text(
        "> package: incomplete:\n  version: 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    with pytest.raises(CorruptInstalledPackageError, match="invalid package manifest"):
        installed_module_path("invalid")
    with pytest.raises(CorruptInstalledPackageError, match="unsafe source"):
        installed_module_path("incomplete")


def test_corrupt_installed_package_does_not_shadow_a_local_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_package = tmp_path / "kedi-home" / "registry" / "profiles"
    registry_package.mkdir(parents=True)
    (registry_package / "package.kedi").write_text("# corrupt\n", encoding="utf-8")
    local = tmp_path / "profiles.kedi"
    local.write_text("= local\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text("> import: profiles\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    assert resolve_module_path("profiles", source_path=main) == local.resolve()


@pytest.mark.parametrize(
    ("source_value", "setup_source", "message"),
    [
        (None, False, "requires a source"),
        ("../outside", False, "relative path"),
        ("/outside", False, "relative path"),
        ("src/kedi_http", False, "source directory not found"),
        ("src/kedi_http", True, "must contain main.kedi"),
    ],
)
def test_install_package_rejects_invalid_source_layout(
    tmp_path: Path,
    source_value: str | None,
    setup_source: bool,
    message: str,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    metadata = "> package: kedi_http:\n  version: 1.0\n"
    if source_value is not None:
        metadata += f"  source: {source_value}\n"
    (package / "package.kedi").write_text(metadata, encoding="utf-8")
    if setup_source:
        (package / "src" / "kedi_http").mkdir(parents=True)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        install_package(package / "package.kedi", registry_root=tmp_path / "registry")


def test_install_package_rejects_sparse_checkout_pattern_in_source(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  source: src/*\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sparse-checkout pattern"):
        install_package(package / "package.kedi", registry_root=tmp_path / "registry")


def test_install_package_replaces_an_existing_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    main = source / "main.kedi"
    main.write_text("= first\n", encoding="utf-8")
    registry = tmp_path / "registry"
    install_package(manifest, registry_root=registry)

    main.write_text("= second\n", encoding="utf-8")
    installed = install_package(manifest, registry_root=registry)

    assert (installed.source / "main.kedi").read_text(encoding="utf-8") == "= second\n"
    assert not list(registry.glob(".kedi_http.backup-*"))


def test_install_package_restores_previous_package_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    main = source / "main.kedi"
    main.write_text("= first\n", encoding="utf-8")
    registry = tmp_path / "registry"
    install_package(manifest, registry_root=registry)
    main.write_text("= second\n", encoding="utf-8")

    original_replace = Path.replace

    def fail_staging_replace(path: Path, target: Path) -> Path:
        if ".kedi_http.staging-" in path.name:
            raise OSError("replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_replace)

    with pytest.raises(OSError, match="replace failed"):
        install_package(manifest, registry_root=registry)

    restored = registry / "kedi_http" / "src" / "kedi_http" / "main.kedi"
    assert restored.read_text(encoding="utf-8") == "= first\n"
    assert not list(registry.glob(".kedi_http.staging-*"))


def test_install_package_keeps_successful_swap_when_backup_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    main = source / "main.kedi"
    main.write_text("= first\n", encoding="utf-8")
    registry = tmp_path / "registry"
    install_package(manifest, registry_root=registry)
    main.write_text("= second\n", encoding="utf-8")

    original_rmtree = package_registry.shutil.rmtree

    def fail_backup_cleanup(path: Path) -> None:
        if ".kedi_http.backup-" in path.name:
            raise OSError("backup cleanup failed")
        original_rmtree(path)

    monkeypatch.setattr(package_registry.shutil, "rmtree", fail_backup_cleanup)

    installed = install_package(manifest, registry_root=registry)

    installed_main = installed.source / "main.kedi"
    assert installed_main.read_text(encoding="utf-8") == "= second\n"
    assert list(registry.glob(".kedi_http.backup-*"))


def test_add_package_installs_from_mock_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "mock-registry" / "kedi_http"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  version: 1.0\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= installed\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    installed = add_package("kedi_http", client=MockPackageRegistryClient(package.parent))

    assert (
        installed.source == tmp_path / "kedi-home" / "registry" / "kedi_http" / "src" / "kedi_http"
    )


def test_add_package_requires_a_configured_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEDI_REGISTRY_MOCK_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="public Kedi package registry"):
        add_package("kedi_http")


def test_mock_registry_rejects_missing_package(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError, match="not available"):
        MockPackageRegistryClient(tmp_path).package_root("kedi_http")


def test_add_package_rejects_registry_name_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "mock-registry" / "requested"
    source = package / "src" / "different"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: different:\n  source: src/different\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= installed\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    with pytest.raises(ValueError, match="name mismatch"):
        add_package("requested", client=MockPackageRegistryClient(package.parent))

    assert not (tmp_path / "kedi-home" / "registry" / "different").exists()


@pytest.mark.parametrize("link_kind", ["source", "file", "directory"])
def test_install_package_rejects_source_tree_symlinks(tmp_path: Path, link_kind: str) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.kedi").write_text("= outside\n", encoding="utf-8")
    (outside / "secret.kedi").write_text("= secret\n", encoding="utf-8")
    (package / "package.kedi").parent.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n", encoding="utf-8"
    )

    if link_kind == "source":
        source.parent.mkdir(parents=True)
        source.symlink_to(outside, target_is_directory=True)
    else:
        source.mkdir(parents=True)
        (source / "main.kedi").write_text("= safe\n", encoding="utf-8")
        if link_kind == "file":
            (source / "secret.kedi").symlink_to(outside / "secret.kedi")
        else:
            (source / "nested").symlink_to(outside, target_is_directory=True)

    registry = tmp_path / "registry"
    with pytest.raises(ValueError, match="symlink"):
        install_package(package / "package.kedi", registry_root=registry)
    assert not (registry / "kedi_http").exists()


def test_install_package_rejects_special_files_in_source_tree(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n", encoding="utf-8"
    )
    (source / "main.kedi").write_text("= safe\n", encoding="utf-8")
    os.mkfifo(source / "unexpected.pipe")

    with pytest.raises(ValueError, match="regular files and directories"):
        install_package(package / "package.kedi", registry_root=tmp_path / "registry")


def test_install_package_rejects_package_root_source_cleanly(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "package.kedi").write_text("> package: kedi_http:\n  source: .\n", encoding="utf-8")
    (package / "main.kedi").write_text("= safe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative path"):
        install_package(package / "package.kedi", registry_root=tmp_path / "registry")


def test_mock_registry_rejects_path_traversal_before_lookup(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "evil"
    outside.mkdir(parents=True)
    (outside / "package.kedi").write_text("> package: evil:\n  source: src\n", encoding="utf-8")

    with pytest.raises(ValueError, match="package names"):
        MockPackageRegistryClient(tmp_path / "mock-registry").package_root("../outside/evil")


def test_installed_module_path_rejects_tampered_manifest_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "kedi-home" / "registry"
    package = registry / "evil"
    outside = tmp_path / "outside"
    package.mkdir(parents=True)
    outside.mkdir()
    (outside / "main.kedi").write_text("= escaped\n", encoding="utf-8")
    (package / "package.kedi").write_text(
        "> package: evil:\n  source: ../../outside\n", encoding="utf-8"
    )
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    with pytest.raises(CorruptInstalledPackageError, match="unsafe source"):
        installed_module_path("evil")


def test_installed_module_path_rejects_tampered_symlinked_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "kedi-home" / "registry"
    package = registry / "evil"
    source = package / "src"
    outside = tmp_path / "outside.kedi"
    source.mkdir(parents=True)
    outside.write_text("= escaped\n", encoding="utf-8")
    (package / "package.kedi").write_text("> package: evil:\n  source: src\n", encoding="utf-8")
    (source / "main.kedi").symlink_to(outside)
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    with pytest.raises(CorruptInstalledPackageError, match="symlink"):
        installed_module_path("evil")


def test_concurrent_package_replacement_is_serialized(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    manifests: list[Path] = []
    for index in range(12):
        package = tmp_path / f"package-{index}"
        source = package / "src" / "kedi_http"
        source.mkdir(parents=True)
        manifest = package / "package.kedi"
        manifest.write_text("> package: kedi_http:\n  source: src/kedi_http\n", encoding="utf-8")
        (source / "main.kedi").write_text(f"= {index}\n", encoding="utf-8")
        manifests.append(manifest)

    with ThreadPoolExecutor(max_workers=len(manifests)) as executor:
        installed = list(
            executor.map(
                lambda manifest: install_package(manifest, registry_root=registry), manifests
            )
        )

    assert all(item.root == registry / "kedi_http" for item in installed)
    assert (registry / "kedi_http" / "src" / "kedi_http" / "main.kedi").is_file()
    assert not list(registry.glob(".kedi_http.staging-*"))


def test_installed_module_path_caches_manifest_until_an_install_invalidates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text("> package: kedi_http:\n  source: src/kedi_http\n", encoding="utf-8")
    (source / "main.kedi").write_text("= first\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))
    install_package(manifest)

    module_registry.clear_installed_module_cache()
    # ``installed_module_path`` imports this dependency lazily, so patch its source module.
    from kedi import package_manifest

    calls = 0
    real_load = package_manifest.load_package_manifest

    def count_load(path: Path):
        nonlocal calls
        calls += 1
        return real_load(path)

    monkeypatch.setattr(package_manifest, "load_package_manifest", count_load)
    assert installed_module_path("kedi_http") is not None
    assert installed_module_path("kedi_http") is not None
    assert calls == 1

    (source / "main.kedi").write_text("= second\n", encoding="utf-8")
    install_package(manifest)
    assert installed_module_path("kedi_http").read_text(encoding="utf-8") == "= second\n"


def test_relative_kedi_home_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KEDI_HOME", "relative-kedi-home")

    with pytest.raises(ValueError, match="KEDI_HOME must be an absolute path"):
        kedi_home()


def test_add_git_package_clones_then_installs_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subprocess import CompletedProcess

    sparse_patterns: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        if args[1] == "clone":
            assert "--filter=blob:none" in args
            assert "--no-checkout" in args
            assert "--no-tags" in args
            checkout = Path(args[-1])
            source = checkout / "src" / "kedi_http"
            source.mkdir(parents=True)
            (checkout / "package.kedi").write_text(
                "> package: kedi_http:\n  source: src/kedi_http\n",
                encoding="utf-8",
            )
            (source / "main.kedi").write_text("= ok\n", encoding="utf-8")
            return CompletedProcess(args, 0, stdout="", stderr="")
        if args[3:5] == ["sparse-checkout", "set"]:
            sparse_patterns.append(args[6:])
            return CompletedProcess(args, 0, stdout="", stderr="")
        if args[3:] == ["checkout", "--quiet"]:
            return CompletedProcess(args, 0, stdout="", stderr="")
        assert args[3:] == ["rev-parse", "HEAD"]
        return CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr("kedi.package_registry.subprocess.run", fake_run)
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    installed = add_package("git+https://github.com/kedi-lang/kedi-http.git")

    assert installed.name == "kedi_http"
    assert installed.source_revision == "abc123"
    assert (installed.source / "main.kedi").is_file()
    assert sparse_patterns == [
        ["/package.kedi"],
        ["/package.kedi", "/src/kedi_http/**"],
    ]
    receipt = json.loads((installed.root / ".kedi-install.json").read_text(encoding="utf-8"))
    assert receipt == {
        "format_version": 1,
        "install_source_kind": "git",
        "manifest_sha256": hashlib.sha256(
            (installed.root / "package.kedi").read_bytes()
        ).hexdigest(),
        "package": "kedi_http",
        "source": "src/kedi_http",
        "source_revision": "abc123",
        "source_url": "https://github.com/kedi-lang/kedi-http.git",
    }


def test_install_git_package_rejects_a_symlinked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subprocess import CompletedProcess

    outside_manifest = tmp_path / "outside-package.kedi"
    outside_manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )

    def fake_run(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        if args[1] == "clone":
            checkout = Path(args[-1])
            checkout.mkdir()
            (checkout / "package.kedi").symlink_to(outside_manifest)
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(package_registry.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="package manifest must not be a symlink"):
        install_git_package("git+https://github.com/kedi-lang/kedi-http.git")


@pytest.mark.parametrize(
    "source",
    [
        "git+ssh://github.com/kedi-lang/package.git",
        "https://example.com/x",
        "git+https://token@example.com/package.git",
        "git+https://example.com/package.git?token=secret",
        "git+https://gitlab.com/kedi-lang/package.git",
    ],
)
def test_install_git_package_rejects_non_https_sources(source: str) -> None:
    with pytest.raises(ValueError, match="Git package sources"):
        install_git_package(source)


@pytest.mark.parametrize(
    ("stdout", "stderr", "message"),
    [
        ("", "clone failed", "clone failed"),
        ("clone failed", "", "clone failed"),
        ("", "", "unknown git error"),
    ],
)
def test_install_git_package_reports_git_errors(
    monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str, message: str
) -> None:
    from subprocess import CompletedProcess

    monkeypatch.setattr(
        package_registry.subprocess,
        "run",
        lambda args, **_kwargs: CompletedProcess(args, 1, stdout=stdout, stderr=stderr),
    )

    with pytest.raises(RuntimeError, match=message):
        install_git_package("git+https://github.com/kedi-lang/kedi-http.git")


def test_install_git_package_redacts_credentials_from_git_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    monkeypatch.setattr(
        package_registry.subprocess,
        "run",
        lambda args, **_kwargs: CompletedProcess(
            args,
            1,
            stdout="",
            stderr="fatal: https://token:secret@example.com/private.git failed",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        install_git_package("git+https://github.com/kedi-lang/kedi-http.git")

    assert "token" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)
    assert "https://<redacted>@example.com" in str(exc_info.value)


@pytest.mark.parametrize(
    ("limit_name", "limit", "files"),
    [
        ("MAX_PACKAGE_SOURCE_FILES", 1, {"extra.kedi": "= extra\n"}),
        ("MAX_PACKAGE_SOURCE_BYTES", 1, {}),
    ],
)
def test_install_package_enforces_source_resource_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    files: dict[str, str],
) -> None:
    package = tmp_path / "package"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= main\n", encoding="utf-8")
    for name, content in files.items():
        (source / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(package_registry, limit_name, limit)

    with pytest.raises(ValueError, match="exceeds the maximum"):
        install_package(manifest, registry_root=tmp_path / "registry")

    assert not (tmp_path / "registry" / "kedi_http").exists()


def test_install_git_package_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(package_registry.subprocess, "run", missing_git)

    with pytest.raises(ModuleNotFoundError, match="Git is required"):
        install_git_package("git+https://github.com/kedi-lang/kedi-http.git")


def test_install_git_package_times_out_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise package_registry.subprocess.TimeoutExpired("git clone", 60)

    monkeypatch.setattr(package_registry.subprocess, "run", timed_out)

    with pytest.raises(RuntimeError, match="did not finish"):
        install_git_package("git+https://github.com/kedi-lang/kedi-http.git")


def test_package_manifest_validation_covers_invalid_ranges_and_compatible_runtime() -> None:
    with pytest.raises(ValueError, match="package python must"):
        parse_python_requirement("python@3")
    with pytest.raises(ValueError, match="end at or after"):
        parse_python_requirement("python@3.14-3.11")

    validate_package_metadata(
        PackageMetadata(name="demo", source="src/demo", python_requirement="python@3.11-3.14"),
        enforce_python_compatibility=True,
        python_version=(3, 12),
    )


def test_installed_module_path_rejects_corrupt_package_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "kedi-home" / "registry"
    registry.mkdir(parents=True)
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    (registry / "missing").mkdir()
    with pytest.raises(CorruptInstalledPackageError, match="missing package.kedi"):
        installed_module_path("missing")

    outside = tmp_path / "outside"
    outside.mkdir()
    (registry / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CorruptInstalledPackageError, match="must not be a symlink"):
        installed_module_path("linked")

    manifest_link = registry / "manifest_link"
    manifest_link.mkdir()
    outside_manifest = tmp_path / "outside-package.kedi"
    outside_manifest.write_text("> package: manifest_link:\n  source: src\n", encoding="utf-8")
    (manifest_link / "package.kedi").symlink_to(outside_manifest)
    with pytest.raises(CorruptInstalledPackageError, match="symlinked package manifest"):
        installed_module_path("manifest_link")

    mismatch = registry / "mismatch"
    mismatch.mkdir()
    (mismatch / "package.kedi").write_text("> package: another:\n  source: src\n", encoding="utf-8")
    with pytest.raises(CorruptInstalledPackageError, match="does not match"):
        installed_module_path("mismatch")


def test_installed_module_path_rejects_a_registry_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        package_registry, "installed_package_root", lambda *_args, **_kwargs: outside
    )

    with pytest.raises(CorruptInstalledPackageError, match="escapes the package registry"):
        module_registry._installed_module_path_cached.__wrapped__("evil", tmp_path / "registry")


def test_package_registry_defensive_helpers_reject_invalid_files_and_paths(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    source = ResolvedPackageSource(tmp_path, Path("source"), source_path)

    with pytest.raises(CorruptInstalledPackageError, match="escapes"):
        package_registry.resolve_installed_package_file(source, Path("../outside.kedi"))
    assert package_registry.resolve_installed_package_file(source, Path("missing.kedi")) is None

    manifest = tmp_path / "package.kedi"
    manifest.write_text("> package: demo:\n  source: src/demo\n", encoding="utf-8")
    symlink = tmp_path / "linked-package.kedi"
    symlink.symlink_to(manifest)
    with pytest.raises(ValueError, match="must not be a symlink"):
        package_registry.prepare_package(symlink)

    with pytest.raises(ValueError, match="Unknown Kedi package installation source kind"):
        install_package(manifest, registry_root=tmp_path / "registry", source_kind="remote")

    with pytest.raises(ValueError, match="not a directory"):
        package_registry._copy_source_tree(manifest, tmp_path / "destination")
    with pytest.raises(FileNotFoundError, match="not found"):
        package_registry._require_regular_file(
            tmp_path / "absent", "test file", error_type=ValueError
        )
    with pytest.raises(ValueError, match="regular file"):
        package_registry._require_regular_file(source_path, "test directory", error_type=ValueError)
    with pytest.raises(ValueError, match="remain inside"):
        package_registry._require_contained(tmp_path / "outside", source_path, "test path")
    with pytest.raises(ValueError, match="package directory not found"):
        package_registry._validated_package_root(tmp_path / "absent-root", error_type=ValueError)


def test_package_registry_copy_and_cleanup_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "main.kedi").write_text("= main\n", encoding="utf-8")
    (nested / "extra.kedi").write_text("= extra\n", encoding="utf-8")
    copied = tmp_path / "copied"
    package_registry._copy_source_tree(source, copied)
    assert (copied / "nested" / "extra.kedi").is_file()

    file_source = source / "main.kedi"
    original_fstat = package_registry.os.fstat
    monkeypatch.setattr(
        package_registry.os,
        "fstat",
        lambda descriptor: SimpleNamespace(st_mode=stat.S_IFIFO, st_size=0),
    )
    with pytest.raises(ValueError, match="not a regular file"):
        package_registry._copy_regular_file(file_source, tmp_path / "bad-copy.kedi")
    monkeypatch.setattr(package_registry.os, "fstat", original_fstat)

    removable = tmp_path / "removable.kedi"
    removable.write_text("= removable\n", encoding="utf-8")
    with monkeypatch.context() as context:
        context.setattr(
            Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no"))
        )
        with pytest.raises(OSError, match="no"):
            package_registry._remove_tree(removable)
        package_registry._remove_tree(removable, ignore_errors=True)
    removable.unlink()

    removable_directory = tmp_path / "removable-directory"
    removable_directory.mkdir()
    with monkeypatch.context() as context:
        context.setattr(
            package_registry.shutil,
            "rmtree",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no")),
        )
        with pytest.raises(OSError, match="no"):
            package_registry._remove_tree(removable_directory)
        package_registry._remove_tree(removable_directory, ignore_errors=True)


def test_package_registry_handles_uncreated_staging_and_truncated_git_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    source = package / "src" / "demo"
    source.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text("> package: demo:\n  source: src/demo\n", encoding="utf-8")
    (source / "main.kedi").write_text("= ok\n", encoding="utf-8")

    original_mkdir = Path.mkdir

    def reject_staging(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".demo.staging-"):
            raise OSError("staging unavailable")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reject_staging)
    with pytest.raises(OSError, match="staging unavailable"):
        install_package(manifest, registry_root=tmp_path / "registry")

    monkeypatch.setattr(
        package_registry.subprocess,
        "run",
        lambda args, **_kwargs: package_registry.subprocess.CompletedProcess(
            args, 1, stdout="", stderr="x" * (package_registry.MAX_GIT_ERROR_CHARS + 1)
        ),
    )
    with pytest.raises(RuntimeError) as exc_info:
        install_git_package("git+https://github.com/kedi-lang/demo.git")
    assert len(str(exc_info.value)) <= package_registry.MAX_GIT_ERROR_CHARS + 120


def test_static_import_analysis_covers_dynamic_exports_and_profile_helpers(tmp_path: Path) -> None:
    compiler = ProgramCompiler(None)
    loop = tmp_path / "loop.kedi"
    loop.write_text("= done\n", encoding="utf-8")
    loop_program = parse_program(loop.read_text(encoding="utf-8"), source_path=str(loop))
    with pytest.raises(RuntimeError, match="Cyclic Kedi import"):
        ProgramCompiler(None, module_stack=(loop.resolve(),)).compile(loop_program)

    assert compiler._static_python_names("not valid python(") == (set(), False)
    names, complete = compiler._static_python_names(
        "def fn(): pass\nclass Type: pass\nimport os.path as path\nfrom mod import value\n"
    )
    assert names == {"fn", "Type", "path", "value"}
    assert complete
    assert compiler._static_python_names("from mod import *\n") == (set(), False)

    source = "[value: str] = value\n[raw] << raw\n>> Block [output].\n"
    names, complete = compiler._static_local_runtime_names(parse_program(source))
    assert names == {"value", "raw", "output"}
    assert complete

    inline_output = Program(
        toplevel_body=[
            ModelDirectiveStmt(plain_value="test-model"),
            TemplateLine(segments=[OutputSegment(name="inline_output")]),
            TemplateLine(segments=[OutputSegment(name="another_output")]),
        ]
    )
    names, complete = compiler._static_local_runtime_names(inline_output)
    assert names == {"inline_output", "another_output"}
    assert complete

    module = tmp_path / "module.kedi"
    module.write_text("> import: module\n> export: *\n", encoding="utf-8")
    program = parse_program(module.read_text(encoding="utf-8"), source_path=str(module))
    assert not compiler._static_export_surface(module, program, visiting=set()).complete

    selective = parse_program("> import: values:\n  missing\n")
    with pytest.raises(ImportError, match="does not export"):
        compiler._selected_import_names(selective.imports[0], ["present"])

    profile = AgentProfile(tools=("tool",), subagents=("child",))
    assert compiler._bind_imported_profile(profile, SimpleNamespace()) is profile
    assert compiler._profile_tool_names(profile) == ("tool",)
    assert compiler._profile_subagent_names(profile) == ("child",)


def test_profile_binding_skips_missing_private_subagent_and_merges_bindings() -> None:
    from kedi.agent_profile.definitions import ProfileDefinition, ProfileSubagentMember

    compiler = ProgramCompiler(None)
    definition = ProfileDefinition(
        name="parent",
        members=[ProfileSubagentMember(name="missing")],
    )
    bound = compiler._bind_imported_profile(
        definition,
        SimpleNamespace(_profile_registry={}),
    )
    child = ProfileDefinition(name="child")
    merged = AgentProfile().merge(AgentProfile(subagent_bindings={"child": child}))

    assert bound.subagent_bindings == {}
    assert merged.subagents == ()
    assert merged.subagent_bindings == {"child": child}


def test_package_registry_handles_commit_collision_and_platform_copy_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    target = root / "demo"
    target.mkdir()
    (target / "old.kedi").write_text("= old\n", encoding="utf-8")
    staging = root / "staging"
    staging.mkdir()
    (staging / "main.kedi").write_text("= new\n", encoding="utf-8")
    original_replace = Path.replace

    def fail_staging_commit(path: Path, destination: Path) -> Path:
        if path == staging:
            destination.mkdir()
            raise OSError("commit collision")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_staging_commit)
    with pytest.raises(OSError, match="commit collision"):
        package_registry._commit_staged_package(root, "demo", staging, target)
    assert target.is_dir()

    source = tmp_path / "source.kedi"
    source.write_text("= copied\n", encoding="utf-8")
    monkeypatch.delattr(package_registry.os, "O_NOFOLLOW", raising=False)
    package_registry._copy_regular_file(source, tmp_path / "copied.kedi")
    assert (tmp_path / "copied.kedi").read_text(encoding="utf-8") == "= copied\n"
