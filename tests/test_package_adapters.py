"""Packages expose adapters: `> import: pi`, then `> agent: pi` (#232).

The packages here are installed into a temporary registry. Their bridge is Kedi's JSON-RPC
test peer, copied into the package source, so the command form runs a real bridge.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.module_registry import clear_installed_module_cache
from kedi.package_registry import install_package

FIXTURES = Path(__file__).parent / "fixtures"
PEER = FIXTURES / "jsonrpc_peer.py"
PROMPT = "[reply] << Say hello.\n= <reply>\n"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "kedi-home"
    monkeypatch.setenv("KEDI_HOME", str(home))
    clear_installed_module_cache()
    yield home
    clear_installed_module_cache()


def install(tmp_path: Path, name: str, adapters: str, *, main: str = "") -> Path:
    """Installs package ``name``, its source holding the test peer as bridge.py."""

    root = tmp_path / "packages" / name
    source = root / "src" / name
    source.mkdir(parents=True)
    (source / "main.kedi").write_text(main or f"# The {name} package.\n", encoding="utf-8")
    shutil.copy(PEER, source / "bridge.py")
    manifest = root / "package.kedi"
    manifest.write_text(
        f"> package: {name}:\n  source: src/{name}\n  adapters:\n{adapters}", encoding="utf-8"
    )
    install_package(manifest)
    clear_installed_module_cache()
    return root


def bridge(name: str, kind: str = "agent", scenario: str = "default") -> str:
    return (
        f"    {name}:\n      kind: {kind}\n      command: {sys.executable} bridge.py {scenario}\n"
    )


def run(source: str, *, path: Path | None = None) -> Any:
    runtime = compile_program(parse_program(source, source_path=str(path) if path else None))
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def test_an_imported_package_adapter_launches_the_bridge_it_declares(
    registry: Path, tmp_path: Path
) -> None:
    install(tmp_path, "pi", bridge("pi"))
    assert run("> import: pi\n> agent: pi\n" + PROMPT) == "echo: Say hello."
    # One bridge serves every call of the runtime.
    source = "> import: pi\n> agent: pi\n[a] << One.\n[b] << Two.\n= `[a, b]`\n"
    runtime = compile_program(parse_program(source))
    try:
        assert runtime.run_main() == ["echo: One.", "echo: Two."]
        assert len(runtime._package_adapter_instances) == 1
    finally:
        asyncio.run(runtime.aclose())


def test_a_package_adapter_takes_no_connection_body(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", bridge("pi"))
    with pytest.raises(Exception, match="comes from package 'pi', which says how to launch it"):
        run("> import: pi\n> agent: pi:\n    command: node other.mjs\n" + PROMPT)


def test_a_profile_snapshot_keeps_its_package_adapter() -> None:
    import json

    from kedi._interactive import persistence
    from kedi._interactive.persistence import SessionValueDecoder, SessionValueEncoder
    from kedi.agent_profile import AgentProfile
    from kedi.package_adapters import PackageAdapter

    adapter = PackageAdapter(
        package="pi", name="pi", kind="agent", command=("node", "/x/bridge.mjs")
    )
    profile = AgentProfile(
        adapter_kind="agent-harness", adapter_shortname="pi", package_adapter=adapter
    )
    encoded = persistence.encode_profile(
        profile, encoder=SessionValueEncoder(kedi_models={}), model_names={}, path="profile"
    )
    decoded = persistence.decode_profile(
        json.loads(json.dumps(encoded)), decoder=SessionValueDecoder(kedi_models={}), kedi_models={}
    )
    assert decoded.package_adapter == adapter
    with pytest.raises(Exception, match="Invalid package adapter"):
        persistence.decode_profile(
            {**json.loads(json.dumps(encoded)), "package_adapter": {"kind": "robot"}},
            decoder=SessionValueDecoder(kedi_models={}),
            kedi_models={},
        )


@pytest.mark.parametrize(
    ("adapters", "message"),
    [
        (
            "    pi:\n      kind: agent\n    pi:\n      kind: agent\n",
            "duplicate package adapter 'pi'",
        ),
        ("    pi:\n      kind: agent\n      shell: x\n", "unknown package adapter field 'shell'"),
        (
            "    pi:\n      kind: agent\n      kind: agent\n",
            "duplicate package adapter field 'kind'",
        ),
    ],
)
def test_the_parser_reports_malformed_adapter_entries(adapters: str, message: str) -> None:
    with pytest.raises(Exception, match=message):
        parse_program(f"> package: pi:\n  source: src/pi\n  adapters:\n{adapters}")


def test_a_manifest_has_one_adapters_block() -> None:
    entry = "  adapters:\n    pi:\n      kind: agent\n      command: node b.mjs\n"
    with pytest.raises(Exception, match="only one adapters block"):
        parse_program("> package: pi:\n  source: src/pi\n" + entry + entry)


def test_the_bridge_command_names_the_installed_file(registry: Path, tmp_path: Path) -> None:
    from kedi.package_adapters import package_adapters

    install(tmp_path, "pi", bridge("pi"))
    (adapter,) = package_adapters("pi")
    assert adapter.command == (
        sys.executable,
        str((registry / "registry" / "pi" / "src" / "pi" / "bridge.py").resolve()),
        "default",
    )


def test_without_the_import_the_name_is_unknown(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", bridge("pi"))
    with pytest.raises(Exception, match="unknown agent harness adapter 'pi'"):
        run("> agent: pi\n" + PROMPT)


def test_an_agent_is_not_an_adapter(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", bridge("pi"))
    with pytest.raises(
        Exception, match="package 'pi' declares it as an agent, selected with > agent:"
    ):
        run("> import: pi\n> adapter: pi\n" + PROMPT)


def test_a_framework_bridge_is_selected_as_an_adapter(registry: Path, tmp_path: Path) -> None:
    # `> agent: rpc:` takes harnesses only; a package declares a framework bridge as an adapter.
    install(tmp_path, "pi", bridge("pi_framework", kind="adapter", scenario="framework"))
    assert run("> import: pi\n> adapter: pi_framework\n" + PROMPT) == "echo: Say hello."


def test_a_bridge_of_another_kind_than_declared_is_refused(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", bridge("pi", scenario="framework"))
    with pytest.raises(
        Exception, match="'pi' from package 'pi' is declared an agent, but it is an adapter"
    ):
        run("> import: pi\n> agent: pi\n" + PROMPT)


def test_a_class_adapter_is_built_from_the_package_python(
    registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(FIXTURES))
    install(
        tmp_path,
        "pi",
        "    pi:\n      kind: agent\n      class: package_adapter_class:PeerAdapter\n",
    )
    assert run("> import: pi\n> agent: pi\n" + PROMPT) == "echo: Say hello."


def test_a_class_of_another_kind_than_declared_is_refused(
    registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(FIXTURES))
    install(
        tmp_path,
        "pi",
        "    pi:\n      kind: agent\n      class: package_adapter_class:FrameworkStub\n",
    )
    with pytest.raises(Exception, match="is declared an agent, but it is an adapter"):
        run("> import: pi\n> agent: pi\n" + PROMPT)


def test_a_class_that_cannot_be_imported_names_the_package(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", "    pi:\n      kind: agent\n      class: no_such_module:Adapter\n")
    with pytest.raises(
        Exception, match="'pi' from package 'pi' names class 'no_such_module:Adapter'"
    ):
        run("> import: pi\n> agent: pi\n" + PROMPT)


def test_a_missing_program_is_reported_before_launch(registry: Path, tmp_path: Path) -> None:
    install(
        tmp_path, "pi", "    pi:\n      kind: agent\n      command: kedi-no-such-node bridge.py\n"
    )
    with pytest.raises(
        Exception, match="runs 'kedi-no-such-node', which is not installed or not on PATH"
    ):
        run("> import: pi\n> agent: pi\n" + PROMPT)


def test_two_packages_declaring_one_name_fail_at_import(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi_a", bridge("shared"))
    install(tmp_path, "pi_b", bridge("shared"))
    with pytest.raises(
        Exception, match="packages 'pi_a' and 'pi_b' both declare the adapter 'shared'"
    ):
        run("> import: pi_a\n> import: pi_b\n" + PROMPT)


def test_a_package_and_its_submodule_are_one_package(registry: Path, tmp_path: Path) -> None:
    root = install(tmp_path, "pi", bridge("pi"))
    del root
    source = registry / "registry" / "pi" / "src" / "pi"
    (source / "tools.kedi").write_text(
        "@greet() -> str:\n  = hi\n> export:\n  greet\n", encoding="utf-8"
    )
    clear_installed_module_cache()
    assert run("> import: pi\n> import: pi/tools\n> agent: pi\n" + PROMPT) == "echo: Say hello."


def test_the_name_stays_in_the_importing_module(registry: Path, tmp_path: Path) -> None:
    install(tmp_path, "pi", bridge("pi"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "lib.kedi").write_text(
        "> import: pi\n@ask() -> str:\n  > agent: pi\n  [r] << Say hello.\n  = <r>\n> export:\n  ask\n",
        encoding="utf-8",
    )
    main = project / "main.kedi"
    # A procedure written where pi is imported keeps its adapter when called from elsewhere.
    assert run("> import: lib\n= `ask()`\n", path=main) == "echo: Say hello."
    # Importing lib does not make pi selectable here.
    with pytest.raises(Exception, match="unknown agent harness adapter 'pi'"):
        run("> import: lib\n> agent: pi\n" + PROMPT, path=main)


@pytest.mark.parametrize(
    ("adapters", "message"),
    [
        (
            "    rpc:\n      kind: agent\n      command: node b.mjs\n",
            "belongs to a built-in adapter",
        ),
        ("    pi:\n      kind: agent\n", "needs exactly one of class"),
        ("    pi:\n      kind: harness\n      command: node b.mjs\n", "kind must be agent"),
        ("    Pi:\n      kind: agent\n      command: node b.mjs\n", "a shortname uses lowercase"),
        (
            "    pi:\n      kind: agent\n      class: kedi_pi.PiAdapter\n",
            "class must be module:Class",
        ),
        ("    pi:\n      kind: agent\n      command: node 'b.mjs\n", "command cannot be split"),
        ('    pi:\n      kind: agent\n      command: ""\n', "command names no program"),
    ],
)
def test_an_invalid_adapter_is_refused_at_install(
    registry: Path, tmp_path: Path, adapters: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        install(tmp_path, "pi", adapters)


def test_the_editor_knows_the_imported_names(registry: Path, tmp_path: Path) -> None:
    from lsprotocol import types as lsp

    from kedi.lsp.features.completion import compute_completion

    install(tmp_path, "pi", bridge("pi"))
    imported = "> import: pi\n> agent: pi\n" + PROMPT
    assert [d for d in compute_diagnostics(imported, None) if d.severity == 1] == []
    unknown = [d for d in compute_diagnostics("> agent: pi\n" + PROMPT, None) if d.severity == 1]
    assert [d.code for d in unknown] == ["K-ADAPTER-NOT-FOUND"]
    wrong = [
        d
        for d in compute_diagnostics("> import: pi\n> adapter: pi\n" + PROMPT, None)
        if d.severity == 1
    ]
    assert [d.code for d in wrong] == ["K-ADAPTER-KIND"]
    assert "package 'pi' declares it as an agent" in wrong[0].message
    source = "> import: pi\n> agent: p"
    items = compute_completion(source, None, lsp.Position(line=1, character=len("> agent: p")))
    assert "pi" in [item.label for item in items]


def test_the_editor_skips_what_it_cannot_read(registry: Path, tmp_path: Path) -> None:
    from lsprotocol import types as lsp

    from kedi.lsp.features.completion import compute_completion

    install(tmp_path, "pi", bridge("pi"))
    # A half-typed import and one that resolves nowhere leave the installed package's names.
    source = "> import: pi/\n> import: nowhere\n> import: pi\n> agent: p"
    items = compute_completion(source, None, lsp.Position(line=3, character=len("> agent: p")))
    assert "pi" in [item.label for item in items]
    # A broken installed manifest leaves its adapters out; the import reports it.
    from kedi.package_adapters import program_package_adapters

    (registry / "registry" / "pi" / "package.kedi").write_text("> package: pi:\n", encoding="utf-8")
    clear_installed_module_cache()
    source = "> import: pi\n> agent: pi\n" + PROMPT
    assert program_package_adapters(parse_program(source)).names("agent-harness") == ()
    assert "K-IMPORT-CORRUPT" in [d.code for d in compute_diagnostics(source, None)]
    items = compute_completion(
        "> import: pi\n> agent: p", None, lsp.Position(line=1, character=len("> agent: p"))
    )
    assert "pi" not in [item.label for item in items]
