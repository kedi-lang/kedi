"""``kedi_dependencies``: Kedi packages that need Kedi packages.

Dependencies come from a filesystem registry (``KEDI_REGISTRY_MOCK_ROOT``). An installation
resolves its whole graph before it changes anything, installs dependencies first, and undoes
itself if a step fails; ``kedi remove`` refuses a package that is needed and takes orphaned
dependencies along.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from kedi import cli, package_registry
from kedi.lang import parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.package_audit import audit_installed_packages
from kedi.package_manifest import load_package_manifest, validate_package_metadata
from kedi.package_registry import INSTALL_RECEIPT_NAME, add_package, install_package, remove_package


def package(root: Path, name: str, version: str = "1.0.0", *needs: str, python: str = "") -> Path:
    directory = root / name
    (directory / "src" / name).mkdir(parents=True)
    (directory / "src" / name / "main.kedi").write_text(f"= {name} {version}\n")
    body = f"> package: {name}:\n  version: {version}\n  source: src/{name}\n"
    if needs:
        body += "  kedi_dependencies:\n" + "".join(f"    {need}\n" for need in needs)
    if python:
        body += f"  python_dependencies:\n    {python}\n"
    (directory / "package.kedi").write_text(body)
    return directory / "package.kedi"


class World:
    """A registry of package sources, a Kedi home to install into, and local packages."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.registry = tmp_path / "registry-sources"
        self.registry.mkdir()
        self.local = tmp_path / "local"
        self.installed = tmp_path / "home" / "registry"
        monkeypatch.setenv("KEDI_REGISTRY_MOCK_ROOT", str(self.registry))
        monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))

    def publish(self, name: str, version: str = "1.0.0", *needs: str, python: str = "") -> None:
        target = self.registry / name
        if target.exists():
            import shutil

            shutil.rmtree(target)
        package(self.registry, name, version, *needs, python=python)

    def write(self, name: str, version: str = "1.0.0", *needs: str, python: str = "") -> Path:
        return package(self.local / version, name, version, *needs, python=python)

    def record(self, name: str) -> dict:
        return json.loads((self.installed / name / INSTALL_RECEIPT_NAME).read_text())

    def version(self, name: str) -> str | None:
        return load_package_manifest(self.installed / name / "package.kedi").version

    def contents(self) -> list[str]:
        return sorted(path.name for path in self.installed.iterdir())


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def test_dependencies_install_first_once_each_and_are_recorded_as_dependencies(
    world: World,
) -> None:
    world.publish("shared", "1.2.0")
    world.publish("left", "1.0.0", "shared>=1")
    world.publish("right", "1.0.0", "shared")

    installed = install_package(world.write("app", "1.0.0", "left", "right>=1.0"))

    assert [dependency.name for dependency in installed.dependencies] == [
        "shared",
        "left",
        "right",
    ]
    assert world.contents() == [
        ".app.lock",
        ".left.lock",
        ".right.lock",
        ".shared.lock",
        "app",
        "left",
        "right",
        "shared",
    ]
    assert world.record("app")["lifecycle"]["installed_as"] == "explicit"
    assert world.record("shared")["lifecycle"]["installed_as"] == "dependency"
    assert world.record("shared")["install_source_kind"] == "registry"


def test_an_installed_dependency_that_satisfies_is_kept(world: World) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("first", "1.0.0", "shared"))
    receipt = (world.installed / "shared" / INSTALL_RECEIPT_NAME).stat().st_mtime_ns
    world.publish("shared", "2.0.0")

    installed = install_package(world.write("second", "1.0.0", "shared>=1"))

    assert installed.dependencies == ()
    assert (world.installed / "shared" / INSTALL_RECEIPT_NAME).stat().st_mtime_ns == receipt
    # Reached twice in one installation, a kept package is looked at once.
    world.publish("left", "1.0.0", "shared")
    installed = install_package(world.write("third", "1.0.0", "shared", "left"))
    assert [dependency.name for dependency in installed.dependencies] == ["left"]


def test_an_upgrade_must_satisfy_every_installed_dependent_or_nothing_changes(
    world: World,
) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("old", "1.0.0", "shared<2"))
    world.publish("shared", "2.0.0")

    with pytest.raises(ValueError, match="old needs shared<2") as error:
        install_package(world.write("new", "1.0.0", "shared>=2"))

    assert "new needs shared>=2" in str(error.value)
    assert world.version("shared") == "1.0.0"
    assert world.contents() == [".old.lock", ".shared.lock", "old", "shared"]


def test_a_conflict_names_each_package_once(world: World) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("old", "1.0.0", "shared<2"))
    world.publish("shared", "2.0.0")
    world.publish("old", "1.0.0", "shared<2")
    with pytest.raises(ValueError) as error:
        install_package(world.write("app", "1.0.0", "old", "shared>=2"))
    assert str(error.value) == (
        "No version of Kedi package 'shared' satisfies every package that needs it: "
        "old needs shared<2; app needs shared>=2 (the registry's active version is 2.0.0)"
    )


def test_an_upgrade_that_every_dependent_accepts_replaces_the_installed_version(
    world: World,
) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("old", "1.0.0", "shared"))
    world.publish("shared", "2.0.0")

    installed = install_package(world.write("new", "1.0.0", "shared>=2"))

    assert [dependency.name for dependency in installed.dependencies] == ["shared"]
    assert world.version("shared") == "2.0.0"


def test_a_reinstall_is_not_held_back_by_what_its_old_version_needed(world: World) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("app", "1.0.0", "shared<2"))
    world.publish("shared", "2.0.0")
    install_package(world.write("app", "2.0.0", "shared>=2"))
    assert world.version("shared") == "2.0.0"


def test_two_requirements_in_one_installation_that_no_version_meets_are_refused(
    world: World,
) -> None:
    world.publish("shared", "2.0.0")
    world.publish("left", "1.0.0", "shared>=2")
    world.publish("right", "1.0.0", "shared<2")
    with pytest.raises(ValueError, match="left needs shared>=2; right needs shared<2"):
        install_package(world.write("app", "1.0.0", "left", "right"))
    assert not world.installed.exists()


def test_a_registry_version_that_does_not_satisfy_is_refused(world: World) -> None:
    world.publish("shared", "1.0.0")
    with pytest.raises(
        ValueError, match=r"app needs shared>=2 \(the registry's active version is 1.0.0\)"
    ):
        install_package(world.write("app", "1.0.0", "shared>=2"))
    assert not world.installed.exists()


def test_a_cycle_is_refused_with_its_path(world: World) -> None:
    world.publish("left", "1.0.0", "right")
    world.publish("right", "1.0.0", "left")
    with pytest.raises(ValueError, match="in a cycle: left → right → left"):
        install_package(world.write("app", "1.0.0", "left"))
    world.publish("back", "1.0.0", "app")
    with pytest.raises(ValueError, match="in a cycle: app → back → app"):
        install_package(world.write("app", "1.0.1", "back"))
    assert not world.installed.exists()


def test_a_cycle_through_installed_packages_is_refused(world: World) -> None:
    world.publish("left", "1.0.0")
    install_package(world.write("right", "1.0.0", "left"))
    world.publish("left", "2.0.0", "right")
    with pytest.raises(ValueError, match="in a cycle: left → right → left"):
        install_package(world.write("app", "1.0.0", "left>=2"))


def test_a_reinstall_that_would_break_an_installed_dependent_is_refused(world: World) -> None:
    world.publish("app", "2.0.0")
    install_package(world.write("user", "1.0.0", "app>=2"))
    with pytest.raises(
        ValueError, match="Installing app 1.0.0 would break installed packages: user needs app>=2"
    ):
        install_package(world.write("app", "1.0.0"))
    assert world.version("app") == "2.0.0"


def test_a_failed_step_undoes_the_installation(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("old", "1.0.0", "shared"))
    world.publish("shared", "2.0.0")
    world.publish("fresh", "1.0.0", python="kedi-fresh-fixture")
    original = package_registry.install_python_dependencies

    def install(requirements: list[str]) -> None:
        if "kedi-fail-fixture" in requirements:
            raise RuntimeError("pip failed")
        original([])

    added = iter([{}, {"kedi-fresh-fixture": "1.0"}])
    monkeypatch.setattr(package_registry, "install_python_dependencies", install)
    monkeypatch.setattr(package_registry, "installed_distributions", lambda: next(added, {}))
    with pytest.raises(package_registry.PackageRegistryError) as error:
        install_package(
            world.write("app", "1.0.0", "shared>=2", "fresh", python="kedi-fail-fixture")
        )

    message = str(error.value)
    assert message.startswith("pip failed\n")
    assert "undone: shared, fresh" in message
    assert "Python distributions they added stay installed: kedi-fresh-fixture" in message
    assert world.version("shared") == "1.0.0"
    assert world.contents() == [".old.lock", ".shared.lock", "old", "shared"]


def test_an_undone_installation_without_python_additions_says_only_what_it_undid(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.publish("shared", "1.0.0")

    def fail(requirements: list[str]) -> None:
        if requirements:
            raise RuntimeError("pip failed")

    monkeypatch.setattr(package_registry, "install_python_dependencies", fail)
    monkeypatch.setattr(package_registry, "installed_distributions", dict)
    with pytest.raises(package_registry.PackageRegistryError) as error:
        install_package(world.write("app", "1.0.0", "shared", python="kedi-fail-fixture"))
    assert str(error.value) == (
        "pip failed\nThe packages this installation had installed were undone: shared."
    )
    assert world.contents() == []


def test_remove_refuses_a_package_that_is_needed(world: World) -> None:
    world.publish("shared", "1.0.0")
    install_package(world.write("app", "1.0.0", "shared>=1"))
    with pytest.raises(ValueError, match=r"needed by installed packages \(app needs shared>=1\)"):
        remove_package("shared")
    assert (world.installed / "shared").is_dir()


def test_remove_takes_the_dependencies_it_orphans_and_keeps_the_rest(world: World) -> None:
    world.publish("deep", "1.0.0")
    world.publish("shared", "1.0.0", "deep")
    world.publish("own", "1.0.0")
    world.publish("explicit", "1.0.0")
    install_package(world.write("app", "1.0.0", "shared", "own", "explicit"))
    install_package(world.write("other", "1.0.0", "shared"))
    add_package("explicit")  # now the user's own

    lines = remove_package("app", dry_run=True)
    assert (
        "own goes too: it was installed as a dependency, and no installed package needs it any more"
        in lines
    )
    assert (world.installed / "own").is_dir()

    remove_package("app")

    assert world.contents() == [
        ".deep.lock",
        ".explicit.lock",
        ".other.lock",
        ".shared.lock",
        "deep",
        "explicit",
        "other",
        "shared",
    ]
    remove_package("other")
    assert world.contents() == [".explicit.lock", "explicit"]


def test_an_explicit_install_stays_explicit_when_a_dependent_upgrades_it(world: World) -> None:
    world.publish("shared", "1.0.0")
    add_package("shared")
    world.publish("shared", "2.0.0")
    install_package(world.write("app", "1.0.0", "shared>=2"))
    assert world.record("shared")["lifecycle"]["installed_as"] == "explicit"
    remove_package("app")
    assert (world.installed / "shared").is_dir()


def test_audit_reports_missing_and_unsatisfied_dependencies(world: World) -> None:
    world.publish("shared", "1.0.0")
    world.publish("gone", "1.0.0")
    world.publish("fine", "1.0.0")
    install_package(world.write("app", "1.0.0", "shared>=1", "gone", "fine"))
    (world.installed / "gone").rename(world.installed / ".gone-moved")
    manifest = world.installed / "shared" / "package.kedi"
    manifest.write_text(manifest.read_text().replace("1.0.0", "0.5.0"))

    report = audit_installed_packages(registry_root=world.installed)

    assert report.dependency_problems == (
        "app needs shared>=1, but shared 0.5.0 is installed",
        "app needs gone, which is not installed",
    )
    assert report.exit_code == 1
    result = CliRunner().invoke(cli.main, ["audit"])
    assert "DEPENDENCY app needs gone, which is not installed" in result.output
    assert "dependency_problems=2" in result.output
    assert result.exit_code == 1


def test_the_cli_names_each_dependency_it_installed(world: World) -> None:
    world.publish("shared", "1.0.0")
    result = CliRunner().invoke(cli.main, ["install", str(world.write("app", "1.0.0", "shared"))])
    assert result.exit_code == 0, result.output
    assert f"Installed dependency shared at {world.installed / 'shared'}" in result.output


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("demo", "lists itself in kedi_dependencies"),
        ("hermes\n    hermes>=1", "'hermes' is listed more than once"),
        ("hermes @ https://example.org/hermes.zip", "only a package name and a version specifier"),
        ("hermes; python_version > '3'", "only a package name and a version specifier"),
        ("hermes[extra]", "only a package name and a version specifier"),
        ("Hermes-Agent", "lowercase letters, digits, and underscores"),
        ("hermes >>= 1", "invalid Kedi dependency"),
    ],
)
def test_kedi_dependencies_are_checked(tmp_path: Path, entry: str, message: str) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(f"> package: demo:\n  source: src\n  kedi_dependencies:\n    {entry}\n")
    with pytest.raises(ValueError, match=message):
        validate_package_metadata(load_package_manifest(manifest))


def test_one_kedi_dependencies_block() -> None:
    source = "> package: demo:\n  kedi_dependencies:\n    a\n  kedi_dependencies:\n    b\n"
    with pytest.raises(KediParseError, match="only one kedi_dependencies"):
        parse_program(source, source_path="/tmp/package.kedi")


def test_hover_lists_kedi_dependencies() -> None:
    from lsprotocol import types as lsp

    from kedi.lsp.features import hover

    source = "> package: demo:\n  kedi_dependencies:\n    hermes>=0.1\n"
    result = hover.compute_hover(source, "/tmp/package.kedi", lsp.Position(line=0, character=4))
    assert result is not None
    assert "Kedi dependencies: `hermes>=0.1`" in result.contents.value
