"""``kedi remove`` and the install record: a removed package leaves nothing behind.

The end-to-end tests run Kedi in throwaway virtual environments, with local wheels, so the
Python distributions a package adds are real installs that real uninstalls remove.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from kedi import package_dependencies, package_lifecycle
from kedi.package_lifecycle import RUNTIME_DIRECTORY, lifecycle_record
from kedi.package_registry import INSTALL_RECEIPT_NAME, install_package, remove_package

SRC = Path(__file__).resolve().parents[1] / "src"


def wheel(directory: Path, name: str, requires: tuple[str, ...] = ()) -> Path:
    module = name.replace("-", "_")
    path = directory / f"{module}-1.0-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n" + "".join(
        f"Requires-Dist: {r}\n" for r in requires
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{module}.py", "VALUE = 1\n")
        archive.writestr(f"{module}-1.0.dist-info/METADATA", metadata)
        archive.writestr(
            f"{module}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{module}-1.0.dist-info/RECORD", "")
    return path


def package(root: Path, name: str, *requirements: str) -> Path:
    source = root / "src" / name
    source.mkdir(parents=True)
    (source / "main.kedi").write_text("= installed\n", encoding="utf-8")
    manifest = root / "package.kedi"
    body = f"> package: {name}:\n  source: src/{name}\n"
    if requirements:
        body += "  python_dependencies:\n" + "".join(f"    {r}\n" for r in requirements)
    manifest.write_text(body, encoding="utf-8")
    return manifest


class Environment:
    """A virtual environment with pip, running this checkout's Kedi."""

    def __init__(self, root: Path) -> None:
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(root / "venv")
        self.python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.env = {
            **os.environ,
            "KEDI_HOME": str(root / "home"),
            "PYTHONPATH": os.pathsep.join([str(SRC), *(path for path in sys.path if path)]),
            "PIP_NO_INDEX": "1",
            "UV_NO_INDEX": "true",
        }
        self.env.pop("VIRTUAL_ENV", None)
        self.registry = root / "home" / "registry"

    def kedi(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.python), "-m", "kedi", *arguments],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def pip(self, *arguments: str) -> None:
        subprocess.run(
            [
                str(self.python),
                "-m",
                "pip",
                *arguments,
                "--disable-pip-version-check",
                "--no-index",
            ],
            env=self.env,
            check=True,
            capture_output=True,
            timeout=120,
        )

    def has(self, module: str) -> bool:
        probe = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
        return (
            subprocess.run([str(self.python), "-c", probe], env=self.env, timeout=30).returncode
            == 0
        )

    def record(self, name: str) -> dict:
        return json.loads((self.registry / name / INSTALL_RECEIPT_NAME).read_text())["lifecycle"]


@pytest.fixture
def environment(tmp_path: Path) -> Environment:
    return Environment(tmp_path)


def recorded(record: dict) -> list[str]:
    return [entry["name"] for entry in record["created"] if entry["kind"] == "python_distribution"]


def test_a_reinstall_keeps_the_record_and_the_flag_removes_what_the_package_added(
    tmp_path: Path, environment: Environment
) -> None:
    fixture = wheel(tmp_path, "kedi-lifecycle-a")
    manifest = package(tmp_path / "source", "example", f"kedi-lifecycle-a @ {fixture.as_uri()}")

    assert environment.kedi("install", str(manifest)).returncode == 0
    assert recorded(environment.record("example")) == ["kedi-lifecycle-a"]
    # A reinstall finds the distribution already there; the record must not lose it.
    assert environment.kedi("install", str(manifest)).returncode == 0
    assert recorded(environment.record("example")) == ["kedi-lifecycle-a"]

    removed = environment.kedi("remove", "example", "--with-python-dependencies")

    assert removed.returncode == 0, removed.stderr
    assert "Uninstalled Python distribution kedi-lifecycle-a 1.0" in removed.stdout
    assert not environment.has("kedi_lifecycle_a")
    assert sorted(path.name for path in environment.registry.iterdir()) == []


def test_without_the_flag_python_is_not_touched(tmp_path: Path, environment: Environment) -> None:
    fixture = wheel(tmp_path, "kedi-lifecycle-b")
    manifest = package(tmp_path / "source", "example", f"kedi-lifecycle-b @ {fixture.as_uri()}")
    assert environment.kedi("install", str(manifest)).returncode == 0

    removed = environment.kedi("remove", "example")

    assert removed.returncode == 0, removed.stderr
    assert not (environment.registry / "example").exists()
    assert environment.has("kedi_lifecycle_b")
    assert (
        "Python distributions these packages added stay installed: kedi-lifecycle-b"
        in removed.stdout
    )
    assert "kedi remove example --with-python-dependencies" in removed.stdout


def test_a_distribution_something_else_needs_stays_even_with_the_flag(
    tmp_path: Path, environment: Environment
) -> None:
    needed = wheel(tmp_path, "kedi-lifecycle-needed")
    user = wheel(tmp_path, "kedi-lifecycle-user", requires=("kedi-lifecycle-needed",))
    shared = wheel(tmp_path, "kedi-lifecycle-shared")
    first = package(
        tmp_path / "first",
        "first",
        f"kedi-lifecycle-needed @ {needed.as_uri()}",
        f"kedi-lifecycle-shared @ {shared.as_uri()}",
    )
    second = package(tmp_path / "second", "second", f"kedi-lifecycle-shared @ {shared.as_uri()}")

    assert environment.kedi("install", str(first)).returncode == 0
    assert environment.kedi("install", str(second)).returncode == 0
    # second found kedi-lifecycle-shared already installed: it records nothing, but needs it.
    assert recorded(environment.record("second")) == []
    environment.pip("install", str(user))  # the user's own, which requires kedi-lifecycle-needed

    removed = environment.kedi("remove", "first", "--with-python-dependencies")

    assert removed.returncode == 0, removed.stderr
    assert (
        "Kept Python distribution kedi-lifecycle-needed: installed distribution kedi-lifecycle-user requires it"
        in removed.stdout
    )
    assert (
        "Kept Python distribution kedi-lifecycle-shared: installed package second needs it"
        in removed.stdout
    )
    assert environment.has("kedi_lifecycle_needed") and environment.has("kedi_lifecycle_shared")


def test_dry_run_removes_nothing_and_says_what_it_would(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    install_package(package(tmp_path / "source", "example"), registry_root=registry)
    lines = remove_package("example", registry_root=registry, dry_run=True)
    assert lines[0] == f"Would remove {registry / 'example'}"
    assert (registry / "example" / "src/example/main.kedi").is_file()


def test_remove_takes_an_interrupted_installs_leftovers_and_the_lock_file(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    install_package(package(tmp_path / "source", "example"), registry_root=registry)
    (registry / ".example.staging-1234").mkdir()
    (registry / ".example.backup-5678").mkdir()
    assert (registry / ".example.lock").exists()
    install_package(package(tmp_path / "other", "other"), registry_root=registry)

    remove_package("example", registry_root=registry)

    assert sorted(path.name for path in registry.iterdir()) == [".other.lock", "other"]


def test_an_install_records_what_it_created_and_when_each_goes(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    install_package(package(tmp_path / "source", "example"), registry_root=registry)
    receipt = json.loads((registry / "example" / INSTALL_RECEIPT_NAME).read_text())
    assert receipt["format_version"] == 1  # provenance format unchanged: audit reads it as before
    assert receipt["lifecycle"] == {
        "installed_as": "explicit",
        "created": [
            {"kind": "package_directory", "path": ".", "removed": "with the package"},
            {"kind": "runtime_directory", "path": RUNTIME_DIRECTORY, "removed": "with the package"},
        ],
        "changed": [],
    }


def test_a_package_installed_before_the_record_is_removed_with_a_note(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    install_package(package(tmp_path / "source", "example"), registry_root=registry)
    receipt_path = registry / "example" / INSTALL_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    del receipt["lifecycle"]
    receipt_path.write_text(json.dumps(receipt))
    lines = remove_package("example", registry_root=registry)
    assert any("are not known and stay in place" in line for line in lines)
    assert not (registry / "example").exists()


def test_removing_a_package_that_is_not_installed_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not installed"):
        remove_package("missing", registry_root=tmp_path / "registry")


def test_an_upgrade_of_a_present_distribution_is_recorded_and_never_reverted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = lifecycle_record(
        previous=None,
        before={"shared": "1.0", "kept": "2.0"},
        after={"shared": "1.1", "kept": "2.0", "new": "3.0"},
    )
    assert [
        (e["name"], e["version"]) for e in record["created"] if e["kind"] == "python_distribution"
    ] == [("new", "3.0")]
    assert record["changed"][0] | {"target": None} == {
        "kind": "python_distribution",
        "name": "shared",
        "from": "1.0",
        "to": "1.1",
        "target": None,
        "removed": package_lifecycle.NEVER_REVERTED,
    }


def test_package_bridges_keep_their_runtime_files_inside_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.registry import package_runtime_environment

    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    install_package(
        package(tmp_path / "source", "example"), registry_root=tmp_path / "home" / "registry"
    )
    environment = package_runtime_environment("example")
    runtime = tmp_path / "home" / "registry" / "example" / RUNTIME_DIRECTORY
    assert environment == {
        "KEDI_PACKAGE_RUNTIME_DIR": str(runtime),
        "KEDI_BRIDGE_CACHE": str(runtime / "bridge-checks"),
    }
    assert runtime.is_dir()


def test_uv_is_found_beside_the_interpreter_when_it_is_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "bin"
    scripts.mkdir()
    (scripts / "uv").write_text("")
    monkeypatch.setattr(package_dependencies.shutil, "which", lambda _: None)
    monkeypatch.setattr(package_dependencies.sysconfig, "get_path", lambda _: str(scripts))
    monkeypatch.setattr(package_dependencies.sys, "platform", "linux")
    assert package_dependencies.find_uv() == str(scripts / "uv")


def test_remove_rejects_unknown_options() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kedi", "remove", "example", "--force"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode != 0
    assert "Unexpected option for 'kedi remove': --force" in result.stderr


# In-process tests of the removal plan, with the interpreter's distributions faked.


class FakeMetadata(dict):
    def get_all(self, key: str) -> list[str] | None:
        value = self.get(key)
        return value if isinstance(value, list) else None


class FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str = "1.0",
        requires: tuple[str, ...] = (),
        extras: tuple[str, ...] = (),
    ) -> None:
        self.metadata = FakeMetadata({"Name": name, "Provides-Extra": list(extras)})
        self.version = version
        self.requires = list(requires)


def fake_interpreter(monkeypatch: pytest.MonkeyPatch, *distributions: FakeDistribution) -> None:
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "distributions", lambda: list(distributions))


def installed(
    registry: Path, name: str, lifecycle: dict | None, manifest: str | None = None
) -> None:
    root = registry / name
    root.mkdir(parents=True)
    receipt = {"package": name} if lifecycle is None else {"package": name, "lifecycle": lifecycle}
    (root / INSTALL_RECEIPT_NAME).write_text(json.dumps(receipt))
    (root / "package.kedi").write_text(manifest or f"> package: {name}:\n  source: src/{name}\n")


def python_entry(name: str, version: str = "1.0", target: str | None = None) -> dict:
    return {
        "kind": "python_distribution",
        "name": name,
        "version": version,
        "target": target or sys.executable,
    }


def test_the_plan_keeps_each_distribution_something_else_needs_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    created = [
        python_entry(name)
        for name in (
            "alone",
            "lost",
            "moved",
            "shared",
            "wanted",
            "chained",
            "via-alone",
            "kept-by-manifest",
        )
    ]
    created += [
        python_entry("elsewhere", target="/other/python"),
        {"kind": "package_directory", "path": "."},
    ]
    changed = [{"kind": "python_distribution", "name": "upgraded", "from": "1.0", "to": "2.0"}]
    installed(registry, "example", {"created": created, "changed": changed})
    installed(registry, "other", {"created": [python_entry("shared")]})
    installed(
        registry,
        "third",
        {"created": []},
        "> package: third:\n  source: src/third\n  python_dependencies:\n    kept-by-manifest>=1\n",
    )
    installed(registry, "broken", None, "not a manifest")
    (registry / ".hidden").mkdir()
    fake_interpreter(
        monkeypatch,
        FakeDistribution("alone", requires=("via-alone",)),
        FakeDistribution("moved", "2.0"),
        FakeDistribution("shared"),
        FakeDistribution("wanted", requires=("chained; extra == 'full'",), extras=("full",)),
        FakeDistribution("chained"),
        FakeDistribution("via-alone"),
        FakeDistribution("kept-by-manifest"),
        FakeDistribution(
            "user",
            requires=(
                "wanted; extra == 'cli'",
                "not a requirement !!",
                "absent; python_version < '2'",
            ),
            extras=("cli",),
        ),
        FakeDistribution(""),
    )

    plan = package_lifecycle.plan_removal(
        "example", registry_root=registry, with_python_dependencies=True
    )

    assert plan.uninstall == [("alone", "1.0"), ("via-alone", "1.0")]
    assert dict(plan.kept) == {
        "elsewhere": "it was installed into another interpreter, /other/python",
        "moved": "its version changed since this package installed it (1.0 → 2.0)",
        "shared": "installed package other needs it",
        "kept-by-manifest": "installed package third needs it",
        "wanted": "installed distribution user requires it",
        "chained": "installed distribution wanted requires it",
    }
    assert "Python distribution lost is no longer installed" in plan.notes
    assert any("upgraded was upgraded from 1.0 to 2.0 by example" in note for note in plan.notes)

    uninstalled: list[list[str]] = []
    monkeypatch.setattr(package_lifecycle, "uninstall_python_distributions", uninstalled.append)
    # A path the plan listed may be gone by the time the lock is held (a concurrent install
    # committed its staging directory).
    plan.packages[0].paths.append(registry / ".example.staging-gone")
    package_lifecycle.remove(plan, registry_root=registry)
    assert uninstalled == [["alone", "via-alone"]]
    assert not (registry / "example").exists() and (registry / "other").exists()


def test_without_the_flag_the_plan_names_what_stays(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    installed(registry, "example", {"created": [python_entry("alone")]})
    plan = package_lifecycle.plan_removal(
        "example", registry_root=registry, with_python_dependencies=False
    )
    assert plan.uninstall == [] and plan.packages[0].lock is None
    assert plan.notes == [
        "Python distributions these packages added stay installed: alone. "
        "Run kedi remove example --with-python-dependencies to remove them as well"
    ]


def test_a_reinstall_with_no_python_dependencies_left_keeps_what_is_still_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_interpreter(monkeypatch, FakeDistribution("still"))
    previous = {
        "created": [python_entry("still"), python_entry("gone")],
        "changed": [{"name": "kept"}],
    }
    record = lifecycle_record(previous=previous, before=None, after=None)
    assert [e["name"] for e in record["created"] if e["kind"] == "python_distribution"] == ["still"]
    assert record["changed"] == [{"name": "kept"}]


def test_an_unreadable_receipt_has_no_record(tmp_path: Path) -> None:
    assert package_lifecycle.read_lifecycle(tmp_path) is None
    (tmp_path / INSTALL_RECEIPT_NAME).write_text("[]")
    assert package_lifecycle.read_lifecycle(tmp_path) is None


def test_the_cli_removes_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from kedi import cli

    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    install_package(
        package(tmp_path / "source", "example"), registry_root=tmp_path / "home" / "registry"
    )
    runner = CliRunner()
    assert "--with-python-dependencies" in runner.invoke(cli.main, ["remove", "--help"]).output
    dry = runner.invoke(cli.main, ["remove", "example", "--dry-run"])
    assert (
        dry.exit_code == 0 and "Would remove" in dry.output and "Removed example" not in dry.output
    )
    done = runner.invoke(cli.main, ["remove", "example"])
    assert done.exit_code == 0 and "Removed example" in done.output
    missing = runner.invoke(cli.main, ["remove", "example"])
    assert missing.exit_code != 0 and "is not installed" in missing.output
    assert runner.invoke(cli.main, ["remove"]).exit_code != 0
    assert "--force" in runner.invoke(cli.main, ["remove", "x", "--force"]).output


def test_the_cli_installs_into_kedis_python(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from kedi import cli

    calls: list[list[str]] = []
    monkeypatch.setattr(
        package_dependencies,
        "install_python_requirements",
        lambda requirements: calls.append(list(requirements)),
    )
    runner = CliRunner()
    assert "kedi python install uv" in runner.invoke(cli.main, ["python", "--help"]).output
    assert runner.invoke(cli.main, ["python", "install", "uv"]).exit_code == 0
    assert calls == [["uv"]]
    assert runner.invoke(cli.main, ["python"]).exit_code != 0

    def fail(_requirements: list[str]) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(package_dependencies, "install_python_requirements", fail)
    failed = runner.invoke(cli.main, ["python", "install", "uv"])
    assert failed.exit_code != 0 and "no network" in failed.output


def test_python_installs_and_uninstalls_go_through_the_interpreters_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", lambda _: True)
    run = Mock()
    monkeypatch.setattr(package_dependencies.subprocess, "run", run)
    package_dependencies.install_python_requirements(["uv"])
    package_dependencies.uninstall_python_distributions(["alone"])
    package_dependencies.uninstall_python_distributions([])
    assert run.call_args_list[0].args[0][-2:] == ["--", "uv"]
    assert (
        run.call_args_list[1].args[0][-3:]
        == ["uninstall", "--disable-pip-version-check", "--yes", "alone"][-3:]
    )
    assert len(run.call_args_list) == 2
    run.side_effect = subprocess.CalledProcessError(1, "pip")
    with pytest.raises(RuntimeError, match="Installing uv into"):
        package_dependencies.install_python_requirements(["uv"])
    with pytest.raises(RuntimeError, match="Uninstalling alone from"):
        package_dependencies.uninstall_python_distributions(["alone"])


def test_uv_comes_from_the_uv_package_when_nothing_else_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    monkeypatch.setattr(package_dependencies.shutil, "which", lambda _: None)
    monkeypatch.setattr(package_dependencies.sysconfig, "get_path", lambda _: str(tmp_path))
    module = types.SimpleNamespace(find_uv_bin=lambda: "/from/the/uv/package")
    monkeypatch.setattr(package_dependencies.importlib, "import_module", lambda _: module)
    assert package_dependencies.find_uv() == "/from/the/uv/package"

    def missing() -> str:
        raise FileNotFoundError

    module.find_uv_bin = missing
    assert package_dependencies.find_uv() is None
