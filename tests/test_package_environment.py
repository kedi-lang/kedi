"""``environment: uv``: a package adapter's command runs in an environment of the package's own,
inside the installed package, and goes with it."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from kedi import package_environment, package_registry
from kedi.package_adapters import package_adapters
from kedi.package_dependencies import find_uv
from kedi.package_environment import ENVIRONMENT_DIRECTORY
from kedi.package_manifest import load_package_manifest, validate_package_metadata
from kedi.package_registry import INSTALL_RECEIPT_NAME, install_package, remove_package

UV = find_uv()
needs_uv = pytest.mark.skipif(UV is None, reason="uv is not installed")


def bridge_package(root: Path, *, version: str = "0.1.0", project: bool = True) -> Path:
    source = root / "src" / "bridged"
    (source / "wheels").mkdir(parents=True)
    (source / "main.kedi").write_text("= installed\n")
    (source / "bridge.py").write_text("import kedi_env_fixture\nprint(kedi_env_fixture.VALUE)\n")
    wheel = source / "wheels" / "kedi_env_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kedi_env_fixture.py", "VALUE = 'from the package environment'\n")
        archive.writestr(
            "kedi_env_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: kedi-env-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "kedi_env_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("kedi_env_fixture-1.0.dist-info/RECORD", "")
    if project:
        (source / "pyproject.toml").write_text(
            '[project]\nname = "bridged-env"\nversion = "0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["kedi-env-fixture"]\n\n'
            "[tool.uv]\npackage = false\n\n"
            '[tool.uv.sources]\nkedi-env-fixture = { path = "wheels/kedi_env_fixture-1.0-py3-none-any.whl" }\n'
        )
        if UV is not None:
            subprocess.run(
                [UV, "lock", "--offline", "--project", str(source)], check=True, capture_output=True
            )
        else:
            (source / "uv.lock").write_text("")
    manifest = root / "package.kedi"
    manifest.write_text(
        f"> package: bridged:\n  version: {version}\n  source: src/bridged\n  adapters:\n    bridged:\n"
        "      kind: agent\n      command: python bridge.py\n      environment: uv\n"
    )
    return manifest


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")


@needs_uv
@pytest.mark.usefixtures("offline")
def test_the_environment_lives_in_the_package_runs_its_command_and_goes_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    registry = tmp_path / "home" / "registry"

    installed = install_package(bridge_package(tmp_path / "source"), registry_root=registry)

    environment = installed.root / ENVIRONMENT_DIRECTORY
    assert (environment / "venv").is_dir()
    assert not (environment / "cache").exists()
    record = json.loads((installed.root / INSTALL_RECEIPT_NAME).read_text())["lifecycle"]
    assert [entry["kind"] for entry in record["created"]] == [
        "package_directory",
        "runtime_directory",
        "environment",
    ]
    (adapter,) = package_adapters("bridged")
    assert adapter.command is not None
    python, script = adapter.command
    assert Path(python).is_relative_to(environment / "venv")
    assert script == str(installed.source / "bridge.py")
    ran = subprocess.run([python, script], capture_output=True, text=True, check=True)
    assert ran.stdout.strip() == "from the package environment"
    # Nothing reached the interpreter running Kedi.
    with pytest.raises(ImportError):
        __import__("kedi_env_fixture")

    remove_package("bridged", registry_root=registry)
    assert list(registry.iterdir()) == []


def test_a_failed_environment_brings_the_previous_version_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    monkeypatch.setattr(package_environment, "find_uv", lambda: "/bin/uv")
    monkeypatch.setattr(
        package_registry,
        "install_environment",
        lambda uv, root, source: (root / ENVIRONMENT_DIRECTORY).mkdir(),
    )
    install_package(bridge_package(tmp_path / "v1"), registry_root=registry)

    def fail(uv: str, root: Path, source: Path) -> None:
        (root / ENVIRONMENT_DIRECTORY).mkdir()
        raise RuntimeError("uv failed")

    monkeypatch.setattr(package_registry, "install_environment", fail)
    with pytest.raises(RuntimeError, match="uv failed"):
        install_package(bridge_package(tmp_path / "v2", version="0.2.0"), registry_root=registry)

    assert "version: 0.1.0" in (registry / "bridged" / "package.kedi").read_text()
    assert sorted(path.name for path in registry.iterdir()) == [".bridged.lock", "bridged"]


def test_the_environment_needs_its_project_and_uv_before_anything_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    with pytest.raises(ValueError, match="has no pyproject.toml or uv.lock"):
        install_package(_without_project(tmp_path), registry_root=registry)
    monkeypatch.setattr(package_environment, "find_uv", lambda: None)
    with pytest.raises(ValueError, match="Run `kedi python install uv`"):
        install_package(bridge_package(tmp_path / "source"), registry_root=registry)
    assert not registry.exists() or list(registry.iterdir()) == []


def _without_project(tmp_path: Path) -> Path:
    manifest = bridge_package(tmp_path / "bare", project=False)
    (manifest.parent / "src/bridged/uv.lock").unlink(missing_ok=True)
    return manifest


def test_installing_the_environment_runs_uv_sync_into_the_package_and_drops_its_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], *, env: dict[str, str], check: bool) -> None:
        calls.append((command, env))
        Path(env["UV_CACHE_DIR"]).mkdir(parents=True)

    monkeypatch.setattr(package_environment.subprocess, "run", run)
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated")
    package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path / "pkg/src")
    command, env = calls[0]
    assert command == [
        "/bin/uv",
        "sync",
        "--frozen",
        "--no-dev",
        "--no-install-project",
        "--project",
        str(tmp_path / "pkg/src"),
    ]
    root = tmp_path / "pkg" / ENVIRONMENT_DIRECTORY
    assert env["UV_PROJECT_ENVIRONMENT"] == str(root / "venv")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(root / "python")
    assert env["UV_PYTHON_BIN_DIR"] == str(root / "python-bin")
    assert "VIRTUAL_ENV" not in env
    assert not (root / "cache").exists()

    def fail(command: list[str], *, env: dict[str, str], check: bool) -> None:
        Path(env["UV_CACHE_DIR"]).mkdir(parents=True)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(package_environment.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Installing the environment of pkg with uv failed"):
        package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path / "pkg/src")
    assert not (root / "cache").exists()


def test_a_program_the_environment_lacks_runs_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert package_environment.environment_program(tmp_path, "node") is None
    scripts = tmp_path / ENVIRONMENT_DIRECTORY / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("")
    monkeypatch.setattr(package_environment.sys, "platform", "win32")
    assert package_environment.environment_program(tmp_path, "python") == str(
        scripts / "python.exe"
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("command: python bridge.py\n      environment: conda", "environment must be uv"),
        (
            "class: kedi_pi:PiAdapter\n      environment: uv",
            "a class runs in Kedi's own interpreter",
        ),
    ],
)
def test_the_environment_field_is_checked(tmp_path: Path, fields: str, message: str) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        f"> package: bridged:\n  source: src\n  adapters:\n    bridged:\n      kind: agent\n      {fields}\n"
    )
    with pytest.raises(ValueError, match=message):
        validate_package_metadata(load_package_manifest(manifest))


def test_a_command_whose_program_the_environment_lacks_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(package_environment, "find_uv", lambda: "/bin/uv")
    monkeypatch.setattr(package_registry, "install_environment", lambda uv, root, source: None)
    manifest = bridge_package(tmp_path / "source")
    manifest.write_text(
        manifest.read_text().replace("command: python bridge.py", "command: node bridge.py")
    )
    install_package(manifest, registry_root=tmp_path / "home" / "registry")
    (adapter,) = package_adapters("bridged")
    assert adapter.command is not None and adapter.command[0] == "node"
