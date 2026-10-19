from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from kedi import package_dependencies
from kedi.package_registry import add_package, install_package


def package_source(root: Path, dependency: str = "example>=1") -> Path:
    source = root / "src" / "example"
    source.mkdir(parents=True)
    (source / "main.kedi").write_text("= installed\n", encoding="utf-8")
    manifest = root / "package.kedi"
    manifest.write_text(
        f"> package: example:\n  source: src/example\n  python_dependencies:\n    {dependency}\n",
        encoding="utf-8",
    )
    return manifest


def test_no_dependencies_does_not_start_an_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = Mock(side_effect=AssertionError("No installer lookup should be needed"))
    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", find_spec)
    package_dependencies.install_python_dependencies([])
    find_spec.assert_not_called()


@pytest.mark.parametrize("has_pip", [True, False])
def test_dependency_installer_targets_running_interpreter(
    monkeypatch: pytest.MonkeyPatch, has_pip: bool
) -> None:
    python = "/selected/venv/bin/python"
    monkeypatch.setattr(package_dependencies.sys, "executable", python)
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated/venv")
    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", lambda _: has_pip or None)
    monkeypatch.setattr(package_dependencies.shutil, "which", lambda _: "/tools/uv")
    run = Mock()
    invalidate = Mock()
    monkeypatch.setattr(package_dependencies.subprocess, "run", run)
    monkeypatch.setattr(package_dependencies.importlib, "invalidate_caches", invalidate)
    requirements = ["example[extra]>=1", 'other; python_version >= "3.11"']

    package_dependencies.install_python_dependencies(requirements)

    expected = (
        [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
        if has_pip
        else ["/tools/uv", "pip", "install", "--python", python]
    )
    run.assert_called_once_with([*expected, "--", *requirements], check=True)
    invalidate.assert_called_once_with()


def test_missing_dependency_installer_reports_target_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(package_dependencies.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="neither pip nor uv") as error:
        package_dependencies.install_python_dependencies(["example"])
    assert sys.executable in str(error.value)


@pytest.mark.parametrize(
    "error", [OSError("cannot start"), subprocess.CalledProcessError(1, "pip")]
)
def test_dependency_failure_preserves_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    manifest = package_source(tmp_path / "source")
    registry = tmp_path / "registry"
    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", lambda _: True)
    run = Mock()
    monkeypatch.setattr(package_dependencies.subprocess, "run", run)
    installed = install_package(manifest, registry_root=registry)
    previous = (installed.source / "main.kedi").read_bytes()
    (manifest.parent / "src/example/main.kedi").write_text("= replacement\n", encoding="utf-8")
    run.side_effect = error

    with pytest.raises(RuntimeError, match="was not replaced") as raised:
        install_package(manifest, registry_root=registry)

    assert raised.value.__cause__ is error
    assert (installed.source / "main.kedi").read_bytes() == previous
    assert not list(registry.glob(".example.staging-*"))


def test_invalid_source_is_rejected_before_dependency_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = package_source(tmp_path / "source")
    (manifest.parent / "src/example/main.kedi").unlink()
    run = Mock()
    monkeypatch.setattr(package_dependencies.subprocess, "run", run)
    with pytest.raises(FileNotFoundError, match="main.kedi"):
        install_package(manifest, registry_root=tmp_path / "registry")
    run.assert_not_called()


def test_named_package_installation_installs_python_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_source(tmp_path / "sources/example")
    monkeypatch.setenv("KEDI_REGISTRY_MOCK_ROOT", str(tmp_path / "sources"))
    monkeypatch.setattr(package_dependencies.importlib.util, "find_spec", lambda _: True)
    run = Mock()
    monkeypatch.setattr(package_dependencies.subprocess, "run", run)

    installed = add_package("example", registry_root=tmp_path / "registry")

    assert installed.name == "example"
    assert run.call_args.args[0][-2:] == ["--", "example>=1"]


def test_cli_installs_dependency_in_each_invoking_venv(tmp_path: Path) -> None:
    wheel = tmp_path / "kedi_dependency_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kedi_dependency_fixture.py", "VALUE = 'installed in selected Python'\n")
        archive.writestr(
            "kedi_dependency_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: kedi-dependency-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "kedi_dependency_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("kedi_dependency_fixture-1.0.dist-info/RECORD", "")
    manifest = package_source(tmp_path / "source", f"kedi-dependency-fixture @ {wheel.as_uri()}")
    environment = dict(os.environ)
    environment.update(
        KEDI_HOME=str(tmp_path / "home"),
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parents[1] / "src"), *(path for path in sys.path if path)]
        ),
        PIP_NO_INDEX="1",
        UV_NO_INDEX="true",
    )
    for name in ("first", "second"):
        root = tmp_path / name
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(root)
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # A conflicting activated environment must not override the selected executable.
        environment["VIRTUAL_ENV"] = str(tmp_path / "unrelated")
        result = subprocess.run(
            [str(python), "-m", "kedi", "install", str(manifest)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        result = subprocess.run(
            [str(python), "-c", "import kedi_dependency_fixture as f; print(f.__file__)"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()).is_relative_to(root)
