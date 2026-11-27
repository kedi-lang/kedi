"""``environment: uv``: a package adapter's command runs in an environment of the package's own,
inside the installed package, and goes with it."""

from __future__ import annotations

import json
import re
import subprocess
import sys
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
    # Offline, uv cannot download the managed Python a package environment uses by default;
    # an explicit interpreter, this one, is used whatever the preference.
    monkeypatch.setenv("UV_PYTHON", sys.executable)


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
    assert (environment / "ready").is_file()
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
        lambda uv, root, source, project=None: (root / ENVIRONMENT_DIRECTORY).mkdir(),
    )
    install_package(bridge_package(tmp_path / "v1"), registry_root=registry)

    def fail(uv: str, root: Path, source: Path, project: object = None) -> None:
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
    monkeypatch.delenv("UV_PYTHON_PREFERENCE", raising=False)
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
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert "VIRTUAL_ENV" not in env
    assert not (root / "cache").exists()
    assert package_environment.environment_ready(tmp_path / "pkg")
    monkeypatch.setenv("UV_PYTHON_PREFERENCE", "system")
    package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path / "pkg/src")
    assert calls[1][1]["UV_PYTHON_PREFERENCE"] == "system"

    def fail(command: list[str], *, env: dict[str, str], check: bool) -> None:
        Path(env["UV_CACHE_DIR"]).mkdir(parents=True)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(package_environment.subprocess, "run", fail)
    (root / "ready").unlink()
    with pytest.raises(RuntimeError, match="Installing the environment of pkg with uv failed"):
        package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path / "pkg/src")
    assert not (root / "cache").exists()
    assert not package_environment.environment_ready(tmp_path / "pkg")


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
    monkeypatch.setattr(
        package_registry, "install_environment", lambda uv, root, source, project=None: None
    )
    manifest = bridge_package(tmp_path / "source")
    manifest.write_text(
        manifest.read_text().replace("command: python bridge.py", "command: node bridge.py")
    )
    install_package(manifest, registry_root=tmp_path / "home" / "registry")
    (adapter,) = package_adapters("bridged")
    assert adapter.command is not None and adapter.command[0] == "node"


def ready(uv: str, root: Path, source: Path, project: object = None) -> None:
    (root / ENVIRONMENT_DIRECTORY).mkdir()
    (root / ENVIRONMENT_DIRECTORY / "ready").write_text("")


def test_an_interrupted_environment_brings_the_previous_version_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    monkeypatch.setattr(package_environment, "find_uv", lambda: "/bin/uv")
    monkeypatch.setattr(package_registry, "install_environment", ready)
    install_package(bridge_package(tmp_path / "v1"), registry_root=registry)

    def interrupted(uv: str, root: Path, source: Path, project: object = None) -> None:
        (root / ENVIRONMENT_DIRECTORY).mkdir()
        raise KeyboardInterrupt

    monkeypatch.setattr(package_registry, "install_environment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        install_package(bridge_package(tmp_path / "v2", version="0.2.0"), registry_root=registry)

    assert "version: 0.1.0" in (registry / "bridged" / "package.kedi").read_text()
    assert sorted(path.name for path in registry.iterdir()) == [".bridged.lock", "bridged"]


def test_a_half_installed_environment_is_refused_and_a_finished_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.adapters import jsonrpc
    from kedi.agent_adapter.registry import build_package_adapter
    from kedi.errors import KediExecutionError

    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(package_environment, "find_uv", lambda: "/bin/uv")
    monkeypatch.setattr(
        package_registry, "install_environment", lambda uv, root, source, project=None: None
    )
    installed = install_package(
        bridge_package(tmp_path / "source"), registry_root=tmp_path / "home" / "registry"
    )
    (adapter,) = package_adapters("bridged")
    with pytest.raises(KediExecutionError, match="whose installation did not finish") as error:
        build_package_adapter(adapter, None)
    assert "kedi add bridged" in str(error.value)

    class Bridge:
        kind = "agent-harness"

        def __init__(self, command: list[str], **_: object) -> None:
            self.command = command

    monkeypatch.setattr(jsonrpc, "JsonRpcAdapter", Bridge)
    monkeypatch.setattr("kedi.agent_adapter.registry.shutil.which", lambda program: program)
    ready("/bin/uv", installed.root, installed.source)
    built = build_package_adapter(adapter, None)
    assert isinstance(built, Bridge) and built.command[0] == "python"


# environment_project: the environment comes from a repository at an exact commit.

COMMIT = "8fc0374447313e276f6b9a2764bfc750190c966a"


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def project_repository(root: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    """A local repository holding ``files`` at its one commit."""
    root.mkdir(parents=True)
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "config", "user.email", "environment-tests@kedi-lang.org")
    git(root, "config", "user.name", "Kedi Environment Tests")
    git(root, "config", "uploadpack.allowReachableSHA1InWant", "true")
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "project")
    return root, git(root, "rev-parse", "HEAD")


def uv_only(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Records uv's commands instead of running them; git still runs."""
    real = subprocess.run
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        if command[0] == "git":
            return real(command, **kwargs)  # type: ignore[call-overload]
        calls.append(command)
        return None

    monkeypatch.setattr(package_environment.subprocess, "run", run)
    return calls


def test_an_environment_project_is_a_github_repository_at_a_full_commit() -> None:
    project = package_environment.parse_environment_project(
        f" git+https://github.com/NousResearch/hermes-agent@{COMMIT} "
    )
    assert project == package_environment.ProjectSource(
        "https://github.com/NousResearch/hermes-agent", COMMIT
    )
    for value, message in [
        ("git+https://github.com/o/r", "must end with @ and a full 40-character commit hash"),
        ("git+https://github.com/o/r@v0.23.4", "a tag or branch can move"),
        (f"git+https://github.com/o/r@{COMMIT[:12]}", "full 40-character commit hash"),
        (f"git+https://github.com/o/r@{COMMIT.upper()}", "full 40-character commit hash"),
        (f"https://github.com/o/r@{COMMIT}", "must start with 'git+'"),
        (f"git+https://gitlab.com/o/r@{COMMIT}", "credential-free https://github.com URL"),
        (f"git+http://github.com/o/r@{COMMIT}", "credential-free https://github.com URL"),
        (f"git+https://github.com/o/r?x=1@{COMMIT}", "without query parameters"),
    ]:
        with pytest.raises(ValueError, match=re.escape(message)):
            package_environment.parse_environment_project(value)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            f"command: python bridge.py\n      environment_project: git+https://github.com/o/r@{COMMIT}",
            "environment_project names the project of an environment: uv",
        ),
        (
            "command: python bridge.py\n      environment: uv\n      environment_project: git+https://github.com/o/r@main",
            "package adapter 'bridged': environment_project 'git\\+https://github.com/o/r@main' must end",
        ),
    ],
)
def test_the_environment_project_field_is_checked(
    tmp_path: Path, fields: str, message: str
) -> None:
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        f"> package: bridged:\n  source: src\n  adapters:\n    bridged:\n      kind: agent\n      {fields}\n"
    )
    with pytest.raises(ValueError, match=message):
        validate_package_metadata(load_package_manifest(manifest))


def test_adapters_sharing_an_environment_name_one_project(tmp_path: Path) -> None:
    manifest = tmp_path / "package.kedi"
    adapter = "      kind: agent\n      command: python bridge.py\n      environment: uv\n"
    manifest.write_text(
        "> package: bridged:\n  source: src\n  adapters:\n"
        f"    first:\n{adapter}      environment_project: git+https://github.com/o/r@{COMMIT}\n"
        f"    second:\n{adapter}"
    )
    with pytest.raises(ValueError, match="name the same environment_project, or none"):
        validate_package_metadata(load_package_manifest(manifest))


def test_a_project_commit_is_checked_out_without_its_history_and_synced_with_its_own_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, commit = project_repository(
        tmp_path / "upstream", {"pyproject.toml": b"[project]\n", "uv.lock": b"", "tools/run": b"x"}
    )
    calls = uv_only(monkeypatch)
    project = package_environment.ProjectSource(repository.as_uri(), commit)

    package_environment.install_environment(
        "/bin/uv", tmp_path / "pkg", tmp_path / "unused", project
    )

    root = tmp_path / "pkg" / ENVIRONMENT_DIRECTORY
    assert calls == [
        ["/bin/uv", "sync", "--frozen", "--no-dev", "--project", str(root / "project")]
    ]
    assert (root / "project" / "pyproject.toml").read_text() == "[project]\n"
    assert (root / "project" / "tools" / "run").read_bytes() == b"x"
    assert not (root / "project" / ".git").exists()
    assert package_environment.environment_ready(tmp_path / "pkg")


def test_a_commit_the_repository_does_not_have_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _ = project_repository(tmp_path / "upstream", {"pyproject.toml": b""})
    uv_only(monkeypatch)
    project = package_environment.ProjectSource(repository.as_uri(), COMMIT)
    with pytest.raises(RuntimeError, match="Could not check out the environment project"):
        package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path, project)
    assert not package_environment.environment_ready(tmp_path / "pkg")


def test_a_checkout_of_another_commit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_git(args: list[str], **_: object) -> str:
        return "0" * 40 if args[-2:] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(package_registry, "_run_git", run_git)
    project = package_environment.ProjectSource("https://github.com/o/r", COMMIT)
    with pytest.raises(RuntimeError, match="git checked out another commit"):
        package_environment.install_environment("/bin/uv", tmp_path / "pkg", tmp_path, project)


@needs_uv
@pytest.mark.usefixtures("offline")
def test_a_package_environment_from_a_project_commit_runs_its_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_root = tmp_path / "wheel-source"
    bridge_package(wheel_root)  # builds the fixture wheel and a uv project around it
    built = wheel_root / "src" / "bridged"
    files = {name: (built / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    files |= {f"wheels/{w.name}": w.read_bytes() for w in (built / "wheels").iterdir()}
    repository, commit = project_repository(tmp_path / "upstream", files)
    value = f"git+https://github.com/example/project@{commit}"
    original = package_environment.parse_environment_project
    monkeypatch.setattr(
        package_environment,
        "parse_environment_project",
        lambda text: package_environment.ProjectSource(repository.as_uri(), original(text).commit),
    )
    source = tmp_path / "source" / "src" / "bridged"
    source.mkdir(parents=True)
    (source / "main.kedi").write_text("= installed\n")
    (source / "bridge.py").write_text("import kedi_env_fixture\nprint(kedi_env_fixture.VALUE)\n")
    manifest = tmp_path / "source" / "package.kedi"
    manifest.write_text(
        "> package: bridged:\n  version: 0.1.0\n  source: src/bridged\n  adapters:\n    bridged:\n"
        "      kind: agent\n      command: python bridge.py\n      environment: uv\n"
        f"      environment_project: {value}\n"
    )
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "home"))
    registry = tmp_path / "home" / "registry"

    installed = install_package(manifest, registry_root=registry)

    record = json.loads((installed.root / INSTALL_RECEIPT_NAME).read_text())["lifecycle"]
    (environment,) = [entry for entry in record["created"] if entry["kind"] == "environment"]
    assert environment["project"] == f"git+{repository.as_uri()}@{commit}"
    assert environment["command"] == "uv sync --frozen --no-dev"
    (adapter,) = package_adapters("bridged")
    assert adapter.command is not None
    ran = subprocess.run(list(adapter.command), capture_output=True, text=True, check=True)
    assert ran.stdout.strip() == "from the package environment"
    remove_package("bridged", registry_root=registry)
    assert list(registry.iterdir()) == []
