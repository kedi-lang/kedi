"""Package environments: a package adapter's command runs in a Python environment of its own.

An adapter with ``environment: uv`` brings a uv project (``pyproject.toml`` and ``uv.lock``) at
the root of its package's ``source``. Kedi installs it with ``uv sync --frozen`` into the
installed package's own directory, never into the interpreter running Kedi: a harness can pin
dependencies Kedi cannot share (Hermes pins openai 2.x; Kedi uses 3.x).

``environment_project`` names the project instead, as a GitHub repository at an exact commit
(``git+https://github.com/OWNER/REPOSITORY@<40-hex commit>``). Kedi checks out that one commit,
and ``uv sync`` installs the project itself, editable, from its own lock: the way a harness that
ships no wheel (Hermes) supports being installed.

Everything stays inside ``<package>/.kedi-env/``, so it goes with the package:
- ``venv/``: the environment;
- ``python/``: the environment's Python, one uv manages and downloads there, and ``python-bin/``
  for its executables. A venv built on a system Python breaks when that Python is upgraded or
  removed; ``UV_PYTHON_PREFERENCE`` lets uv use an installed one instead;
- ``project/``: the ``environment_project`` checkout, without its ``.git``, which an editable
  install keeps using;
- ``cache/``: uv's cache while it installs, removed when it ends;
- ``ready``: written when the sync has finished. An installation that was killed leaves none,
  and Kedi then refuses to run the adapter rather than run a half-installed environment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .lang import PackageMetadata
from .package_dependencies import find_uv

ENVIRONMENT_DIRECTORY = ".kedi-env"
UV_SYNC = ("sync", "--frozen", "--no-dev", "--no-install-project")
UV_SYNC_PROJECT = ("sync", "--frozen", "--no-dev")
PROJECT_FILES = ("pyproject.toml", "uv.lock")
READY_MARKER = "ready"
PROJECT_FETCH_TIMEOUT_SECONDS = 30 * 60
"""A harness's checkout can be hundreds of megabytes: Hermes's is about 110 MB."""
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ProjectSource:
    repository: str
    commit: str


def parse_environment_project(value: str) -> ProjectSource:
    """``git+https://github.com/OWNER/REPOSITORY@<commit>``: a repository and the exact commit, as
    ``kedi add git+https://…`` takes a repository. A tag or branch can move; a commit cannot."""
    from .package_registry import _git_https_url

    source, separator, commit = value.strip().rpartition("@")
    if not separator or not _COMMIT_RE.fullmatch(commit):
        raise ValueError(
            f"environment_project {value!r} must end with @ and a full 40-character commit hash, "
            "such as git+https://github.com/OWNER/REPOSITORY@<commit>; a tag or branch can move"
        )
    try:
        repository = _git_https_url(source)
    except ValueError as exc:
        raise ValueError(f"environment_project {value!r}: {exc}") from exc
    return ProjectSource(repository, commit)


def environment_project(metadata: PackageMetadata) -> ProjectSource | None:
    """The project the package's environment comes from, if its adapters name one. Manifest
    validation makes every adapter of a package name the same one."""
    for adapter in metadata.adapters.values():
        if adapter.environment == "uv" and adapter.environment_project is not None:
            return parse_environment_project(adapter.environment_project)
    return None


def needs_environment(metadata: PackageMetadata) -> bool:
    return any(adapter.environment == "uv" for adapter in metadata.adapters.values())


def environment_installer(metadata: PackageMetadata, source: Path) -> str:
    """The uv that will install the package's environment, after checking what it needs; raised
    before any state changes."""
    missing = [name for name in PROJECT_FILES if not (source / name).is_file()]
    if missing and environment_project(metadata) is None:
        raise ValueError(
            f"Kedi package {metadata.name!r} declares environment: uv, but its source has no "
            f"{' or '.join(missing)}: the environment is the uv project at the source's root, "
            "or the one environment_project names"
        )
    uv = find_uv()
    if uv is None:
        raise ValueError(
            f"Kedi package {metadata.name!r} needs uv to install its environment, and uv is not "
            "installed. Run `kedi python install uv`, then install the package again."
        )
    return uv


def install_environment(
    uv: str, package_root: Path, source: Path, project: ProjectSource | None = None
) -> None:
    """``uv sync`` into ``package_root``'s environment directory: of ``source``'s project, or of
    ``project``, checked out there first."""
    root = package_root / ENVIRONMENT_DIRECTORY
    cache = root / "cache"
    environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(root / "venv"),
        "UV_PYTHON_INSTALL_DIR": str(root / "python"),
        "UV_PYTHON_BIN_DIR": str(root / "python-bin"),
        "UV_CACHE_DIR": str(cache),
        "UV_PYTHON_PREFERENCE": os.environ.get("UV_PYTHON_PREFERENCE", "only-managed"),
    }
    # An activated environment is not the package's; uv would warn about it, or use it.
    environment.pop("VIRTUAL_ENV", None)
    sync = UV_SYNC
    try:
        if project is not None:
            source = _check_out_project(project, root / "project")
            sync = UV_SYNC_PROJECT
        try:
            subprocess.run([uv, *sync, "--project", str(source)], env=environment, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Installing the environment of {package_root.name} with uv failed; see uv's "
                "output above. The previously installed version, if any, was kept."
            ) from exc
        (root / READY_MARKER).write_text("", encoding="utf-8")
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def _check_out_project(project: ProjectSource, destination: Path) -> Path:
    """Fetches the one commit, without history or tags, checks it out, and drops ``.git``: the
    commit is the content's identity, and nothing reads the repository afterwards."""
    from .package_registry import _run_git

    destination.mkdir(parents=True)
    context = f"Could not check out the environment project {project.repository}@{project.commit}"

    def git(*args: str) -> str:
        return _run_git(
            ["-C", str(destination), *args],
            error_context=context,
            timeout=PROJECT_FETCH_TIMEOUT_SECONDS,
        )

    git("init", "--quiet")
    git("remote", "add", "origin", project.repository)
    git("fetch", "--quiet", "--depth", "1", "--no-tags", "origin", project.commit)
    git("checkout", "--detach", "--quiet", project.commit)
    if git("rev-parse", "HEAD") != project.commit:
        raise RuntimeError(f"{context}: git checked out another commit")
    shutil.rmtree(destination / ".git")
    return destination


def environment_ready(package_root: Path) -> bool:
    """Whether the package's environment finished installing."""
    return (package_root / ENVIRONMENT_DIRECTORY / READY_MARKER).is_file()


def environment_program(package_root: Path, program: str) -> str | None:
    """``program`` in the package's environment, or None when the environment has none."""
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    name = f"{program}.exe" if sys.platform == "win32" else program
    candidate = package_root / ENVIRONMENT_DIRECTORY / "venv" / scripts / name
    return str(candidate) if candidate.is_file() else None
