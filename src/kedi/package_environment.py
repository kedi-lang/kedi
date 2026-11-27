"""Package environments: a package adapter's command runs in a Python environment of its own.

An adapter with ``environment: uv`` brings a uv project (``pyproject.toml`` and ``uv.lock``) at
the root of its package's ``source``. Kedi installs it with ``uv sync --frozen`` into the
installed package's own directory, never into the interpreter running Kedi: a harness can pin
dependencies Kedi cannot share (Hermes pins openai 2.x; Kedi uses 3.x).

Everything uv creates stays inside ``<package>/.kedi-env/``, so it goes with the package:
- ``venv/``: the environment;
- ``python/``: the environment's Python, one uv manages and downloads there, and ``python-bin/``
  for its executables. A venv built on a system Python breaks when that Python is upgraded or
  removed; ``UV_PYTHON_PREFERENCE`` lets uv use an installed one instead;
- ``cache/``: uv's cache while it installs, removed when it ends.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .lang import PackageMetadata
from .package_dependencies import find_uv

ENVIRONMENT_DIRECTORY = ".kedi-env"
UV_SYNC = ("sync", "--frozen", "--no-dev", "--no-install-project")
PROJECT_FILES = ("pyproject.toml", "uv.lock")


def needs_environment(metadata: PackageMetadata) -> bool:
    return any(adapter.environment == "uv" for adapter in metadata.adapters.values())


def environment_installer(metadata: PackageMetadata, source: Path) -> str:
    """The uv that will install the package's environment, after checking what it needs; raised
    before any state changes."""
    missing = [name for name in PROJECT_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(
            f"Kedi package {metadata.name!r} declares environment: uv, but its source has no "
            f"{' or '.join(missing)}: the environment is the uv project at the source's root"
        )
    uv = find_uv()
    if uv is None:
        raise ValueError(
            f"Kedi package {metadata.name!r} needs uv to install its environment, and uv is not "
            "installed. Run `kedi python install uv`, then install the package again."
        )
    return uv


def install_environment(uv: str, package_root: Path, source: Path) -> None:
    """``uv sync`` of ``source``'s project into ``package_root``'s environment directory."""
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
    try:
        subprocess.run([uv, *UV_SYNC, "--project", str(source)], env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Installing the environment of {package_root.name} with uv failed; see uv's output "
            "above. The previously installed version, if any, was kept."
        ) from exc
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def environment_program(package_root: Path, program: str) -> str | None:
    """``program`` in the package's environment, or None when the environment has none."""
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    name = f"{program}.exe" if sys.platform == "win32" else program
    candidate = package_root / ENVIRONMENT_DIRECTORY / "venv" / scripts / name
    return str(candidate) if candidate.is_file() else None
