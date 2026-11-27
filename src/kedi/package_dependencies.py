"""Install and uninstall Python distributions in the interpreter running Kedi."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

from packaging.utils import canonicalize_name

BOOTSTRAP_HINT = (
    "Install pip in this interpreter (`python -m ensurepip`), or install uv "
    "(https://docs.astral.sh/uv/getting-started/installation/), then retry."
)


def find_uv() -> str | None:
    """The uv executable: on ``PATH``, beside this interpreter, or from the ``uv`` package.

    ``kedi python install uv`` puts uv beside the interpreter, whose scripts directory need not
    be on ``PATH``."""
    if found := shutil.which("uv"):
        return found
    beside = Path(sysconfig.get_path("scripts")) / ("uv.exe" if sys.platform == "win32" else "uv")
    if beside.is_file():
        return str(beside)
    try:
        uv = importlib.import_module("uv")
        return str(uv.find_uv_bin())
    except (ImportError, AttributeError, FileNotFoundError):
        return None


def _installer(action: str) -> list[str]:
    # Keep the venv's executable path: resolving its symlink selects the base Python.
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", action, "--disable-pip-version-check"]
        return [*command, "--no-input"] if action == "install" else [*command, "--yes"]
    if uv := find_uv():
        return [uv, "pip", action, "--python", sys.executable]
    raise RuntimeError(
        f"Cannot {action} Python packages in {sys.executable}: neither pip nor uv is "
        f"available. {BOOTSTRAP_HINT}"
    )


def install_python_dependencies(requirements: Sequence[str]) -> None:
    if not requirements:
        return
    command = _installer("install")
    try:
        subprocess.run([*command, "--", *requirements], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Python dependency installation failed for {sys.executable}; "
            "the Kedi package was not replaced. See the installer output above."
        ) from exc
    importlib.invalidate_caches()


def install_python_requirements(requirements: Sequence[str]) -> None:
    """``kedi python install``: the user's own installs into Kedi's interpreter. No package
    records them, so no ``kedi remove`` removes them."""
    command = _installer("install")
    try:
        subprocess.run([*command, "--", *requirements], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Installing {', '.join(requirements)} into {sys.executable} failed; see the "
            "installer output above."
        ) from exc
    importlib.invalidate_caches()


def uninstall_python_distributions(names: Sequence[str]) -> None:
    if not names:
        return
    command = _installer("uninstall")
    try:
        subprocess.run([*command, *names], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Uninstalling {', '.join(names)} from {sys.executable} failed; see the "
            "installer output above."
        ) from exc
    importlib.invalidate_caches()


def installed_distributions() -> dict[str, str]:
    """The interpreter's installed distributions: canonical name to version."""
    found: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name and (canonical := canonicalize_name(name)) not in found:
            found[canonical] = distribution.version
    return found
