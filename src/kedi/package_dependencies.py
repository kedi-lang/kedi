"""Install manifest dependencies into the interpreter running Kedi."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Sequence


def install_python_dependencies(requirements: Sequence[str]) -> None:
    if not requirements:
        return

    # Keep the venv's executable path: resolving its symlink selects the base Python.
    if importlib.util.find_spec("pip") is not None:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
        ]
    elif uv := shutil.which("uv"):
        command = [uv, "pip", "install", "--python", sys.executable]
    else:
        raise RuntimeError(
            f"Cannot install Python dependencies into {sys.executable}: neither pip nor uv "
            "is available. Install pip in this interpreter or install uv, then retry."
        )

    try:
        subprocess.run([*command, "--", *requirements], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Python dependency installation failed for {sys.executable}; "
            "the Kedi package was not replaced. See the installer output above."
        ) from exc

    importlib.invalidate_caches()
