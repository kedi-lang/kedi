"""Build and verify release wheels in a clean environment on any supported OS."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="kedi distribution ") as directory:
        temporary = Path(directory)
        dist = temporary / "dist"
        for project in (ROOT / "tree-sitter-kedi", ROOT):
            # uv builds a wheel from the sdist when neither format is selected.
            subprocess.run(
                ["uv", "build", str(project), "--python", args.python, "--out-dir", str(dist)],
                cwd=ROOT,
                check=True,
                timeout=600,
            )
        environment = temporary / "venv"
        subprocess.run(
            ["uv", "venv", "--python", args.python, str(environment)],
            cwd=temporary,
            check=True,
            timeout=300,
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        wheels = sorted(dist.glob("*.whl"))
        assert len(wheels) == 2, wheels
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), *map(str, wheels)],
            cwd=temporary,
            check=True,
            timeout=600,
        )
        subprocess.run(
            ["uv", "pip", "check", "--python", str(python)],
            cwd=temporary,
            check=True,
            timeout=60,
        )
        subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/distribution_smoke.py")],
            cwd=temporary,
            check=True,
            timeout=300,
        )


if __name__ == "__main__":
    main()
