"""Project .env discovery (audit #214, item 7).

Bare ``dotenv.load_dotenv()`` anchors discovery at the caller's file
(python-dotenv's ``find_dotenv(usecwd=False)`` walks up from the file that
called it), i.e. at the installed library - it never sees the ``.env`` of
the project the user launched from. ``load_project_dotenv`` searches from
the current working directory instead, so each project loads its own
``.env``; existing environment variables still win.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kedi.utils import load_project_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = "\n".join(
    [
        "import os",
        "import kedi.settings  # import-time load_project_dotenv()",
        "print(os.environ.get('KEDI_214_PROJECT_VAR', '<missing>'))",
        "",
    ]
)


def _load_in_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    project: str,
    content: str,
    var: str,
) -> str | None:
    directory = tmp_path / project
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text(content, encoding="utf-8")
    monkeypatch.chdir(directory)
    monkeypatch.delenv(var, raising=False)
    try:
        load_project_dotenv()
        return os.environ.get(var)
    finally:
        os.environ.pop(var, None)


def test_load_project_dotenv_loads_cwd_env(tmp_path: Path, monkeypatch) -> None:
    assert (
        _load_in_project(
            tmp_path,
            monkeypatch,
            project="proj-a",
            content="KEDI_214_PROJECT_VAR=from-proj-a\n",
            var="KEDI_214_PROJECT_VAR",
        )
        == "from-proj-a"
    )


def test_load_project_dotenv_loads_distinct_projects(tmp_path: Path, monkeypatch) -> None:
    assert (
        _load_in_project(
            tmp_path,
            monkeypatch,
            project="proj-a",
            content="KEDI_214_PROJECT_VAR=from-proj-a\n",
            var="KEDI_214_PROJECT_VAR",
        )
        == "from-proj-a"
    )
    assert (
        _load_in_project(
            tmp_path,
            monkeypatch,
            project="proj-b",
            content="KEDI_214_PROJECT_VAR=from-proj-b\n",
            var="KEDI_214_PROJECT_VAR",
        )
        == "from-proj-b"
    )


def test_load_project_dotenv_finds_parent_env_from_subdirectory(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / ".env").write_text("KEDI_214_PROJECT_VAR=from-parent\n", encoding="utf-8")
    monkeypatch.chdir(project / "sub")
    monkeypatch.delenv("KEDI_214_PROJECT_VAR", raising=False)
    try:
        assert load_project_dotenv() is True
        assert os.environ.get("KEDI_214_PROJECT_VAR") == "from-parent"
    finally:
        os.environ.pop("KEDI_214_PROJECT_VAR", None)


def test_load_project_dotenv_keeps_existing_environment(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".env").write_text("KEDI_214_PROJECT_VAR=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("KEDI_214_PROJECT_VAR", "from-environment")
    assert load_project_dotenv() is True
    assert os.environ["KEDI_214_PROJECT_VAR"] == "from-environment"


def test_load_project_dotenv_returns_false_without_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_project_dotenv() is False


def test_importing_kedi_settings_loads_project_dotenv(tmp_path: Path) -> None:
    """Subprocess check: a real .py file run from a project directory (not
    `python -c`, where dotenv treats the run as interactive and falls back
    to cwd, masking the bug) must see the project's .env at import time."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("KEDI_214_PROJECT_VAR=from-project-dotenv\n", encoding="utf-8")
    probe = project / "run_probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "from-project-dotenv"
