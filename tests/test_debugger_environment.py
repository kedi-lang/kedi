from __future__ import annotations

import os
from pathlib import Path

import pytest
from kedi_debugger.server import validate_launch


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    for name in ("KEDI_ADAPTER", "KEDI_ADAPTER_MODEL", "MODEL_NAME", "DEBUG_TEST_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    program = tmp_path / "main.kedi"
    program.write_text("= done\n", encoding="utf-8")
    return program


def test_dotenv_loaded_before_worker_start_without_mutating_supervisor(project: Path) -> None:
    (project.parent / ".env").write_text(
        "KEDI_ADAPTER=langchain\nKEDI_ADAPTER_MODEL=openai:debug-test\n"
        "DEBUG_TEST_API_KEY='synthetic-env-secret'\n",
        encoding="utf-8",
    )
    _, environment = validate_launch({"program": str(project)})
    assert environment["KEDI_ADAPTER"] == "langchain"
    assert environment["KEDI_ADAPTER_MODEL"] == "openai:debug-test"
    assert environment["DEBUG_TEST_API_KEY"] == "synthetic-env-secret"
    assert "DEBUG_TEST_API_KEY" not in os.environ
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"


def test_dotenv_search_uses_launch_directory_and_nearest_parent(project: Path) -> None:
    cwd = project.parent / "workspace" / "nested"
    cwd.mkdir(parents=True)
    (project.parent / ".env").write_text("MODEL_NAME=wrong-parent\n", encoding="utf-8")
    (cwd.parent / ".env").write_text("MODEL_NAME=correct-project\n", encoding="utf-8")
    _, environment = validate_launch({"program": str(project), "cwd": str(cwd)})
    assert environment["MODEL_NAME"] == "correct-project"


def test_dotenv_precedence_interpolation_and_explicit_removal(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEDI_ADAPTER_MODEL", "inherited-model")
    monkeypatch.setenv("DEBUG_TEST_API_KEY", "inherited-secret")
    (project.parent / ".env").write_text(
        "KEDI_ADAPTER=langchain\nKEDI_ADAPTER_MODEL=file-model\n"
        "DEBUG_TEST_API_KEY=file-secret\n"
        "DEBUG_COMBINED=${KEDI_ADAPTER}:${KEDI_ADAPTER_MODEL}\n"
        "DEBUG_REMOVED=${DEBUG_TEST_API_KEY:-absent}\n"
        "DEBUG_EMPTY=\nDEBUG_UNSET\n",
        encoding="utf-8",
    )
    _, environment = validate_launch(
        {
            "program": str(project),
            "env": {"KEDI_ADAPTER": "pydantic", "DEBUG_TEST_API_KEY": None},
        }
    )
    assert environment["KEDI_ADAPTER"] == "pydantic"
    assert environment["KEDI_ADAPTER_MODEL"] == "inherited-model"
    assert environment["DEBUG_COMBINED"] == "pydantic:inherited-model"
    assert environment["DEBUG_REMOVED"] == "absent"
    assert "DEBUG_TEST_API_KEY" not in environment
    assert environment["DEBUG_EMPTY"] == ""
    assert "DEBUG_UNSET" not in environment
    assert os.environ["DEBUG_TEST_API_KEY"] == "inherited-secret"


@pytest.mark.parametrize("disabled", ["1", "true", "YES"])
def test_dotenv_loading_can_be_disabled(project: Path, disabled: str) -> None:
    (project.parent / ".env").write_text("DEBUG_TEST_API_KEY=unused\n", encoding="utf-8")
    _, environment = validate_launch(
        {"program": str(project), "env": {"PYTHON_DOTENV_DISABLED": disabled}}
    )
    assert "DEBUG_TEST_API_KEY" not in environment


@pytest.mark.parametrize(
    "text", ["DEBUG_TEST_API_KEY=secret\x00value\n", "BAD\x00KEY=secret\n", "'bad=key'=secret\n"]
)
def test_invalid_dotenv_entries_are_rejected_without_exposing_values(
    project: Path, text: str
) -> None:
    (project.parent / ".env").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="environment") as exc:
        validate_launch({"program": str(project)})
    assert "secret" not in str(exc.value)


def test_dotenv_supports_multiline_quotes_export_and_defaults(project: Path) -> None:
    (project.parent / ".env").write_text(
        'export DEBUG_TEST_API_KEY="first\nsecond"\n'
        "DEBUG_VALUE=${KEDI_ADAPTER_MODEL:-test-default} # comment\n",
        encoding="utf-8",
    )
    _, environment = validate_launch({"program": str(project)})
    assert environment["DEBUG_TEST_API_KEY"] == "first\nsecond"
    assert environment["DEBUG_VALUE"] == "test-default"
