from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import kedi.cli as cli_module


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "main.kedi"
    source.write_text("= ok\n", encoding="utf-8")
    return source


def test_cli_module_reloads_without_path_mutation() -> None:
    reloaded = importlib.reload(cli_module)

    assert not hasattr(reloaded, "SRC_DIR")


def test_cli_reports_source_read_errors() -> None:
    result = CliRunner().invoke(cli_module.main, ["missing.kedi"])

    assert result.exit_code == 1
    assert "Error reading missing.kedi" in result.output


def test_cli_passes_extra_args_to_program_env(tmp_path: Path) -> None:
    source = tmp_path / "main.kedi"
    source.write_text('= `f"{args.id}:{args.enabled}:{args.missing}"`\n', encoding="utf-8")

    result = CliRunner().invoke(cli_module.main, [str(source), "--id", "1", "--enabled"])

    assert result.exit_code == 0
    assert result.output == "1:True:None\n"


def test_cli_parse_command_renders_kedi_parse_error(tmp_path: Path) -> None:
    source = tmp_path / "bad.kedi"
    source.write_text("@boom(\n  = nope\n", encoding="utf-8")

    result = CliRunner().invoke(cli_module.main, ["parse", str(source)])

    assert result.exit_code == 1
    assert "Kedi parse error: unexpected source at top level" in result.output
    assert f'File "{source}", line 1, column 1' in result.output
    assert "@boom(" in result.output
    assert "^" in result.output


def test_cli_parse_command_requires_source_argument() -> None:
    result = CliRunner().invoke(cli_module.main, ["parse"])

    assert result.exit_code == 2
    assert "Missing argument 'SOURCE'." in result.output


def test_cli_parse_flag_parses_without_compile(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = _source(tmp_path)

    def fail_compile(*args: Any, **kwargs: Any) -> tuple[bool, str | None]:
        raise AssertionError("parse-only should not run codegen")

    monkeypatch.setattr(cli_module, "maybe_run_codegen", fail_compile)

    command_result = CliRunner().invoke(cli_module.main, ["parse", str(source)])
    flag_result = CliRunner().invoke(cli_module.main, [str(source), "--parse"])

    assert command_result.exit_code == 0
    assert command_result.output == ""
    assert flag_result.exit_code == 0
    assert flag_result.output == ""


def test_cli_optimize_success_and_failure_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = _source(tmp_path)
    optimize_calls: list[dict[str, Any]] = []

    def optimize_success(*args: Any, **kwargs: Any) -> tuple[int, str]:
        optimize_calls.append(kwargs)
        return 0, "optimized"

    monkeypatch.setattr(cli_module, "run_optimization_cli", optimize_success)
    monkeypatch.setattr(cli_module, "maybe_run_codegen", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(cli_module, "run_program_cli", lambda *args, **kwargs: (0, "program"))

    result = CliRunner().invoke(cli_module.main, [str(source), "--optimize", "--optimizer", "gepa"])

    assert result.exit_code == 0
    assert "optimized" in result.output
    assert "program" in result.output
    assert optimize_calls[0]["reflection_model"] is not None

    monkeypatch.setattr(cli_module, "run_optimization_cli", lambda *args, **kwargs: (4, "bad opt"))
    result = CliRunner().invoke(cli_module.main, [str(source), "--optimize"])

    assert result.exit_code == 4
    assert "bad opt" in result.output

    monkeypatch.setattr(cli_module, "run_optimization_cli", lambda *args, **kwargs: (0, ""))
    result = CliRunner().invoke(cli_module.main, [str(source), "--optimize"])
    assert result.exit_code == 0
    assert result.output == "program\n"


def test_cli_validation_codegen_program_and_no_cache_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = _source(tmp_path)

    monkeypatch.setattr(cli_module, "run_tests_cli", lambda *args, **kwargs: (2, "tests bad"))
    result = CliRunner().invoke(cli_module.main, [str(source), "--test"])
    assert result.exit_code == 2
    assert "tests bad" in result.output

    monkeypatch.setattr(cli_module, "run_tests_cli", lambda *args, **kwargs: (0, "tests ok"))
    result = CliRunner().invoke(cli_module.main, [str(source), "--test"])
    assert result.exit_code == 0
    assert "tests ok" in result.output

    monkeypatch.setattr(cli_module, "run_evals_cli", lambda *args, **kwargs: (3, "evals bad"))
    result = CliRunner().invoke(cli_module.main, [str(source), "--eval"])
    assert result.exit_code == 3
    assert "evals bad" in result.output

    monkeypatch.setattr(cli_module, "run_evals_cli", lambda *args, **kwargs: (0, ""))
    result = CliRunner().invoke(cli_module.main, [str(source), "--eval"])
    assert result.exit_code == 0
    assert result.output == ""

    monkeypatch.setattr(cli_module, "maybe_run_codegen", lambda *args, **kwargs: (False, None))
    result = CliRunner().invoke(cli_module.main, [str(source)])
    assert result.exit_code == 1
    assert "Code generation failed." in result.output

    cache = source.with_suffix(".cache.kedi")
    cache.write_text("@cached():\n  = ok\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "maybe_run_codegen", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(cli_module, "run_program_cli", lambda *args, **kwargs: (5, "program bad"))

    result = CliRunner().invoke(cli_module.main, [str(source), "--no-cache"])

    assert result.exit_code == 5
    assert "program bad" in result.output
    assert not cache.exists()
