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


def test_cli_requires_source_or_command() -> None:
    result = CliRunner().invoke(cli_module.main, [])

    assert result.exit_code == 2
    assert "Missing argument 'SOURCE'." in result.output


def test_cli_passes_extra_args_to_program_env(tmp_path: Path) -> None:
    source = tmp_path / "main.kedi"
    source.write_text('= `f"{args.id}:{args.enabled}:{args.missing}"`\n', encoding="utf-8")

    result = CliRunner().invoke(cli_module.main, [str(source), "--id", "1", "--enabled"])

    assert result.exit_code == 0
    assert result.output == "1:True:None\n"


def test_cli_command_runs_inline_source() -> None:
    result = CliRunner().invoke(cli_module.main, ["-c", "= done"])

    assert result.exit_code == 0
    assert result.output == "done\n"


def test_cli_install_installs_local_package(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "src" / "kedi_http"
    source.mkdir(parents=True)
    manifest = tmp_path / "package.kedi"
    manifest.write_text(
        "> package: kedi_http:\n  version: 1.0\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= done\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    result = CliRunner().invoke(cli_module.main, ["install", str(manifest)])

    assert result.exit_code == 0
    assert "Installed kedi_http" in result.output
    assert (tmp_path / "kedi-home" / "registry" / "kedi_http" / "package.kedi").is_file()


def test_cli_install_uses_package_manifest_in_the_current_directory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "src" / "kedi_http"
    source.mkdir(parents=True)
    (tmp_path / "package.kedi").write_text(
        "> package: kedi_http:\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= done\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    result = CliRunner().invoke(cli_module.main, ["install"])

    assert result.exit_code == 0
    assert "Installed kedi_http" in result.output


def test_cli_add_installs_from_the_mock_registry(tmp_path: Path, monkeypatch: Any) -> None:
    package = tmp_path / "mock-registry" / "kedi_http"
    source = package / "src" / "kedi_http"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: kedi_http:\n  version: 1.0\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= done\n", encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))
    monkeypatch.setenv("KEDI_REGISTRY_MOCK_ROOT", str(package.parent))

    result = CliRunner().invoke(cli_module.main, ["add", "kedi_http"])

    assert result.exit_code == 0
    assert "Installed kedi_http" in result.output


def test_cli_add_requires_a_package_name() -> None:
    result = CliRunner().invoke(cli_module.main, ["add"])

    assert result.exit_code == 2
    assert "Missing argument for 'kedi add'." in result.output


def test_cli_package_commands_reject_extra_arguments() -> None:
    add_result = CliRunner().invoke(cli_module.main, ["add", "kedi_http", "unexpected"])
    install_result = CliRunner().invoke(cli_module.main, ["install", "one.kedi", "two.kedi"])

    assert add_result.exit_code == 2
    assert "accepts exactly one package source" in add_result.output
    assert install_result.exit_code == 2
    assert "accepts at most one manifest path" in install_result.output


def test_cli_package_commands_reject_options_and_render_add_errors(
    monkeypatch: Any,
) -> None:
    option_result = CliRunner().invoke(cli_module.main, ["add", "--unexpected"])
    monkeypatch.setattr(
        cli_module, "add_package", lambda _source: (_ for _ in ()).throw(ValueError("bad package"))
    )
    error_result = CliRunner().invoke(cli_module.main, ["add", "demo"])

    assert option_result.exit_code == 2
    assert "Unexpected option" in option_result.output
    assert error_result.exit_code == 1
    assert "Error: bad package" in error_result.output


def test_cli_package_commands_have_specific_help() -> None:
    install_result = CliRunner().invoke(cli_module.main, ["install", "--help"])
    add_result = CliRunner().invoke(cli_module.main, ["add", "--help"])

    assert install_result.exit_code == 0
    assert "Usage: kedi install [PACKAGE.KEDI]" in install_result.output
    assert add_result.exit_code == 0
    assert "Usage: kedi add <PACKAGE-NAME" in add_result.output


def test_cli_general_help_uses_click_help() -> None:
    result = CliRunner().invoke(cli_module.main, ["--help"])

    assert result.exit_code == 0
    assert "Run a .kedi program" in result.output


def test_cli_package_errors_are_rendered_as_click_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli_module.main, ["install", str(tmp_path / "missing.kedi")])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Kedi stack trace" not in result.output


def test_cli_add_git_source_reports_the_cloned_revision(monkeypatch: Any) -> None:
    class Installed:
        name = "kedi_http"
        root = Path("/tmp/registry/kedi_http")
        source_revision = "abc123"

    monkeypatch.setattr(cli_module, "add_package", lambda _source: Installed())

    result = CliRunner().invoke(
        cli_module.main,
        ["add", "git+https://github.com/kedi-lang/kedi-http.git"],
    )

    assert result.exit_code == 0
    assert "Installed kedi_http" in result.output
    assert "abc123" in result.output


def test_cli_acp_command_sets_process_env(monkeypatch: Any) -> None:
    seen: dict[str, str | None] = {}

    def fake_run_source_cli(*args: Any, **kwargs: Any) -> tuple[int, str]:
        seen["command"] = cli_module.os.getenv("KEDI_ACP_AGENT_COMMAND")
        return 0, "ok"

    monkeypatch.delenv("KEDI_ACP_AGENT_COMMAND", raising=False)
    monkeypatch.setattr(cli_module, "run_source_cli", fake_run_source_cli)

    result = CliRunner().invoke(
        cli_module.main,
        ["-c", "= done", "--adapter", "acp", "--acp-command", "vsh run --acp"],
    )

    assert result.exit_code == 0
    assert result.output == "ok\n"
    assert seen == {"command": "vsh run --acp"}


def test_cli_command_passes_extra_args_to_program_env() -> None:
    result = CliRunner().invoke(
        cli_module.main,
        [
            "-c",
            '= `f"{args.id}:{args.enabled}:{args.missing}"`',
            "--id",
            "1",
            "--enabled",
        ],
    )

    assert result.exit_code == 0
    assert result.output == "1:True:None\n"


def test_cli_command_parse_flag_and_parse_command() -> None:
    flag_result = CliRunner().invoke(cli_module.main, ["-c", "= done", "--parse"])
    short_flag_result = CliRunner().invoke(cli_module.main, ["-c", "= done", "-p"])
    command_result = CliRunner().invoke(cli_module.main, ["parse", "-c", "= done"])

    assert flag_result.exit_code == 0
    assert flag_result.output == "Parsed successfully.\n"
    assert short_flag_result.exit_code == 0
    assert short_flag_result.output == "Parsed successfully.\n"
    assert command_result.exit_code == 0
    assert command_result.output == "Parsed successfully.\n"


def test_cli_command_renders_parse_error() -> None:
    result = CliRunner().invoke(cli_module.main, ["-c", "@boom(\n  = nope"])

    assert result.exit_code == 1
    assert "Kedi parse error:" in result.output
    assert "Kedi error: ParseError" in result.output
    assert "unexpected source at top level" in result.output
    assert 'File "<command>", line 1, column 1' in result.output


def test_cli_command_exits_with_nonzero_run_source_code(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli_module, "run_source_cli", lambda *args, **kwargs: (7, ""))

    result = CliRunner().invoke(cli_module.main, ["-c", "= done"])

    assert result.exit_code == 7
    assert result.output == ""


def test_cli_command_rejects_file_and_file_bound_modes(tmp_path: Path) -> None:
    source = _source(tmp_path)

    with_positional_arg = CliRunner().invoke(
        cli_module.main, ["-c", "= `args.name`", "--name", "Ada"]
    )
    with_test = CliRunner().invoke(cli_module.main, ["-c", "= done", "--test"])

    assert with_positional_arg.exit_code == 0
    assert with_positional_arg.output == "Ada\n"
    assert source.exists()
    assert with_test.exit_code == 2
    assert "-c/--command only supports direct run or --parse." in with_test.output


def test_cli_parse_command_renders_kedi_parse_error(tmp_path: Path) -> None:
    source = tmp_path / "bad.kedi"
    source.write_text("@boom(\n  = nope\n", encoding="utf-8")

    result = CliRunner().invoke(cli_module.main, ["parse", str(source)])

    assert result.exit_code == 1
    assert "Kedi parse error:" in result.output
    assert "Kedi error: ParseError" in result.output
    assert "unexpected source at top level" in result.output
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
    short_flag_result = CliRunner().invoke(cli_module.main, [str(source), "-p"])

    assert command_result.exit_code == 0
    assert command_result.output == "Parsed successfully.\n"
    assert flag_result.exit_code == 0
    assert flag_result.output == "Parsed successfully.\n"
    assert short_flag_result.exit_code == 0
    assert short_flag_result.output == "Parsed successfully.\n"


def test_cli_parse_success_uses_green_output() -> None:
    result = CliRunner().invoke(cli_module.main, ["-c", "= done", "--parse"], color=True)

    assert result.exit_code == 0
    assert "\x1b[32mParsed successfully.\x1b[0m" in result.output


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
