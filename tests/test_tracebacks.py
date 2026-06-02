from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from kedi.cli import main as cli_main
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.source_loading import load_kedi_source_bundle


def test_runtime_error_carries_kedi_frames(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        "@boom(x: int) -> int:\n  [y: int] = `x + 1`\n  = <`1 / 0`>\n\n= <boom(`1`)>\n",
        encoding="utf-8",
    )
    program = parse_program(
        srcp.read_text(encoding="utf-8"),
        source_path=str(srcp),
    )
    rt = compile_program(program)

    with pytest.raises(KediExecutionError) as info:
        rt.run_main()

    err = info.value
    assert "Inline Python error" in str(err)
    assert any(frame.kind == "program" for frame in err.frames)
    assert any(frame.kind == "procedure" and frame.procedure_name == "boom" for frame in err.frames)
    assert any(frame.kind == "python_inline" for frame in err.frames)
    assert err.python_traces
    assert err.python_traces[-1].python_lineno == 1
    assert err.python_traces[-1].span is not None
    assert err.python_traces[-1].span.source_path == str(srcp)


def test_cli_renders_traceback_for_inline_python_error(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        "@boom() -> str:\n  = <`1 / 0`>\n\n= <boom()>\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert "Kedi stack trace:" in result.output
    assert f'File "{srcp}"' in result.output
    assert "in inline python" in result.output
    assert f"embedded python line 1 at {srcp}:2" in result.output
    assert "1 / 0" in result.output
    assert "procedure _main" not in result.output


def test_companion_file_source_map_uses_real_val_path(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    valp = tmp_path / "sample.val.kedi"
    srcp.write_text("@main() -> str:\n  = ok\n\n= <main()>\n", encoding="utf-8")
    valp.write_text("@eval: main:\n  > data: rows:\n    = `[(1, 2)]`\n", encoding="utf-8")

    loaded = load_kedi_source_bundle(srcp)
    program = parse_program(
        loaded.text,
        source_path=str(srcp),
        source_map=loaded.source_map,
    )

    assert getattr(program, "evals")[0].loc is not None
    assert getattr(program, "evals")[0].loc.source_path == str(valp.resolve())


def test_cli_renders_parse_error_with_location(tmp_path: Path) -> None:
    srcp = tmp_path / "bad.kedi"
    srcp.write_text("@boom(\n  = nope\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f"{srcp}:1:1: error:" in result.output
    assert "unexpected source at top level" in result.output
    assert "@boom(" in result.output


def test_cli_maps_fenced_python_error_to_inner_kedi_line(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        'hello\n\nworld\n\nfoo\n\nbar\n\nbaz\n\nqux\n```\nprint(f"{")\n```\n= none\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f'File "{srcp}", line 13, in python statement' in result.output
    assert f"embedded python line 1 at {srcp}:13" in result.output
    assert 'print(f"{")' in result.output


def test_cli_renders_toplevel_body_error_without_return(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("[x: int] = `1 / 0`\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert "Kedi stack trace:" in result.output
    assert f'File "{srcp}", line 1, in assignment x' in result.output
    assert "in inline python" in result.output
    assert "1 / 0" in result.output


def test_cli_renders_embedded_python_trace_once_for_assignment_block(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        '[x: str] = ```\n# comment\nreturn "value\n```\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f'File "{srcp}", line 1, in assignment x' in result.output
    assert f'File "{srcp}", line 3, in python block' in result.output
    assert result.output.count("embedded python line") == 1
    assert result.output.count('return "value') == 1
