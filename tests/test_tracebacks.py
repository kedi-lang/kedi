from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from kedi.cli import main as cli_main
from kedi.error_rendering import (
    RenderStyle,
    _best_python_trace_frame_index,
    _column_caret,
    _inline_python_source_caret,
    _line_at,
    _role_style,
    _span_caret,
    render_execution_error,
    render_parse_error,
)
from kedi.errors import (
    KediExecutionError,
    KediKeyExecutionError,
    KediNameError,
    KediPythonOrigin,
    KediPythonTrace,
    KediTraceFrame,
    KediTypeExecutionError,
    KediValueExecutionError,
    build_execution_error,
    frame_label,
    python_error_message,
)
from kedi.executors.base import DefaultExecutor
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import Span
from kedi.lang.parser.diagnostics import KediDiagnostic, KediParseError
from kedi.source_loading import load_kedi_source_bundle, single_file_source_map


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


def test_missing_template_name_is_rendered_as_name_error() -> None:
    runtime = compile_program(
        parse_program(
            ">> <user>\n",
            source_path="<playground>",
        )
    )

    with pytest.raises(KediNameError) as info:
        runtime.run_main()

    rendered = str(info.value)
    assert 'File "<playground>", line 1, in template block' in rendered
    assert "Kedi error: NameError" in rendered
    assert "Value not found in environment: user" in rendered
    assert "KeyError" not in rendered


def test_traceback_uses_source_map_path_when_span_has_no_path(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("[value] = ok\n`assert value == 'nope'`\n= <value>\n", encoding="utf-8")
    bundle = load_kedi_source_bundle(srcp, include_val=False, include_cache=False)
    program = parse_program(bundle.text, source_map=bundle.source_map)
    rt = compile_program(program)

    with pytest.raises(KediExecutionError) as info:
        rt.run_main()

    err = info.value
    trace = err.__traceback__
    last_filename = None
    while trace is not None:
        last_filename = trace.tb_frame.f_code.co_filename
        trace = trace.tb_next

    # each embedded Python snippet gets a UNIQUE virtual co_filename for
    # traceback identity; the real source path is preserved on the python trace span
    # (used for rendering), not on the raw Python frame.
    assert last_filename is not None and last_filename.startswith("<kedi-python:")
    assert err.python_traces[-1].span is not None
    assert err.python_traces[-1].span.source_path == str(srcp.resolve())
    assert err.python_traces[-1].kedi_lineno == 2


def test_executor_falls_back_to_source_map_for_pathless_span(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("[value] = ok\n`assert value == 'nope'`\n= <value>\n", encoding="utf-8")
    rt = compile_program(parse_program("= ok"))
    rt._source_map = single_file_source_map(srcp, srcp.read_text(encoding="utf-8"))
    span = Span(
        start_byte=13,
        end_byte=37,
        start_line=2,
        start_col=1,
        end_line=2,
        end_col=25,
    )

    with pytest.raises(KediExecutionError) as info:
        with rt.trace_frame(KediTraceFrame(kind="python_side_effects", label="python", span=span)):
            DefaultExecutor().execute_side_effects(rt, "assert value == 'nope'", {"value": "ok"})

    err = info.value
    assert err.python_traces[-1].span is not None
    assert err.python_traces[-1].span.source_path == str(srcp)
    assert err.python_traces[-1].kedi_lineno == 2


def test_executor_keeps_virtual_filename_when_source_map_cannot_resolve(
    tmp_path: Path,
) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("= ok\n", encoding="utf-8")
    rt = compile_program(parse_program("= ok"))
    rt._source_map = single_file_source_map(srcp, srcp.read_text(encoding="utf-8"))
    span = Span(
        start_byte=0,
        end_byte=1,
        start_line=99,
        start_col=1,
        end_line=99,
        end_col=2,
    )

    with pytest.raises(KediExecutionError) as info:
        with rt.trace_frame(KediTraceFrame(kind="python_side_effects", label="python", span=span)):
            DefaultExecutor().execute_side_effects(rt, "raise RuntimeError('boom')", {})

    assert info.value.python_traces[-1].filename.startswith("<kedi-python:side_effects:")


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
    assert "embedded Python line 1" in result.output
    assert "Inline Python error: ZeroDivisionError" in result.output
    assert "division by zero" in result.output
    assert "<kedi-python:" not in result.output
    assert "            ^^^^^" in result.output
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
    assert "Kedi parse error:" in result.output
    assert "Kedi error: ParseError" in result.output
    assert f'File "{srcp}", line 1, column 1' in result.output
    assert "unexpected source at top level" in result.output
    assert "@boom(" in result.output
    assert "^" in result.output


def test_cli_maps_fenced_python_error_to_inner_kedi_line(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        '[a] = hello\n\n[b] = world\n\n[c] = foo\n\n[d] = bar\n\n[e] = baz\n\n[f] = qux\n```\nprint(f"{")\n```\n= none\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f'File "{srcp}", line 13, in python statement' in result.output
    assert "embedded Python line 1" in result.output
    assert 'print(f"{")' in result.output
    assert "Inline Python error: SyntaxError" in result.output


def test_cli_renders_toplevel_body_error_without_return(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("[x: int] = `1 / 0`\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert "Kedi stack trace:" in result.output
    assert f'File "{srcp}", line 1, in variable initialization x' in result.output
    assert "in inline python" in result.output
    assert "Inline Python error: ZeroDivisionError" in result.output
    assert "1 / 0" in result.output


def test_cli_renders_embedded_python_trace_once_for_variable_initialization_block(
    tmp_path: Path,
) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        '[x: str] = ```\n# comment\nreturn "value\n```\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f'File "{srcp}", line 1, in variable initialization x' in result.output
    assert f'File "{srcp}", line 3, in python block' in result.output
    assert result.output.count("embedded Python line") == 1
    assert result.output.count('return "value') == 1
    assert "Python block error: SyntaxError" in result.output
    assert "<kedi-python:" not in result.output
    assert "detected at line" not in result.output


def test_cli_labels_nested_python_trace_as_call_then_error(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text(
        "[x: int] = ```\ndef fn():\n    return 1 / 0\nreturn fn()\n```\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert f'File "{srcp}", line 3, in python block' in result.output
    assert "embedded Python call, line 3" in result.output
    assert "embedded Python error, line 2" in result.output
    assert "embedded Python line" not in result.output
    assert "Python block error: ZeroDivisionError" in result.output
    assert "division by zero" in result.output


def test_cli_renders_missing_import_with_kedi_trace(tmp_path: Path) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("> import: missing_registry_module\n\n= ok\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert "Execution error:" not in result.output
    assert "Kedi stack trace:" in result.output
    assert f'File "{srcp}", line 1, in import missing_registry_module' in result.output
    assert "Kedi error: ModuleNotFoundError" in result.output
    assert "missing_registry_module.kedi" in result.output


def test_cli_unexpected_exception_uses_kedi_error_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    srcp = tmp_path / "sample.kedi"
    srcp.write_text("= ok\n", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("kedi.cli.run_program_cli", _boom)

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp)])

    assert result.exit_code == 1
    assert "Execution error:" not in result.output
    assert "Kedi stack trace:" in result.output
    assert "Kedi error: RuntimeError" in result.output
    assert "boom" in result.output


def test_error_helpers_cover_message_subclasses_and_frame_labels() -> None:
    assert python_error_message(RuntimeError(), "inline") == "Inline Python error: RuntimeError"
    assert (
        python_error_message(RuntimeError("boom"), "unknown") == "Python error: RuntimeError: boom"
    )

    assert isinstance(
        build_execution_error(KeyError("missing"), frames=(), python_origins={}),
        KediKeyExecutionError,
    )
    assert isinstance(
        build_execution_error(NameError("missing"), frames=(), python_origins={}),
        KediNameError,
    )
    assert isinstance(
        build_execution_error(TypeError("bad type"), frames=(), python_origins={}),
        KediTypeExecutionError,
    )
    assert isinstance(
        build_execution_error(ValueError("bad value"), frames=(), python_origins={}),
        KediValueExecutionError,
    )

    assert frame_label("program") == "top-level program"
    assert frame_label("procedure") == "procedure"
    assert frame_label("procedure", "run") == "procedure run"
    assert frame_label("template") == "template line"
    assert frame_label("assignment") == "assignment"
    assert frame_label("assignment_block") == "assignment block"
    assert frame_label("return") == "return"
    assert frame_label("return_block") == "return block"
    assert frame_label("python_inline") == "inline python"
    assert frame_label("python_block") == "python block"
    assert frame_label("python_side_effects") == "python statement"
    assert frame_label("prelude") == "prelude"
    assert frame_label("type_expression") == "type expression"
    assert frame_label("custom_frame") == "custom frame"


def test_python_origin_trace_maps_source_line_when_source_map_is_available(
    tmp_path: Path,
) -> None:
    source_path = str(tmp_path / "sample.kedi")
    span = Span(
        0,
        10,
        3,
        1,
        3,
        11,
        source_path=source_path,
    )

    class SourceMap:
        def get_line(self, path: str, line_no: int) -> str:
            assert path == source_path
            return f"line {line_no}"

    traced = KediPythonOrigin(
        filename="<kedi-python:block:1>",
        kind="block",
        code="return 1",
        span=span,
        wrapper_line_offset=2,
        kedi_line_offset=1,
        procedure_name="boom",
    ).to_trace(5, source_map=SourceMap(), python_col=7)

    assert traced.python_lineno == 3
    assert traced.kedi_lineno == 6
    assert traced.source_snippet == "line 6"
    assert traced.procedure_name == "boom"
    assert traced.python_col == 7

    unspanned = KediPythonOrigin(
        filename="<kedi-python:inline:1>",
        kind="inline",
        code="1 / 0",
        span=None,
        wrapper_line_offset=5,
    ).to_trace(2)

    assert unspanned.python_lineno == 1
    assert unspanned.kedi_lineno is None
    assert unspanned.source_snippet is None

    spanned_without_map = KediPythonOrigin(
        filename="<kedi-python:inline:2>",
        kind="inline",
        code="name",
        span=span,
    ).to_trace(1)

    assert spanned_without_map.kedi_lineno == 3
    assert spanned_without_map.source_snippet is None


def test_render_execution_error_covers_empty_frames_keyerror_and_multitrace_styles() -> None:
    key_error = KediExecutionError("missing", original=KeyError("missing"))
    key_output = render_execution_error(key_error, use_color=False)
    assert "Kedi stack trace:" in key_output
    assert "Kedi error: KeyError" in key_output
    assert "missing" in key_output
    assert key_error.render() == key_output
    assert str(key_error) == key_output
    assert repr(key_error) == key_output

    span = Span(0, 22, 7, 1, 7, 23, source_path="/tmp/sample.kedi")
    frame = KediTraceFrame(
        kind="python_block",
        label="python block",
        span=span,
        source_snippet="[x: int] = ```",
    )
    traces = (
        KediPythonTrace(
            filename="<kedi-python:block:1>",
            kind="block",
            code="def call():\n    return 1",
            span=span,
            traceback_lineno=1,
            python_lineno=1,
            kedi_lineno=7,
        ),
        KediPythonTrace(
            filename="<kedi-python:block:1>",
            kind="block",
            code="value = 1 # comment\nreturn value",
            span=span,
            traceback_lineno=2,
            python_lineno=1,
            python_col=12,
            kedi_lineno=7,
        ),
    )
    output = render_execution_error(
        KediExecutionError(
            "division",
            frames=(
                KediTraceFrame(kind="program", label="top-level program"),
                frame,
            ),
            python_traces=traces,
            original=ZeroDivisionError("division by zero"),
        ),
        use_color=False,
    )

    assert "in top-level program" in output
    assert "embedded Python call, line 1" in output
    assert "embedded Python error, line 1" in output
    assert "value = 1 # comment" in output
    assert "Python block error: ZeroDivisionError" in output
    assert _role_style(RenderStyle.MONO_LIGHT, "header") == "bold"


def test_render_execution_error_cleans_syntax_and_fallback_details() -> None:
    syntax = SyntaxError("unterminated string literal (detected at line 3)")
    output = render_execution_error(
        KediExecutionError("fallback message", original=syntax),
        use_color=False,
    )
    assert "detected at line" not in output
    assert "unterminated string literal" in output

    class EmptyError(Exception):
        def __str__(self) -> str:
            return ""

    fallback = render_execution_error(
        KediExecutionError("fallback message", original=EmptyError()),
        use_color=False,
    )
    assert "fallback message" in fallback


def test_render_execution_error_limits_embedded_python_context() -> None:
    code = "\n".join(f"line_{index} = {index}" for index in range(1, 41))
    span = Span(0, 5, 1, 1, 1, 6, source_path="/tmp/long_python.kedi")
    output = render_execution_error(
        KediExecutionError(
            "boom",
            frames=(
                KediTraceFrame(
                    kind="python_block",
                    label="python block",
                    span=span,
                    source_snippet="= ```",
                ),
            ),
            python_traces=(
                KediPythonTrace(
                    filename="<kedi-python:block:1>",
                    kind="block",
                    code=code,
                    span=span,
                    traceback_lineno=20,
                    python_lineno=20,
                    kedi_lineno=1,
                ),
            ),
            original=RuntimeError("boom"),
        ),
        use_color=False,
    )

    assert "line_18 = 18" in output
    assert "line_20 = 20" in output
    assert "line_25 = 25" in output
    assert "line_17 = 17" not in output
    assert "line_26 = 26" not in output


def test_render_parse_error_handles_multiple_diagnostics_and_linecache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.kedi"
    source.write_text("@boom(\n= nope\n", encoding="utf-8")
    diagnostics = [
        KediDiagnostic(
            severity="error",
            message="first error",
            span=Span(0, 5, 1, 1, 1, 6, source_path=str(source)),
        ),
        KediDiagnostic(
            severity="error",
            message="second error",
            span=Span(6, 7, 2, 1, 2, 2),
        ),
    ]

    output = render_parse_error(KediParseError(diagnostics), use_color=False)

    assert output.count("Kedi parse error:") == 2
    assert "Kedi error: ParseError" in output
    assert "first error" in output
    assert "second error" in output
    assert "@boom(" in output
    assert 'File "<source>", line 2, column 1' in output

    snippet_output = render_parse_error(
        KediParseError(
            [
                KediDiagnostic(
                    severity="error",
                    message="snippet error",
                    span=Span(0, 3, 1, 1, 1, 4),
                    snippet="# comment",
                )
            ]
        ),
        use_color=False,
    )
    assert "# comment" in snippet_output


def test_error_renderer_helper_edges_cover_carets_matching_and_empty_details() -> None:
    output = render_execution_error(
        KediExecutionError(
            "",
            frames=(
                KediTraceFrame(
                    kind="template",
                    label="template line",
                    span=Span(0, 7, 1, 1, 1, 8, source_path="/tmp/no_snippet.kedi"),
                ),
            ),
            python_traces=(
                KediPythonTrace(
                    filename="<kedi-python:unknown>",
                    kind="unknown",
                    code="x = 1",
                    span=Span(10, 17, 2, 1, 2, 8, source_path="/tmp/other.kedi"),
                    traceback_lineno=1,
                    python_lineno=1,
                ),
            ),
            original=None,
        ),
        use_color=False,
    )
    assert "Python error: RuntimeError" in output

    no_detail_output = render_execution_error(KediExecutionError(""), use_color=False)
    assert "Kedi error: RuntimeError" in no_detail_output

    inline_span = Span(0, 18, 1, 1, 1, 19, source_path="/tmp/inline.kedi")
    inline_trace = KediPythonTrace(
        filename="<kedi-python:inline>",
        kind="inline",
        code="1 / 0",
        span=inline_span,
        traceback_lineno=1,
        python_lineno=1,
        kedi_lineno=1,
    )
    inline_output = render_execution_error(
        KediExecutionError(
            "division",
            frames=(
                KediTraceFrame(
                    kind="python_inline",
                    label="inline python",
                    span=inline_span,
                    source_snippet="= <`1 / 0`>",
                ),
            ),
            python_traces=(inline_trace,),
            original=ZeroDivisionError("division by zero"),
        ),
        use_color=False,
    )
    assert "= <`1 / 0`>" in inline_output
    assert "embedded Python line 1" in inline_output

    same_span_frame = KediTraceFrame(
        kind="template",
        label="template line",
        span=inline_span,
    )
    assert _best_python_trace_frame_index((same_span_frame,), inline_trace) == 0
    assert (
        _best_python_trace_frame_index(
            (KediTraceFrame(kind="template", label="template line", span=None),),
            inline_trace,
        )
        is None
    )

    multiline_span = Span(0, 10, 1, 1, 2, 4, source_path="/tmp/multiline.kedi")
    assert _span_caret(multiline_span, "`x`") is None
    assert _inline_python_source_caret(multiline_span, "`x`") is None
    assert _inline_python_source_caret(Span(0, 5, 1, 1, 1, 6), "value") == (0, 5)
    assert _inline_python_source_caret(Span(0, 5, 1, 1, 1, 6), "`value") == (0, 5)
    assert _inline_python_source_caret(Span(0, 2, 1, 1, 1, 3), "``") == (0, 2)
    assert _column_caret("abc", 0, None) is None
    assert _line_at("one\nTwo", 3) == ""

    highlighted_output = render_parse_error(
        KediParseError(
            [
                KediDiagnostic(
                    severity="error",
                    message="type declaration",
                    span=Span(0, 7, 1, 1, 1, 8),
                    snippet="~Person",
                )
            ]
        ),
        use_color=False,
    )
    assert "~Person" in highlighted_output
