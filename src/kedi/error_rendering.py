from __future__ import annotations

from enum import StrEnum
from io import StringIO
import linecache

from rich.console import Console
from rich.syntax import Syntax

from .errors import KediExecutionError, KediTraceFrame
from .lang.parser.diagnostics import KediDiagnostic, KediParseError


class RenderStyle(StrEnum):
    COLOR = "color"
    GRAYSCALE = "grayscale"
    MONO_DARK = "mono_dark"
    MONO_LIGHT = "mono_light"


def render_execution_error(
    error: KediExecutionError,
    *,
    use_color: bool,
) -> str:
    style = _default_style(use_color)
    console = _console(style)
    console.print("Kedi stack trace:")
    python_traces_by_frame = _python_trace_targets(error)
    for index, frame in enumerate(error.frames):
        traces = python_traces_by_frame.get(index, ())
        _render_trace_frame(console, frame, trace=traces[-1] if traces else None)
        for trace in traces:
            path = trace.span.source_path if trace.span is not None else "<source>"
            kedi_loc = f"{path}:{trace.kedi_lineno}" if trace.kedi_lineno is not None else path
            console.print(
                f"  embedded python line {trace.python_lineno} at {kedi_loc}",
                style="italic",
            )
            console.print(
                Syntax(
                    trace.code,
                    "python",
                    theme=_theme_name(style),
                    line_numbers=True,
                    word_wrap=False,
                    highlight_lines={trace.python_lineno},
                )
            )
    console.print(error.message)
    return console.file.getvalue()


def render_parse_error(error: KediParseError, *, use_color: bool) -> str:
    style = _default_style(use_color)
    console = _console(style)
    for diagnostic in error.diagnostics:
        span = diagnostic.span
        path = span.source_path or "<source>"
        console.print(f"{path}:{span.start_line}:{span.start_col}: error: {diagnostic.message}")
        snippet = _snippet_for_diagnostic(diagnostic)
        if snippet:
            console.print(
                Syntax(
                    snippet,
                    "text",
                    theme=_theme_name(style),
                    line_numbers=True,
                    start_line=span.start_line,
                    word_wrap=False,
                    highlight_lines={span.start_line},
                )
            )
    return console.file.getvalue()


def _render_trace_frame(console: Console, frame: KediTraceFrame, *, trace=None) -> None:
    span = frame.span
    if span is None:
        console.print(f"  in {frame.label}")
        return
    line_no = (
        trace.kedi_lineno
        if trace is not None and trace.kedi_lineno is not None
        else span.start_line
    )
    snippet = None if trace is not None else frame.source_snippet
    path = span.source_path or "<source>"
    console.print(f'  File "{path}", line {line_no}, in {frame.label}')
    if snippet:
        console.print(
            Syntax(
                snippet,
                "text",
                theme=_theme_name(_style_from_console(console)),
                line_numbers=True,
                start_line=line_no,
                word_wrap=False,
                highlight_lines={line_no},
            )
        )


def _python_trace_targets(error: KediExecutionError) -> dict[int, tuple]:
    targets: dict[int, list] = {}
    for trace in error.python_traces:
        index = _best_python_trace_frame_index(error.frames, trace)
        if index is None:
            continue
        targets.setdefault(index, []).append(trace)
    return {index: tuple(traces) for index, traces in targets.items()}


def _best_python_trace_frame_index(frames: tuple[KediTraceFrame, ...], trace) -> int | None:
    preferred_kinds = {
        "inline": {"python_inline"},
        "block": {"python_block"},
        "side_effects": {"python_side_effects"},
        "prelude": {"prelude"},
        "type_expression": {"type_expression"},
    }.get(trace.kind, set())

    for index in range(len(frames) - 1, -1, -1):
        frame = frames[index]
        if frame.kind in preferred_kinds and _same_span(frame, trace):
            return index

    for index in range(len(frames) - 1, -1, -1):
        if _same_span(frames[index], trace):
            return index
    return None


def _same_span(frame: KediTraceFrame, trace) -> bool:
    if frame.span is None or trace.span is None:
        return False
    return (
        trace.span.source_path == frame.span.source_path
        and trace.span.start_line == frame.span.start_line
        and trace.span.start_col == frame.span.start_col
    )


def _snippet_for_diagnostic(diagnostic: KediDiagnostic) -> str | None:
    if diagnostic.snippet:
        return diagnostic.snippet
    path = diagnostic.span.source_path
    if path is None:
        return None
    line = linecache.getline(path, diagnostic.span.start_line)
    return line.rstrip("\n") or None


def _console(style: RenderStyle) -> Console:
    color_enabled = style in {RenderStyle.COLOR, RenderStyle.GRAYSCALE}
    return Console(
        file=StringIO(),
        force_terminal=color_enabled,
        color_system="truecolor" if color_enabled else None,
        width=240,
        soft_wrap=False,
    )


def _theme_name(style: RenderStyle) -> str:
    if style == RenderStyle.COLOR:
        return "ansi_dark"
    if style == RenderStyle.GRAYSCALE:
        return "github-dark"
    if style == RenderStyle.MONO_LIGHT:
        return "bw"
    return "bw"


def _default_style(use_color: bool) -> RenderStyle:
    return RenderStyle.COLOR if use_color else RenderStyle.MONO_DARK


def _style_from_console(console: Console) -> RenderStyle:
    if console.color_system is None:
        return RenderStyle.MONO_DARK
    return RenderStyle.COLOR
