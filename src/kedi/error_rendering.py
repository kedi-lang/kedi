from __future__ import annotations

import linecache
import re
from enum import Enum
from io import StringIO
from typing import cast

from rich.console import Console
from rich.text import Text

from .errors import KediExecutionError, KediPythonTrace, KediTraceFrame
from .lang.ast import Span
from .lang.parser.diagnostics import KediDiagnostic, KediParseError


class RenderStyle(str, Enum):
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
    console.print(Text("Kedi stack trace:", style=_role_style(style, "header")))

    python_traces_by_frame = _python_trace_targets(error)
    previous_substantial = False

    for index, frame in enumerate(error.frames):
        traces = python_traces_by_frame.get(index, ())
        substantial = _frame_has_body(frame, traces)
        if index > 0 and (previous_substantial or substantial):
            console.print()

        next_frame = error.frames[index + 1] if index + 1 < len(error.frames) else None
        previous_frame = error.frames[index - 1] if index > 0 else None
        caret = _frame_caret(frame, next_frame)
        _render_trace_frame(
            console,
            frame,
            style=style,
            trace=traces[-1] if traces else None,
            caret=caret,
            render_trace_source=_should_render_trace_source(
                frame, traces[-1] if traces else None, previous_frame
            ),
        )
        for trace_index, trace in enumerate(traces):
            if trace_index > 0:
                console.print()
            console.print(
                Text(
                    _embedded_python_label(trace, trace_index=trace_index, trace_count=len(traces)),
                    style=_role_style(style, "python_label"),
                )
            )
            _render_python_snippet(
                console,
                trace.code,
                style=style,
                highlight_line=trace.python_lineno,
                indent=4,
                caret=_python_trace_caret(trace),
            )
        previous_substantial = substantial

    if error.frames:
        console.print()
    _render_error_footer(console, error, style)
    return cast(StringIO, console.file).getvalue()


def render_parse_error(error: KediParseError, *, use_color: bool) -> str:
    style = _default_style(use_color)
    console = _console(style)
    for index, diagnostic in enumerate(error.diagnostics):
        if index > 0:
            console.print()
        _render_parse_diagnostic(console, diagnostic, style)
    return cast(StringIO, console.file).getvalue()


def _render_parse_diagnostic(
    console: Console, diagnostic: KediDiagnostic, style: RenderStyle
) -> None:
    span = diagnostic.span
    path = span.source_path or "<source>"
    console.print(
        Text(
            f"Kedi parse error: {diagnostic.message}",
            style=_role_style(style, "error_title"),
        )
    )
    console.print()
    console.print(_file_line_text(path, span.start_line, f"column {span.start_col}", style))
    snippet = _snippet_for_diagnostic(diagnostic)
    if snippet:
        _render_kedi_snippet(
            console,
            snippet,
            line_no=span.start_line,
            style=style,
            caret=_span_caret(span, snippet),
        )


def _render_trace_frame(
    console: Console,
    frame: KediTraceFrame,
    *,
    style: RenderStyle,
    trace: KediPythonTrace | None,
    caret: tuple[int, int] | None,
    render_trace_source: bool,
) -> None:
    span = frame.span
    if span is None:
        console.print(
            Text.assemble(
                ("  in ", _role_style(style, "muted")),
                (frame.label, _role_style(style, "frame_label")),
            )
        )
        return

    line_no = (
        trace.kedi_lineno
        if trace is not None and trace.kedi_lineno is not None
        else span.start_line
    )
    suffix = f"in {frame.label}"
    console.print(_file_line_text(span.source_path or "<source>", line_no, suffix, style))

    if trace is not None and not render_trace_source:
        return
    if frame.source_snippet:
        rendered_caret = caret
        if trace is not None and frame.span is not None:
            rendered_caret = _inline_python_source_caret(frame.span, frame.source_snippet)
        _render_kedi_snippet(
            console,
            frame.source_snippet,
            line_no=line_no,
            style=style,
            caret=rendered_caret,
        )


def _render_error_footer(
    console: Console,
    error: KediExecutionError,
    style: RenderStyle,
) -> None:
    title = _error_title(error)
    detail = _error_detail(error)
    console.print(Text(title, style=_role_style(style, "error_title")))
    if detail:
        console.print(Text(f"  {detail}", style=_role_style(style, "error_detail")))


def _error_title(error: KediExecutionError) -> str:
    exc = error.original
    exc_name = type(exc).__name__ if exc is not None else "RuntimeError"
    if error.python_traces:
        prefix = {
            "inline": "Inline Python error",
            "block": "Python block error",
            "side_effects": "Inline Python error",
            "prelude": "Python prelude error",
            "type_expression": "Python type expression error",
        }.get(error.python_traces[-1].kind, "Python error")
        return f"{prefix}: {exc_name}"
    return f"Kedi error: {exc_name}"


def _error_detail(error: KediExecutionError) -> str | None:
    exc = error.original
    if exc is None:
        return error.message or None
    if isinstance(exc, SyntaxError):
        return _clean_syntax_error_message(exc.msg) or None
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    detail = str(exc).strip()
    if detail:
        return detail
    fallback = error.message.strip()
    return fallback if fallback else None


def _clean_syntax_error_message(message: str) -> str:
    return re.sub(r"\s+\(detected at line \d+\)$", "", message).strip()


def _embedded_python_label(
    trace: KediPythonTrace,
    *,
    trace_index: int,
    trace_count: int,
) -> str:
    if trace_count == 1:
        return f"    embedded Python line {trace.python_lineno}"
    role = "error" if trace_index == trace_count - 1 else "call"
    return f"    embedded Python {role}, line {trace.python_lineno}"


def _file_line_text(path: str, line_no: int, suffix: str, style: RenderStyle) -> Text:
    text = Text("  ")
    text.append('File "', style=_role_style(style, "muted"))
    text.append(path, style=_role_style(style, "path"))
    text.append('"', style=_role_style(style, "muted"))
    text.append(", line ", style=_role_style(style, "muted"))
    text.append(str(line_no), style=_role_style(style, "line_number"))
    text.append(", ", style=_role_style(style, "muted"))
    text.append(suffix, style=_role_style(style, "frame_label"))
    return text


def _render_kedi_snippet(
    console: Console,
    snippet: str,
    *,
    line_no: int,
    style: RenderStyle,
    caret: tuple[int, int] | None = None,
) -> None:
    rendered = _highlight_kedi_line(_normalize_line(snippet), style)
    _render_numbered_line(console, rendered, line_no=line_no, indent=2, style=style, active=True)
    if caret is not None:
        _render_caret_line(console, indent=2, line_no=line_no, style=style, caret=caret)


def _render_python_snippet(
    console: Console,
    code: str,
    *,
    style: RenderStyle,
    highlight_line: int,
    indent: int,
    caret: tuple[int, int] | None = None,
) -> None:
    lines = code.splitlines() or [""]
    width = len(str(max(len(lines), highlight_line)))
    for line_no, line in enumerate(lines, start=1):
        rendered = _highlight_python_line(_normalize_line(line), style)
        _render_numbered_line(
            console,
            rendered,
            line_no=line_no,
            indent=indent,
            width=width,
            style=style,
            active=line_no == highlight_line,
        )
        if line_no == highlight_line and caret is not None:
            _render_caret_line(
                console,
                indent=indent,
                line_no=line_no,
                style=style,
                caret=caret,
                width=width,
            )


def _render_numbered_line(
    console: Console,
    rendered: Text,
    *,
    line_no: int,
    indent: int,
    style: RenderStyle,
    active: bool,
    width: int | None = None,
) -> None:
    number_width = width or len(str(line_no))
    line = _line_prefix(line_no, indent=indent, active=active, width=number_width, style=style)
    line.append_text(rendered)
    console.print(line)


def _line_prefix(
    line_no: int,
    *,
    indent: int,
    active: bool,
    width: int,
    style: RenderStyle,
) -> Text:
    prefix = Text(" " * indent)
    marker = "❱ " if active else "  "
    prefix.append(marker, style=_role_style(style, "marker" if active else "muted"))
    prefix.append(str(line_no).rjust(width), style=_role_style(style, "line_number"))
    prefix.append(" ", style=_role_style(style, "muted"))
    return prefix


def _render_caret_line(
    console: Console,
    *,
    indent: int,
    line_no: int,
    style: RenderStyle,
    caret: tuple[int, int],
    width: int | None = None,
) -> None:
    start, length = caret
    number_width = width or len(str(line_no))
    prefix_length = indent + 2 + number_width + 1
    caret_line = Text(" " * prefix_length)
    caret_line.append(" " * max(start, 0))
    caret_line.append("^" * max(length, 1), style=_role_style(style, "caret"))
    console.print(caret_line)


def _python_trace_targets(
    error: KediExecutionError,
) -> dict[int, tuple[KediPythonTrace, ...]]:
    targets: dict[int, list[KediPythonTrace]] = {}
    for trace in error.python_traces:
        index = _best_python_trace_frame_index(error.frames, trace)
        if index is None:
            continue
        targets.setdefault(index, []).append(trace)
    return {index: tuple(traces) for index, traces in targets.items()}


def _best_python_trace_frame_index(
    frames: tuple[KediTraceFrame, ...],
    trace: KediPythonTrace,
) -> int | None:
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


def _same_span(frame: KediTraceFrame, trace: KediPythonTrace) -> bool:
    if frame.span is None or trace.span is None:
        return False
    return (
        trace.span.source_path == frame.span.source_path
        and trace.span.start_line == frame.span.start_line
        and trace.span.start_col == frame.span.start_col
    )


def _frame_has_body(frame: KediTraceFrame, traces: tuple[KediPythonTrace, ...]) -> bool:
    return frame.source_snippet is not None or bool(traces)


def _frame_caret(
    frame: KediTraceFrame,
    next_frame: KediTraceFrame | None,
) -> tuple[int, int] | None:
    if (
        frame.span is None
        or frame.source_snippet is None
        or next_frame is None
        or next_frame.kind != "python_inline"
        or next_frame.span is None
        or not _same_source_line(frame.span, next_frame.span)
    ):
        return None
    return _inline_python_source_caret(next_frame.span, frame.source_snippet)


def _should_render_trace_source(
    frame: KediTraceFrame,
    trace: KediPythonTrace | None,
    previous_frame: KediTraceFrame | None,
) -> bool:
    if (
        trace is None
        or frame.kind != "python_inline"
        or frame.source_snippet is None
        or frame.span is None
        or trace.kedi_lineno != frame.span.start_line
    ):
        return False
    if previous_frame is None or previous_frame.span is None:
        return True
    return not _same_source_line(previous_frame.span, frame.span)


def _python_trace_caret(trace: KediPythonTrace) -> tuple[int, int] | None:
    if trace.python_col is None:
        return None
    code_line = _line_at(trace.code, trace.python_lineno)
    return _column_caret(code_line, trace.python_col, trace.python_col)


def _span_caret(span: Span, snippet: str) -> tuple[int, int] | None:
    if span.start_line != span.end_line:
        return None
    return _column_caret(snippet, span.start_col, span.end_col)


def _inline_python_source_caret(span: Span, snippet: str) -> tuple[int, int] | None:
    if span.start_line != span.end_line:
        return None
    start = max(span.start_col - 1, 0)
    end = min(max(span.end_col - 1, start), len(snippet))
    segment = snippet[start:end]
    open_tick = segment.find("`")
    if open_tick < 0:
        return _span_caret(span, snippet)
    close_tick = segment.find("`", open_tick + 1)
    if close_tick < 0:
        return _span_caret(span, snippet)
    content_start = start + open_tick + 1
    content_end = start + close_tick
    if content_end <= content_start:
        return _span_caret(span, snippet)
    return _column_caret(snippet, content_start + 1, content_end + 1)


def _column_caret(line: str, start_col: int, end_col: int | None) -> tuple[int, int] | None:
    if start_col <= 0:
        return None
    start = _visual_offset(line, start_col)
    if end_col is None or end_col <= start_col:
        return start, 1
    end = _visual_offset(line, end_col)
    return start, max(end - start, 1)


def _visual_offset(line: str, col_1_based: int) -> int:
    raw_prefix = line[: max(col_1_based - 1, 0)]
    return len(raw_prefix.expandtabs(4))


def _same_source_line(left: Span, right: Span) -> bool:
    return left.source_path == right.source_path and left.start_line == right.start_line


def _line_at(text: str, line_no: int) -> str:
    lines = text.splitlines() or [""]
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _normalize_line(line: str) -> str:
    return line.expandtabs(4)


def _highlight_kedi_line(line: str, style: RenderStyle) -> Text:
    text = Text(line, style=_role_style(style, "source"))
    stripped = line.lstrip()
    if stripped.startswith("#"):
        text.stylize(_role_style(style, "comment"), 0, len(line))
        return text

    for match in re.finditer(r"@\w+", line):
        text.stylize(_role_style(style, "procedure"), match.start(), match.end())
    for match in re.finditer(r"~\w+", line):
        text.stylize(_role_style(style, "type_decl"), match.start(), match.end())
    for match in re.finditer(r"\[[^\]\n]+\]", line):
        text.stylize(_role_style(style, "placeholder"), match.start(), match.end())
    for match in re.finditer(r"<[^>\n]+>", line):
        text.stylize(_role_style(style, "substitution"), match.start(), match.end())
    for match in re.finditer(r"`[^`\n]*`|```", line):
        text.stylize(_role_style(style, "backtick"), match.start(), match.end())
    return text


def _highlight_python_line(line: str, style: RenderStyle) -> Text:
    text = Text(line, style=_role_style(style, "source"))
    stripped = line.lstrip()
    if stripped.startswith("#"):
        text.stylize(_role_style(style, "comment"), 0, len(line))
        return text

    for match in re.finditer(
        r"\b(def|return|if|else|elif|for|while|try|except|import|from|raise|class|with|as|print)\b",
        line,
    ):
        text.stylize(_role_style(style, "python_keyword"), match.start(), match.end())
    for match in re.finditer(
        r"""(?x)
        f"[^"\n]*"
        |f'[^'\n]*'
        |"[^"\n]*"
        |'[^'\n]*'
    """,
        line,
    ):
        text.stylize(_role_style(style, "python_string"), match.start(), match.end())
    comment_index = line.find("#")
    if comment_index >= 0:
        text.stylize(_role_style(style, "comment"), comment_index, len(line))
    return text


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


def _default_style(use_color: bool) -> RenderStyle:
    return RenderStyle.COLOR if use_color else RenderStyle.MONO_DARK


def _role_style(style: RenderStyle, role: str) -> str:
    palettes = {
        RenderStyle.COLOR: {
            "header": "bold bright_cyan",
            "path": "bright_black",
            "muted": "bright_black",
            "line_number": "bright_black",
            "frame_label": "white",
            "marker": "bold bright_cyan",
            "python_label": "italic bright_black",
            "error_title": "bold bright_red",
            "error_detail": "white",
            "caret": "bold bright_red",
            "source": "white",
            "comment": "italic green",
            "procedure": "bold bright_cyan",
            "type_decl": "bold bright_magenta",
            "placeholder": "bold bright_green",
            "substitution": "bold bright_blue",
            "backtick": "yellow",
            "python_keyword": "bold bright_magenta",
            "python_string": "yellow",
        },
        RenderStyle.GRAYSCALE: {
            "header": "bold white",
            "path": "bright_black",
            "muted": "bright_black",
            "line_number": "bright_black",
            "frame_label": "white",
            "marker": "bold white",
            "python_label": "italic bright_black",
            "error_title": "bold white",
            "error_detail": "white",
            "caret": "bold white",
            "source": "white",
            "comment": "italic bright_black",
            "procedure": "bold white",
            "type_decl": "bold white",
            "placeholder": "bold white",
            "substitution": "bold white",
            "backtick": "white",
            "python_keyword": "bold white",
            "python_string": "white",
        },
        RenderStyle.MONO_DARK: {
            "header": "bold",
            "path": "dim",
            "muted": "dim",
            "line_number": "dim",
            "frame_label": "none",
            "marker": "bold",
            "python_label": "italic dim",
            "error_title": "bold",
            "error_detail": "none",
            "caret": "bold",
            "source": "none",
            "comment": "dim",
            "procedure": "bold",
            "type_decl": "bold",
            "placeholder": "bold",
            "substitution": "bold",
            "backtick": "none",
            "python_keyword": "bold",
            "python_string": "none",
        },
        RenderStyle.MONO_LIGHT: {
            "header": "bold",
            "path": "dim",
            "muted": "dim",
            "line_number": "dim",
            "frame_label": "none",
            "marker": "bold",
            "python_label": "italic dim",
            "error_title": "bold",
            "error_detail": "none",
            "caret": "bold",
            "source": "none",
            "comment": "dim",
            "procedure": "bold",
            "type_decl": "bold",
            "placeholder": "bold",
            "substitution": "bold",
            "backtick": "none",
            "python_keyword": "bold",
            "python_string": "none",
        },
    }
    return palettes[style][role]
