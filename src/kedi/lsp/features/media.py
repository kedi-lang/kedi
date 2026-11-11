"""Static diagnostics for media values (``> import: media``).

A media value reaches a model only as a prompt attachment. Rendered into text anywhere else
it would fail at runtime, so the editor reports those uses. Inference is deliberately
simple: a binding holds media when it is declared with a media type, or initialized from a
media loader or from a procedure that returns a media type.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AssignmentStmt,
    CallSegment,
    InputSegment,
    LiteralArg,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    PythonExprNativeSegment,
    RawInvokeStmt,
    ReturnStmt,
    Segment,
    SendStmt,
    Span,
    SystemDirectiveStmt,
    TaskStmt,
    TemplateBlock,
    TextSegment,
    VariableInitStmt,
)
from kedi.lsp.coordinates import span_to_range
from kedi.lsp.features.scope import _iter_control_statements, _procedure_scope_entries

MEDIA_LOADERS = frozenset({"image", "audio", "video", "document"})
_MEDIA_TYPE = re.compile(r"\s*(?:list\[\s*)?(?:Image|Audio|Video|Document|Media)\s*\]?\s*")

AnyProgram = Union[Program, ProgramWithValidation]


@dataclass(frozen=True, slots=True)
class MediaNames:
    """Names that statically produce media: bindings in any scope, and procedures."""

    bindings: frozenset[str] = frozenset()
    calls: frozenset[str] = frozenset()


def media_names(prog: AnyProgram) -> MediaNames:
    """Collect a program's media names; empty unless it imports the ``media`` module."""

    if not _imports_media(prog):
        return MediaNames()
    calls = _media_calls(prog)
    bindings: set[str] = set()
    for statements, proc in _scopes(prog):
        bindings.update(_scope_media_names(statements, proc, calls))
    return MediaNames(bindings=frozenset(bindings), calls=calls)


def media_prompt_spans(segments: Sequence[Segment], names: MediaNames) -> list[Span | None]:
    """Spans in a model prompt that attach media."""

    return [
        segment.loc
        for segment in segments
        if (isinstance(segment, InputSegment) and segment.name in names.bindings)
        or (isinstance(segment, CallSegment) and segment.name in names.calls)
    ]


def media_text_diagnostics(prog: AnyProgram) -> list[lsp.Diagnostic]:
    """Report media rendered into text, where it would fail at runtime."""

    if not _imports_media(prog):
        return []
    calls = _media_calls(prog)
    out: list[lsp.Diagnostic] = []
    for statements, proc in _scopes(prog):
        media = _scope_media_names(statements, proc, calls)
        for stmt in _iter_control_statements(statements):
            out.extend(_statement_diagnostics(stmt, media))
    return out


def _statement_diagnostics(stmt: object, media: set[str]) -> Iterator[lsp.Diagnostic]:
    if isinstance(stmt, (VariableInitStmt, AssignmentStmt)):
        if not _native_rhs(stmt.expr):
            yield from _text_uses(stmt.expr, media)
        yield from _argument_uses(stmt.expr, media)
    elif isinstance(stmt, ReturnStmt):
        if not _native_rhs(stmt.segments):
            yield from _text_uses(stmt.segments, media)
        yield from _argument_uses(stmt.segments, media)
    elif isinstance(stmt, SystemDirectiveStmt):
        for line in stmt.template_lines:
            yield from _text_uses(
                line.segments,
                media,
                reason="instructions are text only; mention the media in a >> prompt instead",
            )
    elif isinstance(stmt, (TaskStmt, SendStmt)):
        for line in stmt.template.lines:
            yield from _text_uses(
                line.segments, media, reason="subagent tasks do not take media yet"
            )
    elif isinstance(stmt, TemplateBlock):
        for line in stmt.lines:
            yield from _argument_uses(line.segments, media)
    elif isinstance(stmt, RawInvokeStmt):
        yield from _argument_uses(stmt.segments, media)


def _text_uses(
    segments: Sequence[object], media: set[str], *, reason: str | None = None
) -> Iterator[lsp.Diagnostic]:
    for segment in segments:
        if isinstance(segment, InputSegment) and segment.name in media and segment.loc:
            detail = reason or f"to pass it as a value use backticks: `{segment.name}`"
            yield lsp.Diagnostic(
                range=span_to_range(segment.loc),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-MEDIA-TEXT",
                source="kedi",
                message=(
                    f"Media value '{segment.name}' cannot be embedded in text: use it directly "
                    f"in a model prompt; {detail}."
                ),
            )


def _argument_uses(segments: Sequence[object], media: set[str]) -> Iterator[lsp.Diagnostic]:
    for segment in segments:
        if not isinstance(segment, CallSegment):
            continue
        for argument in segment.args:
            if isinstance(argument, LiteralArg) and argument.render_segments:
                yield from _text_uses(argument.render_segments, media)


def _native_rhs(segments: Sequence[object]) -> bool:
    content = [
        segment
        for segment in segments
        if not (isinstance(segment, TextSegment) and not segment.text.strip())
    ]
    return len(content) == 1 and isinstance(content[0], (CallSegment, PythonExprNativeSegment))


def _scope_media_names(
    statements: Sequence[object], proc: ProcedureDef | None, calls: frozenset[str]
) -> set[str]:
    names: set[str] = set()
    if proc is not None:
        for name, annotation in (proc.param_type_annotations or {}).items():
            if annotation and _MEDIA_TYPE.fullmatch(annotation):
                names.add(name)
    for stmt in _iter_control_statements(statements):
        if not isinstance(stmt, VariableInitStmt):
            continue
        if stmt.type_annotation is not None:
            if _MEDIA_TYPE.fullmatch(stmt.type_annotation):
                names.add(stmt.name)
            continue
        content = [
            segment
            for segment in stmt.expr
            if not (isinstance(segment, TextSegment) and not segment.text.strip())
        ]
        if len(content) == 1 and isinstance(content[0], CallSegment):
            if content[0].name in calls:
                names.add(stmt.name)
    return names


def _imports_media(prog: AnyProgram) -> bool:
    return any(module_import.module == "media" for module_import in prog.imports)


def _media_calls(prog: AnyProgram) -> frozenset[str]:
    """Imported media loaders, and procedures declared to return a media type."""
    calls: set[str] = set()
    for module_import in prog.imports:
        if module_import.module == "media":
            calls.update(MEDIA_LOADERS & set(module_import.names or MEDIA_LOADERS))
    for proc, _ in _procedure_scope_entries(prog):
        if proc.return_type_annotation and _MEDIA_TYPE.fullmatch(proc.return_type_annotation):
            calls.add(proc.name)
    return frozenset(calls)


def _scopes(prog: AnyProgram) -> Iterator[tuple[Sequence[object], ProcedureDef | None]]:
    yield prog.toplevel_body, None
    for proc, _ in _procedure_scope_entries(prog):
        yield proc.body, proc


__all__ = [
    "MEDIA_LOADERS",
    "MediaNames",
    "media_names",
    "media_prompt_spans",
    "media_text_diagnostics",
]
