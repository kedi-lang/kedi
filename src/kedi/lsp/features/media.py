"""Static diagnostics for media values (``> import: media``).

``<x>`` renders text and media has no text form, so ``<image>`` fails at runtime; a model prompt
passes media by reference, ``<<image>>``. The editor reports text uses and checks references
against the model's capabilities. Inference is deliberately simple: a binding holds media when
it is declared with a media type; a call returns media when it is a media loader or a procedure
declared to return a media type.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    REFERENCE_SEGMENTS,
    CallReference,
    CallSegment,
    InputReference,
    InputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    Segment,
    Span,
    SystemDirectiveStmt,
    VariableInitStmt,
    reference_span,
)
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lsp.coordinates import span_to_range
from kedi.lsp.features.scope import _iter_control_statements, _procedure_scope_entries, _segments_of

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
    """Spans in a model prompt that attach media: ``<<image>>`` and ``<<image(...)>>``."""

    return [
        reference_span(segment)
        for segment in segments
        if (isinstance(segment, InputReference) and segment.name in names.bindings)
        or (isinstance(segment, CallReference) and segment.name in names.calls)
    ]


def media_text_diagnostics(prog: AnyProgram) -> list[lsp.Diagnostic]:
    """Report ``<media>`` substitutions: ``<>`` renders text, and media has no text form."""

    if not _imports_media(prog):
        return []
    calls = _media_calls(prog)
    out: list[lsp.Diagnostic] = []
    for statements, proc in _scopes(prog):
        media = _scope_media_names(statements, proc, calls)
        for stmt in _iter_control_statements(statements):
            segments = list(_segments_of(stmt))
            if isinstance(stmt, SystemDirectiveStmt):
                segments = [segment for line in stmt.template_lines for segment in line.segments]
            for segment in _nested(segments):
                diagnostic = _text_use(segment, media, calls)
                if diagnostic is not None:
                    out.append(diagnostic)
    return out


def _text_use(segment: object, media: set[str], calls: frozenset[str]) -> lsp.Diagnostic | None:
    if isinstance(segment, REFERENCE_SEGMENTS):
        return None
    if isinstance(segment, CallSegment) and segment.name in calls:
        subject = f"{segment.name}(...)"
    elif isinstance(segment, InputSegment) and segment.name in media:
        subject = segment.name
    else:
        return None
    assert segment.loc is not None  # parsed segments always carry a span
    return lsp.Diagnostic(
        range=span_to_range(segment.loc),
        severity=lsp.DiagnosticSeverity.Error,
        code="K-SUBSTITUTION-TYPE",
        source="kedi",
        message=(
            f"<{subject}> renders text, but {subject} is media. In a model prompt pass it by "
            f"reference with <<{subject}>>; to keep the value use single-backtick Python."
        ),
    )


def _nested(segments: Sequence[object]) -> Iterator[object]:
    for segment in segments:
        yield segment
        if isinstance(segment, CallSegment):
            for argument in segment.args:
                yield from _nested(list(iter_call_arg_nested(argument)))


def _scope_media_names(
    statements: Sequence[object], proc: ProcedureDef | None, calls: frozenset[str]
) -> set[str]:
    names: set[str] = set()
    if proc is not None:
        for name, annotation in (proc.param_type_annotations or {}).items():
            if annotation and _MEDIA_TYPE.fullmatch(annotation):
                names.add(name)
    for stmt in _iter_control_statements(statements):
        if isinstance(stmt, VariableInitStmt) and stmt.type_annotation is not None:
            if _MEDIA_TYPE.fullmatch(stmt.type_annotation):
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
