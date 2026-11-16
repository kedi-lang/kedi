"""Static diagnostics for media values (``> import: media``).

A model prompt passes media by reference, ``<<image>>``; the editor checks references against the
model's capabilities. ``<image>`` as text is reported by :mod:`.substitution_types`. Inference is deliberately simple: a binding holds media when
it is declared with a media type; a call returns media when it is a media loader or a procedure
declared to return a media type.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Union

from kedi.lang.ast import (
    CallReference,
    InputReference,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    Segment,
    Span,
    VariableInitStmt,
    reference_span,
)
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
    """Spans in a model prompt that attach media: ``<<image>>`` and ``<<image(...)>>``."""

    return [
        reference_span(segment)
        for segment in segments
        if (isinstance(segment, InputReference) and segment.name in names.bindings)
        or (isinstance(segment, CallReference) and segment.name in names.calls)
    ]


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
]
