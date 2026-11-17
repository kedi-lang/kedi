"""Static ``K-SUBSTITUTION-TYPE`` diagnostics: ``<x>`` of a value declared without a text form.

``<x>`` renders text, and only values with a text form render (:mod:`kedi.text_form`). The editor
reports a substitution when the declaration decides it: a binding or parameter annotated, or a
procedure declared to return, a type that never has a text form, such as media, a Kedi ``~Type``,
``set[...]``, ``bytes``, a path, ``list[Line]`` of a model, or a dict with non-``str`` keys.
``list``, ``dict``, ``Any``, unions with a text member, unknown names and unannotated values are
left to the run-time check.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence

from lsprotocol import types as lsp

from kedi.lang.ast import (
    REFERENCE_SEGMENTS,
    CallSegment,
    InputSegment,
    OutputSegment,
    ProcedureDef,
    RawInvokeStmt,
    SystemDirectiveStmt,
    VariableInitStmt,
)
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lsp.coordinates import span_to_range
from kedi.lsp.features.media import (
    _MEDIA_TYPE,
    MEDIA_LOADERS,
    AnyProgram,
    _imports_media,
    _scopes,
)
from kedi.lsp.features.scope import _iter_control_statements, _procedure_scope_entries, _segments_of

_NO_TEXT_NAMES = frozenset(
    {
        "set",
        "frozenset",
        "Set",
        "FrozenSet",
        "bytes",
        "bytearray",
        "Path",
        "PurePath",
        "PosixPath",
        "PurePosixPath",
        "WindowsPath",
        "timedelta",
        "complex",
    }
)
_MEDIA_NAMES = frozenset({"Image", "Audio", "Video", "Document", "Media"})
_NON_STR_KEYS = frozenset({"int", "float", "bool", "bytes", "tuple", "complex", "frozenset"})
_SEQUENCES = frozenset({"list", "List", "tuple", "Tuple", "Sequence"})
_MAPPINGS = frozenset({"dict", "Dict", "Mapping"})


def substitution_type_diagnostics(prog: AnyProgram) -> list[lsp.Diagnostic]:
    """Report ``<x>`` and ``<f(...)>`` whose declared type has no text form."""

    media = _imports_media(prog)
    no_text = _NO_TEXT_NAMES | {typedef.name for typedef in prog.type_defs}
    if media:
        no_text |= _MEDIA_NAMES
    calls = _calls_without_text(prog, no_text, media)
    out: list[lsp.Diagnostic] = []
    for statements, proc in _scopes(prog):
        bindings = _bindings_without_text(statements, proc, no_text)
        for stmt in _iter_control_statements(statements):
            segments = list(_segments_of(stmt))
            if isinstance(stmt, SystemDirectiveStmt):
                segments = [segment for line in stmt.template_lines for segment in line.segments]
            for segment in _nested(segments):
                diagnostic = _text_use(segment, bindings, calls, media=media)
                if diagnostic is not None:
                    out.append(diagnostic)
    return out


def annotation_has_no_text(annotation: str, no_text: frozenset[str]) -> bool:
    """Whether every value of ``annotation`` lacks a text form; ``False`` when undecided."""

    try:
        node = ast.parse(annotation.strip(), mode="eval").body
    except SyntaxError:
        return False
    return _no_text(node, no_text, nested=False)


def _no_text(node: ast.expr, no_text: frozenset[str], *, nested: bool) -> bool:
    if isinstance(node, ast.Constant):
        # A bare None renders nothing at the top; nested, it is null.
        return node.value is None and not nested
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _name(node) in no_text
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _no_text(node.left, no_text, nested=nested) and _no_text(
            node.right, no_text, nested=nested
        )
    if not isinstance(node, ast.Subscript):
        return False
    base = _name(node.value)
    args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if base in {"set", "frozenset", "Set", "FrozenSet"}:
        return True
    if base in _SEQUENCES:
        return any(
            _no_text(arg, no_text, nested=True)
            for arg in args
            if not (isinstance(arg, ast.Constant) and arg.value is Ellipsis)
        )
    if base in _MAPPINGS and len(args) == 2:
        key, value = args
        return _name(key) in _NON_STR_KEYS | no_text or _no_text(value, no_text, nested=True)
    if base == "Union":
        return all(_no_text(arg, no_text, nested=nested) for arg in args)
    if base == "Optional":
        return not nested and _no_text(args[0], no_text, nested=nested)
    if base == "Annotated":
        return _no_text(args[0], no_text, nested=nested)
    return False


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _calls_without_text(prog: AnyProgram, no_text: frozenset[str], media: bool) -> dict[str, str]:
    """Calls that return a value without a text form, with the declared type."""

    calls: dict[str, str] = {}
    if media:
        for module_import in prog.imports:
            if module_import.module == "media":
                for loader in MEDIA_LOADERS & set(module_import.names or MEDIA_LOADERS):
                    calls[loader] = "Media"
    for proc, _ in _procedure_scope_entries(prog):
        annotation = proc.return_type_annotation
        if annotation and annotation_has_no_text(annotation, no_text):
            calls[proc.name] = annotation
    return calls


def _bindings_without_text(
    statements: Sequence[object], proc: ProcedureDef | None, no_text: frozenset[str]
) -> dict[str, str]:
    """Names whose every declaration in the scope is annotated with a type without text.

    An unannotated declaration, or one whose type may have a text form, leaves the name to the
    run-time check.
    """

    declared: dict[str, list[str]] = {}
    undecided: set[str] = set()
    declarations: list[tuple[str, str | None]] = []
    if proc is not None:
        annotations = proc.param_type_annotations or {}
        declarations = [(name, annotations.get(name)) for name in proc.params]
    declarations += [
        (stmt.name, stmt.type_annotation)
        for stmt in _iter_control_statements(statements)
        if isinstance(stmt, VariableInitStmt)
    ]
    for name, annotation in declarations:
        if annotation is None or not annotation_has_no_text(annotation, no_text):
            undecided.add(name)
        else:
            declared.setdefault(name, []).append(annotation)
    # A template output or a raw invoke rebinds the name to what the model writes.
    for stmt in _iter_control_statements(statements):
        if isinstance(stmt, RawInvokeStmt):
            undecided.add(stmt.name)
        undecided.update(
            segment.name for segment in _segments_of(stmt) if isinstance(segment, OutputSegment)
        )
    return {name: found[-1] for name, found in declared.items() if name not in undecided}


def _only_empty_renders(annotation: str) -> bool:
    """Whether the type is a container: then an empty value still renders (``[]``, ``{}``)."""

    node = ast.parse(annotation.strip(), mode="eval").body
    while isinstance(node, ast.Subscript) and _name(node.value) in {"Annotated", "Optional"}:
        node = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
    return isinstance(node, ast.Subscript) and _name(node.value) in _SEQUENCES | _MAPPINGS


def _text_use(
    segment: object, bindings: dict[str, str], calls: dict[str, str], *, media: bool
) -> lsp.Diagnostic | None:
    if isinstance(segment, REFERENCE_SEGMENTS):
        return None
    if isinstance(segment, CallSegment) and segment.name in calls:
        subject, annotation, verb = f"{segment.name}(...)", calls[segment.name], "returns"
    elif isinstance(segment, InputSegment) and segment.name in bindings:
        subject, annotation, verb = segment.name, bindings[segment.name], "is declared"
    else:
        return None
    assert segment.loc is not None  # parsed segments always carry a span
    severity = lsp.DiagnosticSeverity.Error
    if media and _MEDIA_TYPE.fullmatch(annotation) and not annotation.strip().startswith("list"):
        problem = f"{subject} is media"
    elif _only_empty_renders(annotation):
        # Only an empty container renders, so the program may still run: a warning.
        severity = lsp.DiagnosticSeverity.Warning
        problem = f"{subject} {verb} {annotation.strip()}, which has a text form only when empty"
    else:
        problem = f"{subject} {verb} {annotation.strip()}, which has no text form"
    return lsp.Diagnostic(
        range=span_to_range(segment.loc),
        severity=severity,
        code="K-SUBSTITUTION-TYPE",
        source="kedi",
        message=(
            f"<{subject}> renders text, but {problem}. In a model prompt pass it by reference "
            f"with <<{subject}>>; to keep the value use single-backtick Python."
        ),
    )


def _nested(segments: Sequence[object]) -> Iterator[object]:
    for segment in segments:
        yield segment
        if isinstance(segment, CallSegment):
            for argument in segment.args:
                yield from _nested(list(iter_call_arg_nested(argument)))


__all__ = ["annotation_has_no_text", "substitution_type_diagnostics"]
