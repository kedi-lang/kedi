"""Kedi DSL parser package.

The parser is backed by tree-sitter and a Python CST-to-AST walker.
The public entry point is ``parse_program(source)``; it returns a
``Program`` or ``ProgramWithValidation`` and raises ``KediParseError``
for syntax or structural diagnostics.

Package modules:

  - ``parser``: tree-sitter language/parser setup.
  - ``cst_to_ast``: CST-to-AST conversion into :mod:`kedi.lang.ast`.
  - ``diagnostics``: parser diagnostics and ``KediParseError``.
  - ``escapes``: template escaping and fenced-block dedenting helpers.
"""

from __future__ import annotations

from typing import Optional, Union

from ...source_loading import CompositeSourceMap, single_file_source_map
from ...telemetry import capture_enabled, record_histogram, runtime_span
from ...telemetry.naming import program_name
from ..ast import Program, ProgramWithValidation
from .cst_to_ast import cst_to_ast
from .diagnostics import KediDiagnostic, KediParseError
from .parser import parse as _parse


def parse_program(
    source: str,
    *,
    source_path: Optional[str] = None,
    source_map: CompositeSourceMap | None = None,
) -> Union[Program, ProgramWithValidation]:
    """Parse Kedi source text into an AST.

    Args:
        source: the program text.
        source_path: optional path string attached to every emitted
            ``Span`` for diagnostic / LSP rendering. Pass the file
            path if you have one.

    Returns:
        A ``Program`` when no validation blocks are present, or a
        ``ProgramWithValidation`` (which is a ``Program``) when one or
        more ``@test:`` / ``@eval:`` blocks appear at the top level.

    Raises:
        :class:`KediParseError` (a ``ValueError`` subclass) if the
        source contains parse errors or fails structural validation.
    """
    source_text = source.decode("utf-8") if isinstance(source, bytes) else source
    source_bytes = source_text.encode("utf-8")
    attributes: dict[str, bool | str | int | float] = {
        "kedi.program.name": program_name(source_path),
        "kedi.source.bytes": len(source_bytes),
        "kedi.source.lines": source_text.count("\n") + (1 if source_text else 0),
    }
    if source_path is not None and capture_enabled("source"):
        attributes["code.file.path"] = source_path

    source_type = "file" if source_path and not source_path.startswith("<") else "memory"
    record_histogram(
        "runtime",
        "kedi.parse.source.size",
        len(source_bytes),
        unit="By",
        attributes={"source_type": source_type},
    )
    with runtime_span(
        f"parse {attributes['kedi.program.name']}",
        "parse_program",
        attributes=attributes,
        metric_attributes={"source_type": source_type},
    ) as active:
        try:
            tree = _parse(source_bytes)
            resolved_source_map = source_map
            if resolved_source_map is None and source_path is not None:
                resolved_source_map = single_file_source_map(source_path, source_text)
            program = cst_to_ast(
                source_bytes,
                tree,
                source_path=source_path,
                source_map=resolved_source_map,
            )
        except KediParseError as exc:
            active.set_attribute("error.type", type(exc).__name__)
            active.set_attribute("kedi.parse.error_count", len(exc.diagnostics))
            if exc.diagnostics:
                primary = exc.diagnostics[0]
                active.set_attribute("code.line.number", primary.span.start_line)
                active.set_attribute("code.column.number", primary.span.start_col)
                for diagnostic in exc.diagnostics[1:11]:
                    event_attributes: dict[str, str | int] = {
                        "error.type": type(exc).__name__,
                        "code.line.number": diagnostic.span.start_line,
                        "code.column.number": diagnostic.span.start_col,
                    }
                    if diagnostic.code is not None:
                        event_attributes["kedi.diagnostic.code"] = diagnostic.code
                    active.add_event("kedi.diagnostic", event_attributes)
            raise
        except BaseException as exc:
            active.set_attribute("error.type", type(exc).__name__)
            raise

        program.source_text = source_text
        program.source_map = resolved_source_map
        statement_count = (
            len(program.ordered_toplevel)
            + len(program.exports)
            + int(program.package is not None)
            + int(program.prelude is not None)
            + int(program.toplevel_return is not None)
        )
        if isinstance(program, ProgramWithValidation):
            statement_count += len(program.tests) + len(program.evals)
        active.set_attribute("kedi.parse.statement_count", statement_count)
        active.set_attribute("kedi.parse.warning_count", len(program.parser_warnings))
        return program


__all__ = [
    "parse_program",
    "KediDiagnostic",
    "KediParseError",
]
