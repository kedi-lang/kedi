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
    program.source_text = source_text
    program.source_map = resolved_source_map
    return program


__all__ = [
    "parse_program",
    "KediDiagnostic",
    "KediParseError",
]
