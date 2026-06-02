"""Kedi DSL parser package.

Phase-6 refactor: the hand-written parser previously in this package
has been replaced by a tree-sitter-generated parser plus a Python
CST→AST walker. The public API is unchanged:

  - ``parse_program(source) -> Program | ProgramWithValidation``

Modules:

  - ``parser`` — tree-sitter ``Parser`` singleton.
  - ``cst_to_ast`` — walks the tree-sitter CST and produces AST
    dataclasses defined in :mod:`kedi.lang.ast`.
  - ``diagnostics`` — :class:`KediDiagnostic` records and
    :class:`KediParseError` (which subclasses ``ValueError`` for
    backward compatibility).
  - ``escapes`` — template-text escape unescaping and fenced-block
    dedenting helpers, ported verbatim from the pre-refactor parser.

The legacy modules (``preprocessing``, ``program_parser``,
``segment_parser``, ``statement_parser``, ``block_parser``) are
retained on disk until the Phase-8 cleanup so anything that imports
them by name continues to work during the migration window.
"""

from __future__ import annotations

from typing import Optional, Union

from ..ast import Program, ProgramWithValidation
from ...source_loading import CompositeSourceMap, single_file_source_map
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
