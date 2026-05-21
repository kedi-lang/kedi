"""Public API for the Kedi language layer.

The pre-refactor parser package (``preprocessing.py``, ``program_parser.py``,
``segment_parser.py``, ``statement_parser.py``, ``block_parser.py``) was
deleted in Phase 8 of the tree-sitter refactor. So were the legacy
``TypeValidator`` / ``ValidationResult`` helpers from ``lang/types.py`` —
the structured ``TypeExpr`` hierarchy in :mod:`kedi.lang.ast` covers
the cases ``TypeValidator`` used to regex-parse.
"""

from __future__ import annotations

from .ast import (
    CallSegment,
    InputSegment,
    OutputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    ReturnStmt,
    Segment,
    Span,
    TemplateLine,
    TextSegment,
    TypeApply,
    TypeExpr,
    TypePython,
    TypeRef,
    TypeUnion,
)
from .compiler import compile_program
from .parser import KediDiagnostic, KediParseError, parse_program

__all__ = [
    "Program",
    "ProgramWithValidation",
    "ProcedureDef",
    "TemplateLine",
    "ReturnStmt",
    "Segment",
    "TextSegment",
    "InputSegment",
    "CallSegment",
    "OutputSegment",
    "Span",
    "TypeExpr",
    "TypeRef",
    "TypeApply",
    "TypeUnion",
    "TypePython",
    "parse_program",
    "compile_program",
    "KediDiagnostic",
    "KediParseError",
]
