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
    ModuleExport,
    ModuleImport,
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
from .procedure_schema import (
    procedure_json_schema,
    procedure_return_json_schema,
    program_procedure_json_schemas,
)

__all__ = [
    "Program",
    "ProgramWithValidation",
    "ModuleImport",
    "ModuleExport",
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
    "procedure_json_schema",
    "procedure_return_json_schema",
    "program_procedure_json_schemas",
    "KediDiagnostic",
    "KediParseError",
]
