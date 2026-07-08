"""Public API for the Kedi language layer.

Parsing is handled by the tree-sitter grammar and CST→AST walker.
The legacy ``TypeValidator`` / ``ValidationResult`` helpers from
``lang/types.py`` were removed — the structured ``TypeExpr`` hierarchy
in :mod:`kedi.lang.ast` covers the cases ``TypeValidator`` used to
regex-parse.
"""

from __future__ import annotations

from .ast import (
    CallSegment,
    InputSegment,
    ModuleExport,
    ModuleImport,
    OutputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    ReturnStmt,
    Segment,
    Span,
    TemplateBlock,
    TemplateLine,
    TextSegment,
    TypeApply,
    TypeExpr,
    TypeLiteral,
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
    "TemplateBlock",
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
    "TypeLiteral",
    "parse_program",
    "compile_program",
    "procedure_json_schema",
    "procedure_return_json_schema",
    "program_procedure_json_schemas",
    "KediDiagnostic",
    "KediParseError",
]
