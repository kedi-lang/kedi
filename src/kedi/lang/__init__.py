"""Public API for the Kedi language layer.

Parsing is handled by the tree-sitter grammar and CST→AST walker.
The legacy ``TypeValidator`` / ``ValidationResult`` helpers from
``lang/types.py`` were removed — the structured ``TypeExpr`` hierarchy
in :mod:`kedi.lang.ast` covers the cases ``TypeValidator`` used to
regex-parse.
"""

from __future__ import annotations

from .ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    CallSegment,
    ConditionalLoopStmt,
    IfStmt,
    InputSegment,
    LoopStmt,
    ModuleExport,
    ModuleImport,
    OutputSegment,
    PackageMetadata,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    RawInvokeStmt,
    ReturnStmt,
    ShowStmt,
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
    VariableInitBlockStmt,
    VariableInitStmt,
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
    "PackageMetadata",
    "ProcedureDef",
    "TemplateBlock",
    "TemplateLine",
    "IfStmt",
    "ConditionalLoopStmt",
    "LoopStmt",
    "RawInvokeStmt",
    "VariableInitBlockStmt",
    "VariableInitStmt",
    "AssignmentBlockStmt",
    "AssignmentStmt",
    "ReturnStmt",
    "ShowStmt",
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
