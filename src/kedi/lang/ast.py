"""Kedi AST dataclasses.

This module is the result of the Phase-6 tree-sitter refactor. Compared
to the pre-refactor AST it gains:

  - ``Span`` — byte and line/col extents of every source construct,
    populated by the CST→AST walker from tree-sitter node positions.
  - ``TypeExpr`` and friends (``TypeRef``, ``TypeApply``, ``TypeUnion``,
    ``TypePython``) — a STRUCTURED representation of Kedi type
    annotations. Previously every type annotation was a raw string
    later ``eval``'d at runtime; now the grammar parses the annotation
    structurally and the compiler dispatches on TypeExpr nodes.
  - A ``loc`` field on every node that records its ``Span``.
  - ``ProgramWithValidation`` is now what ``parse_program`` returns
    whenever ``@test:`` or ``@eval:`` blocks are present (rather than
    being unused dead code, with validation suites grafted onto a plain
    ``Program`` via ``setattr``).

Backward compatibility: the legacy string fields (``type_annotation``,
``param_type_annotations``, ``return_type_annotation``) are retained on
the relevant classes so existing compiler / codegen / optimizer code
keeps working until those call sites migrate to ``type_expr`` in
Phase 7. The CST→AST walker fills BOTH the legacy string field AND the
new ``type_expr`` field for every annotation it observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# =====================================================================
# Source-range tracking
# =====================================================================


@dataclass
class Span:
    """Byte and line/col extent of an AST node in its source file.

    Line and column are 1-based for consumer ergonomics (LSP, error
    messages). Byte offsets are 0-based (Python slicing).
    """

    start_byte: int
    end_byte: int
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    source_path: Optional[str] = None


# =====================================================================
# Structured type expressions
# =====================================================================


@dataclass
class TypeExpr:
    """Base for structured type expressions."""

    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TypeRef(TypeExpr):
    """A bare type name, e.g. ``int``, ``str``, ``Person``."""

    name: str = ""


@dataclass
class TypeApply(TypeExpr):
    """A generic type application, e.g. ``list[int]``, ``dict[str, int]``."""

    name: str = ""
    args: List[TypeExpr] = field(default_factory=list)


@dataclass
class TypeUnion(TypeExpr):
    """A type union via ``|``, e.g. ``int | str`` (left-associative)."""

    left: Optional[TypeExpr] = None
    right: Optional[TypeExpr] = None


@dataclass
class TypePython(TypeExpr):
    """A backtick-wrapped Python type expression (escape hatch).

    Carries arbitrary Python source that the compiler evaluates at
    runtime in the type environment. This is the ONLY remaining
    ``eval`` path in the type system; structured ``TypeRef`` /
    ``TypeApply`` / ``TypeUnion`` cover the common cases statically.
    """

    code: str = ""


# =====================================================================
# Segments used inside template / return expressions
# =====================================================================


@dataclass
class Segment:
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TextSegment(Segment):
    text: str = ""
    # Positions (within the unescaped text) of characters that were
    # produced by `\t`, `\n`, `\s` escapes. The runtime preserves these
    # as whitespace when rendering the template.
    escaped_whitespace: set[int] = field(default_factory=set)


@dataclass
class InputSegment(Segment):
    name: str = ""


@dataclass
class CallArg:
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class LiteralArg(CallArg):
    value: str = ""


@dataclass
class VarArg(CallArg):
    name: str = ""


@dataclass
class CallSegment(Segment):
    name: str = ""
    args: List[CallArg] = field(default_factory=list)


@dataclass
class OutputSegment(Segment):
    name: str = ""
    # Legacy string form (kept for code that reads the raw annotation
    # text — codegen, optimization, the in-progress KediType migration).
    type_annotation: Optional[str] = None
    # New structured form populated by the Phase-6 CST→AST walker.
    type_expr: Optional[TypeExpr] = None


@dataclass
class PythonExprSegment(Segment):
    code: str = ""


@dataclass
class PythonExprNativeSegment(Segment):
    code: str = ""


Expr = List[Segment]


# =====================================================================
# Statements
# =====================================================================


@dataclass
class Stmt:
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TemplateLine(Stmt):
    segments: Expr = field(default_factory=list)
    # Set by the GEPA optimizer when this line is the target of an
    # `> optimize:` span; either an extra prompt prefix or a dict of
    # deterministic output overrides.
    optimized_prompt_prefix: Optional[str] = None
    optimized_output_override: Optional[dict[str, Any]] = None


@dataclass
class ReturnStmt(Stmt):
    segments: Expr = field(default_factory=list)


@dataclass
class PythonBlock(Stmt):
    code: str = ""


@dataclass
class BacktickLineStmt(Stmt):
    code: str = ""


@dataclass
class AssignBlockStmt(Stmt):
    name: str = ""
    code: str = ""
    type_annotation: Optional[str] = None
    type_expr: Optional[TypeExpr] = None


@dataclass
class ReturnBlockStmt(Stmt):
    code: str = ""


@dataclass
class AssignStmt(Stmt):
    name: str = ""
    expr: Expr = field(default_factory=list)
    type_annotation: Optional[str] = None
    type_expr: Optional[TypeExpr] = None


@dataclass
class TypeField:
    name: str
    type_annotation: Optional[str] = None
    type_expr: Optional[TypeExpr] = None
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TypeDef:
    name: str
    fields: List[TypeField]
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class OptimizeSpan:
    name: str
    template_lines: List[TemplateLine] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class AutoSpec:
    prompt: str  # Content after `> auto:`
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProcedureDef:
    name: str
    params: List[str]
    body: List[Stmt] = field(default_factory=list)
    return_type_annotation: Optional[str] = None
    return_type_expr: Optional[TypeExpr] = None
    # Maps parameter name to legacy string annotation. Kept for
    # backward-compat; ``param_type_exprs`` is the new structured form.
    param_type_annotations: Optional[dict[str, Optional[str]]] = None
    param_type_exprs: Optional[dict[str, Optional[TypeExpr]]] = None
    lexical_env: Optional[dict[str, Any]] = field(
        default=None, repr=False, compare=False
    )
    # Legacy AI-prompt-only spec (superseded by `auto_spec`).
    ai_prompt: Optional[str] = None
    auto_spec: Optional[AutoSpec] = None
    optimize_spans: List[OptimizeSpan] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProcDefStmt(Stmt):
    proc: Optional[ProcedureDef] = None


@dataclass
class TypeDefStmt(Stmt):
    typedef: Optional[TypeDef] = None


@dataclass
class Program:
    procedures: List[ProcedureDef] = field(default_factory=list)
    toplevel_return: Optional[Union[ReturnStmt, ReturnBlockStmt]] = None
    toplevel_body: List[Stmt] = field(default_factory=list)
    prelude: Optional[str] = None
    type_defs: List[TypeDef] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


# =====================================================================
# Testing and evaluation
# =====================================================================


@dataclass
class TestCase:
    name: str
    code: str
    is_block: bool = False
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TestSuite:
    procedure_name: str
    cases: List[TestCase] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class EvalDataset:
    name: str
    code: str
    is_block: bool = False
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class EvalMetric:
    name: str
    code: str
    is_block: bool = False
    dataset_name: Optional[str] = None  # None for legacy single-eval behavior
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class EvalSuite:
    procedure_name: str
    datasets: dict[str, EvalDataset] = field(default_factory=dict)
    test_datasets: dict[str, EvalDataset] = field(default_factory=dict)
    metrics: List[EvalMetric] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProgramWithValidation(Program):
    tests: List[TestSuite] = field(default_factory=list)
    evals: List[EvalSuite] = field(default_factory=list)
