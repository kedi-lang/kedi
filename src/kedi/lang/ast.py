"""Kedi AST dataclasses.

Provides typed dataclass nodes for parsed Kedi programs:

  - ``Span`` — byte and line/col extents of every source construct,
    populated by the CST→AST walker from tree-sitter node positions.
  - ``TypeExpr`` and friends (``TypeRef``, ``TypeApply``, ``TypeUnion``,
    ``TypePython``) — a structured representation of Kedi type
    annotations parsed by the grammar and dispatched on by the compiler.
  - A ``loc`` field on every node that records its ``Span``.
  - ``ProgramWithValidation`` is what ``parse_program`` returns
    whenever ``@test:`` or ``@eval:`` blocks are present.

Backward compatibility: the legacy string fields (``type_annotation``,
``param_type_annotations``, ``return_type_annotation``) are retained on
the relevant classes. The CST→AST walker fills both the legacy string
field and the ``type_expr`` field for every annotation it observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from .parser.diagnostics import KediDiagnostic

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
    source_id: Optional[str] = None


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
    native_backtick: Optional[str] = field(default=None, repr=False, compare=False)
    render_segments: List["Segment"] = field(default_factory=list, repr=False, compare=False)
    call_segment: Optional["CallSegment"] = field(default=None, repr=False, compare=False)
    python_segment: Optional["PythonExprSegment"] = field(default=None, repr=False, compare=False)


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
    # Structured form populated by the CST→AST walker.
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
class TemplateBlock(Stmt):
    """Multiline template run opened by ``>>``.

    Rows are newline-joined and executed as a single LLM prompt. Within the
    block, ``<name>`` references to outputs declared on earlier rows are kept
    as literal placeholders in the merged prompt.
    """

    lines: List[TemplateLine] = field(default_factory=list)


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
class ModelDirectiveStmt(Stmt):
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None


@dataclass
class EffortDirectiveStmt(Stmt):
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None


@dataclass
class SystemDirectiveStmt(Stmt):
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None
    template_lines: List[TemplateLine] = field(default_factory=list)


@dataclass
class McpField:
    name: str = ""
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class McpDirectiveStmt(Stmt):
    fields: List[McpField] = field(default_factory=list)


@dataclass
class UseDirectiveStmt(Stmt):
    target: str = ""
    tool_names: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.target

    @name.setter
    def name(self, value: str) -> None:
        self.target = value


UseProfileStmt = UseDirectiveStmt


@dataclass
class ProfileDefStmt(Stmt):
    name: str = ""
    model_members: List["ProfileModelEntry"] = field(default_factory=list)
    effort_members: List["ProfileEffortEntry"] = field(default_factory=list)
    system_members: List["ProfileSystemEntry"] = field(default_factory=list)
    mcp_members: List["ProfileMcpEntry"] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    docstring_loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProfileModelEntry:
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProfileEffortEntry:
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProfileSystemEntry:
    plain_value: Optional[str] = None
    backtick_code: Optional[str] = None
    template_lines: List[TemplateLine] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProfileMcpEntry:
    fields: List[McpField] = field(default_factory=list)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class TypeField:
    name: str
    type_annotation: Optional[str] = None
    type_expr: Optional[TypeExpr] = None
    default_expr: Optional[str] = None
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
    docstring: Optional[str] = None
    docstring_loc: Optional[Span] = field(default=None, repr=False, compare=False)
    return_type_annotation: Optional[str] = None
    return_type_expr: Optional[TypeExpr] = None
    # Maps parameter name to legacy string annotation. Kept for
    # backward-compat; ``param_type_exprs`` is the new structured form.
    param_type_annotations: Optional[dict[str, Optional[str]]] = None
    param_type_exprs: Optional[dict[str, Optional[TypeExpr]]] = None
    param_defaults: Optional[dict[str, str]] = None
    lexical_env: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)
    # Legacy AI-prompt-only spec (superseded by `auto_spec`).
    ai_prompt: Optional[str] = None
    auto_spec: Optional[AutoSpec] = None
    optimize_spans: List[OptimizeSpan] = field(default_factory=list)
    captured_agent_profile: Any = field(default=None, repr=False, compare=False)
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ProcDefStmt(Stmt):
    proc: Optional[ProcedureDef] = None


@dataclass
class TypeDefStmt(Stmt):
    typedef: Optional[TypeDef] = None


@dataclass
class ModuleImport:
    module: str
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class ModuleExport:
    names: List[str] = field(default_factory=list)
    export_all: bool = False
    loc: Optional[Span] = field(default=None, repr=False, compare=False)


@dataclass
class Program:
    imports: List[ModuleImport] = field(default_factory=list)
    exports: List[ModuleExport] = field(default_factory=list)
    procedures: List[ProcedureDef] = field(default_factory=list)
    toplevel_return: Optional[Union[ReturnStmt, ReturnBlockStmt]] = None
    toplevel_body: List[Stmt] = field(default_factory=list)
    prelude: Optional[str] = None
    prelude_loc: Optional[Span] = field(default=None, repr=False, compare=False)
    type_defs: List[TypeDef] = field(default_factory=list)
    profile_defs: dict[str, "ProfileDefStmt"] = field(default_factory=dict)
    ordered_toplevel: List[Union["ProcedureDef", Stmt]] = field(default_factory=list)
    source_text: Optional[str] = field(default=None, repr=False, compare=False)
    source_map: Any = field(default=None, repr=False, compare=False)
    parser_warnings: List[KediDiagnostic] = field(default_factory=list, repr=False, compare=False)
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
