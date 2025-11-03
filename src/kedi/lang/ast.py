from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# Segments used inside template/return expressions
@dataclass
class Segment:
    pass


@dataclass
class TextSegment(Segment):
    text: str
    escaped_whitespace: set[int] = field(
        default_factory=set
    )  # Positions of escaped whitespace chars


@dataclass
class InputSegment(Segment):
    name: str


@dataclass
class CallArg:
    pass


@dataclass
class LiteralArg(CallArg):
    value: str


@dataclass
class VarArg(CallArg):
    name: str


@dataclass
class CallSegment(Segment):
    name: str
    args: List[CallArg] = field(default_factory=list)


@dataclass
class OutputSegment(Segment):
    name: str
    type_annotation: Optional[str] = None


@dataclass
class PythonExprSegment(Segment):
    code: str


@dataclass
class PythonExprNativeSegment(Segment):
    code: str


Expr = List[Segment]


# Statements
@dataclass
class Stmt:
    pass


@dataclass
class TemplateLine(Stmt):
    segments: Expr


@dataclass
class ReturnStmt(Stmt):
    segments: Expr


@dataclass
class PythonBlock(Stmt):
    code: str


@dataclass
class BacktickLineStmt(Stmt):
    code: str


@dataclass
class AssignBlockStmt(Stmt):
    name: str
    code: str
    type_annotation: Optional[str] = None


@dataclass
class ReturnBlockStmt(Stmt):
    code: str


@dataclass
class AssignStmt(Stmt):
    name: str
    expr: Expr
    type_annotation: Optional[str] = None


@dataclass
class TypeField:
    name: str
    type_annotation: Optional[str] = None


@dataclass
class TypeDef:
    name: str
    fields: List[TypeField]


@dataclass
class OptimizeSpan:
    name: str
    template_lines: List[TemplateLine]  # Contiguous template lines to optimize


@dataclass
class AutoSpec:
    prompt: str  # Content after > auto:


@dataclass
class ProcedureDef:
    name: str
    params: List[str]
    body: List[Stmt] = field(default_factory=list)
    return_type_annotation: Optional[str] = None
    # Optional mapping of parameter name to type annotation string
    param_type_annotations: Optional[dict[str, Optional[str]]] = None
    lexical_env: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)
    # If present, this procedure is an AI-prompt-only spec (deprecated, use auto_spec instead)
    ai_prompt: Optional[str] = None
    # New codegen spec using explicit > auto: syntax
    auto_spec: Optional[AutoSpec] = None
    # Optimize spans for prompt optimization
    optimize_spans: List[OptimizeSpan] = field(default_factory=list)


@dataclass
class ProcDefStmt(Stmt):
    proc: ProcedureDef


@dataclass
class TypeDefStmt(Stmt):
    typedef: TypeDef


@dataclass
class Program:
    procedures: List[ProcedureDef] = field(default_factory=list)
    toplevel_return: Optional[Union[ReturnStmt, ReturnBlockStmt]] = None
    toplevel_body: List[Stmt] = field(default_factory=list)
    prelude: Optional[str] = None
    type_defs: List[TypeDef] = field(default_factory=list)


# --- Testing and Evaluation AST nodes ---


@dataclass
class TestCase:
    name: str
    code: str
    is_block: bool = False


@dataclass
class TestSuite:
    procedure_name: str
    cases: List[TestCase] = field(default_factory=list)


@dataclass
class EvalDataset:
    name: str
    code: str
    is_block: bool = False


@dataclass
class EvalMetric:
    name: str
    code: str
    is_block: bool = False
    dataset_name: Optional[str] = None  # If None, legacy single-eval behavior


@dataclass
class EvalSuite:
    procedure_name: str
    datasets: dict[str, EvalDataset] = field(default_factory=dict)
    test_datasets: dict[str, EvalDataset] = field(default_factory=dict)
    metrics: List[EvalMetric] = field(default_factory=list)


@dataclass
class ProgramWithValidation(Program):
    tests: List[TestSuite] = field(default_factory=list)
    evals: List[EvalSuite] = field(default_factory=list)
