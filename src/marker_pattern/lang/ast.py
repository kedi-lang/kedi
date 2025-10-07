from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


# Segments used inside template/return expressions
@dataclass
class Segment:
    pass


@dataclass
class TextSegment(Segment):
    text: str


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
class AssignStmt(Stmt):
    name: str
    expr: Expr
    type_annotation: Optional[str] = None


@dataclass
class ProcedureDef:
    name: str
    params: List[str]
    body: List[Stmt] = field(default_factory=list)


@dataclass
class Program:
    procedures: List[ProcedureDef] = field(default_factory=list)
    toplevel_return: Optional[ReturnStmt] = None
    toplevel_body: List[Stmt] = field(default_factory=list)


