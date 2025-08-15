"""
Implementation module for syntax nodes.
"""

from statements import Statement as Stmt
from expressions import Expression as Expr

from typing import Any, List, Optional, TypeVar

ParamT = TypeVar("ParamT", bound=Any)
OutputT = TypeVar("OutputT", bound=str)


class ASTNode(object):
    pass


class PythonExpr(Expr):
    pass


class Param(ASTNode):
    id: str
    type_: type[ParamT]


class In(Expr):
    id: str


class Out(Expr):
    id: str
    text: str | None
    type_: type[OutputT]


class MarkerRun(Expr):
    markers: List[Expr]


class DefineProcedure(Stmt):
    id: str
    params: Optional[List[Param]]
    stmt_list: List[Stmt]


class CallProcedure(Expr):
    id: str
    args: List[Expr]


class ExprStmt(Stmt):
    expr: Expr


class Program(ASTNode):
    statements: List[Stmt]
