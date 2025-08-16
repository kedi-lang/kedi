"""
Implementation module for syntax nodes.
"""

from statements import Statement as Stmt
from expressions import Expression as Expr

from typing import Any, List, Optional, TypeVar, Union

ParamT = TypeVar("ParamT", bound=Any)
OutputT = TypeVar("OutputT", bound=str)


class ASTNode(object):
    pass


class PythonExpr(Expr):
    pass


class Param(ASTNode):
    id: str
    type_: type[ParamT]


class Literal(Expr):
    value: Any


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


class ForEachLoop(Stmt):
    iter_id: str
    iterable: Expr
    stmt_list: List[Stmt]


class WhileLoop(Stmt):
    condition: Expr
    stmt_list: List[Stmt]


class ExprStmt(Stmt):
    expr: Expr


class Program(ASTNode):
    statements: List[Stmt]
