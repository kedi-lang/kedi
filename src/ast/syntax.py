"""
Implementation module for syntax nodes.
"""

from statements import Statement as Stmt
from expressions import Expression as Expr

from pydantic import BaseModel

from typing import Any, List, Optional, TypeVar, Union

ParamT  = TypeVar("ParamT" , bound=str)
OutputT = TypeVar("OutputT", bound=str)
ReturnT = TypeVar("ReturnT", bound=str)


class ASTNode(BaseModel):
    pass


class PythonExpr(Expr):
    pass


class Param(ASTNode):
    id: str
    type_: type[ParamT]

class Literal(Expr):
    value: Any


class In(Expr):
    expr: Expr


class Out(ASTNode):
    id: str
    text: str | None
    type_: type[OutputT]

class MarkerText(ASTNode):
    text: str

class MarkerReturn(ASTNode):
    text: str
    type_: type[ReturnT]

class MarkerRun(Expr):
    markers: List[Union[MarkerText, In, Out, MarkerReturn]]

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