"""
Implementation module for syntax nodes.
"""

from .statements import Statement as Stmt
from .expressions import Expression as Expr

from pydantic import BaseModel

from typing import Any, List, Optional, TypeVar, Union

ParamT = TypeVar("ParamT", bound=type)
OutputT = TypeVar("OutputT", bound=type)
ReturnT = TypeVar("ReturnT", bound=type)


class ASTNode(BaseModel):
    pass

class ExprStmt(Stmt):
    expr: Expr | None = None

class PythonExpr(Expr):
    pass


class Param(ASTNode):
    id: str
    type_: ParamT


class Literal(Expr):
    value: Any

class Identifier(Expr):
    name: str

    def __str__(self):
        return self.name

class In(Expr):
    expr: Expr

    def __str__(self):
        return str(self.expr)


class Out(ASTNode):
    id: str
    text: str | None
    type_: OutputT


class MarkerText(ASTNode):
    text: str

    def __str__(self):
        return self.text


class MarkerReturn(ASTNode):
    text: str
    type_: ReturnT


class MarkerRun(Expr):
    markers: List[Union[In, Out, MarkerText, MarkerReturn]]


class DefineProcedure(Stmt):
    id: str
    params: Optional[List[Param]]
    stmt_list: List[Stmt]
    
    def __str__(self):
        return self.id


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


class Program(ASTNode):
    statements: List[Stmt]
