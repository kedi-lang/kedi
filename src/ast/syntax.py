"""
Implementation module for syntax elements.
"""

from statements  import Statement  as Stmt
from expressions import Expression as Expr

from typing import Any, List, TypeVar

InferT  = TypeVar('InferT' , bound=Any)
OutputT = TypeVar('OutputT', bound=str)

class In(Expr):
    """
    An input node to inject dependencies into prompt.
    """
    id: str
    type_: type[InferT]
    

class Out(Expr):
    """
    An output node to provide output based on LLM invokes.
    """
    id: str
    text: str | None
    type_: type[OutputT]


class MarkerRunExpr(Expr):
    """
    MarkerRunExpr is an expression to run a collection of markers that has return statement.
    Also can be used as an expression in statements.
    """
    pass 

class MarkerRunStmt(Stmt):
    """
    MarkerRunStmt is an expression to run a collection of markers.
    """
    pass

class Program(Stmt):
    """
    A program that runs every single MarkerRunStmt.
    """
    run_list: List[MarkerRunStmt]