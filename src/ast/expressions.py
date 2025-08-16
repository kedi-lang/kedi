"""
Defines the Expression base class and its dependent classes for the AST (Abstract Syntax Tree).
"""

import abc
from pydantic import BaseModel


class Expression(BaseModel):
    """
    Abstract class for all expressions in the AST.
    This class can be extended to create specific types of expressions.
    """

    @abc.abstractmethod
    def evaluate(self, expr: str):
        """
        Evaluate the expression and return its value.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError
