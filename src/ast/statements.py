"""
Defines the Statement base class and its dependent classes for the AST (Abstract Syntax Tree).
"""

import abc
from pydantic import BaseModel

class Statement(BaseModel):
    """
    Abstract class for all statements.
    """

    @abc.abstractmethod
    def run(self):
        """
        Run the statement.
        """
        raise NotImplementedError
