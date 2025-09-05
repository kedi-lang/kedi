"""
Defines the Statement base class and its dependent classes for the AST (Abstract Syntax Tree).
"""

from pydantic import BaseModel

class Statement(BaseModel):
    """
    Abstract class for all statements.
    """

    def run(self):
        """
        Run the statement.
        """
        raise NotImplementedError
