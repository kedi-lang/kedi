"""
Defines the Statement base class and its dependent classes for the AST (Abstract Syntax Tree).
"""

import abc


class Statement(abc.ABC):
    """
    Abstract class for all statements.
    """

    @abc.abstractmethod
    def run(self):
        """
        Run the statement.
        """
        raise NotImplementedError("Subclasses must implement this method.")
