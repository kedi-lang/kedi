"""
Defines the Expression base class and its dependent classes for the AST (Abstract Syntax Tree).
"""

class Expression:
    """
    Base class for all expressions in the AST.
    This class can be extended to create specific types of expressions.
    """
    
    def evaluate(self, expr: str):
        """
        Evaluate the expression and return its value.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")