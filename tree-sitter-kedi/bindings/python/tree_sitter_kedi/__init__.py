"""Tree-sitter grammar bindings for the Kedi DSL.

Usage:

    from tree_sitter import Language, Parser
    import tree_sitter_kedi

    parser = Parser(Language(tree_sitter_kedi.language()))
    tree = parser.parse(open("program.kedi", "rb").read())
"""

from ._binding import language

__all__ = ["language"]
