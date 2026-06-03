"""Tree-sitter parser singleton for Kedi.

Loads the compiled ``tree_sitter_kedi`` shared library once per process
and exposes a single ``parse(source)`` function that returns a
``tree_sitter.Tree``. The CST→AST walker in :mod:`cst_to_ast` consumes
that tree.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import tree_sitter_kedi
from tree_sitter import Language, Parser, Tree

_lock = Lock()
_language: Optional[Language] = None
_parser: Optional[Parser] = None


def _get_parser() -> Parser:
    """Return the process-wide ``Parser`` instance, creating it lazily."""
    global _language, _parser
    if _parser is not None:
        return _parser
    with _lock:
        if _parser is None:
            _language = Language(tree_sitter_kedi.language())
            _parser = Parser(_language)
    assert _parser is not None
    return _parser


def get_language() -> Language:
    """Return the process-wide Kedi ``Language`` (needed for Query)."""
    _get_parser()
    assert _language is not None
    return _language


def parse(source: str | bytes) -> Tree:
    """Parse Kedi source text into a tree-sitter ``Tree``.

    The parser accepts ``bytes`` natively; ``str`` is encoded as UTF-8
    here. Callers that already have bytes should call
    ``_get_parser().parse(...)`` directly.
    """
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source
    return _get_parser().parse(source_bytes)
