"""Tree-sitter parser singleton for Kedi.

Loads the compiled ``tree_sitter_kedi`` shared library once per process, keeps one
parser per thread, and exposes a single ``parse(source)`` function that returns a
``tree_sitter.Tree``. The CST→AST walker in :mod:`cst_to_ast` consumes
that tree.
"""

from __future__ import annotations

from threading import Lock, local
from typing import Optional

import tree_sitter_kedi
from tree_sitter import Language, Parser, Tree

_lock = Lock()
_language: Optional[Language] = None
# tree-sitter releases the GIL while parsing, so one Parser must not be shared by threads.
_thread_state = local()


def get_language() -> Language:
    """Return the process-wide Kedi ``Language`` (needed for Query)."""
    global _language
    if _language is not None:
        return _language
    with _lock:
        if _language is None:
            _language = Language(tree_sitter_kedi.language())
    return _language


def _get_parser() -> Parser:
    """Return this thread's ``Parser``, creating it lazily."""
    parser: Parser | None = getattr(_thread_state, "parser", None)
    if parser is None:
        parser = Parser(get_language())
        _thread_state.parser = parser
    return parser


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
