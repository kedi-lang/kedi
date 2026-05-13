"""Tree-sitter grammar bindings for the Kedi DSL — typing stub."""

from typing import Any

def language() -> Any:
    """Return a PyCapsule wrapping the TSLanguage pointer for Kedi.

    The returned capsule is consumed by `tree_sitter.Language(capsule)`.
    """
