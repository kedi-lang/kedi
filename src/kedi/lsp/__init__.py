"""``kedi-lsp`` — Language Server Protocol scaffold for Kedi.

Phase-9 deliverable of the tree-sitter refactor. The server is a thin
``pygls`` shim over the tree-sitter parser + CST→AST walker:

  - Diagnostics: re-parse on every ``didOpen`` / ``didChange`` and
    publish ``KediDiagnostic``s with their ``Span`` info.
  - Hover: identify the symbol under the cursor (procedure name, type
    identifier, parameter, output placeholder) and surface a short
    description.
  - Definition: jump to the corresponding ``ProcedureDef`` /
    ``TypeDef`` / parameter / output declaration when the cursor is on
    a reference.

Deferred (intentionally out of scope for the scaffold): completions,
rename, formatting, semantic-tokens, incremental parsing wiring.

Entry point::

    kedi-lsp

is registered via ``[project.scripts]`` and starts the server over
stdio (the LSP convention).
"""

from __future__ import annotations

from .server import main

__all__ = ["main"]
