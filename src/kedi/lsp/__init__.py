"""``kedi-lsp`` — Language Server Protocol scaffold for Kedi.

The server is a thin ``pygls`` shim over the tree-sitter parser +
CST→AST walker:

  - Diagnostics: re-parse on every ``didOpen`` / ``didChange`` and
    publish ``KediDiagnostic``s with their ``Span`` info.
  - Hover: identify the symbol under the cursor (procedure name, type
    identifier, parameter, output placeholder) and surface a short
    description.
  - Definition: jump to the corresponding ``ProcedureDef`` /
    ``TypeDef`` / parameter / output declaration when the cursor is on
    a reference.

The server also exposes completions, references, rename, formatting,
semantic tokens, signatures, inlay hints, and embedded-Python projections.

Entry point::

    kedi-lsp

is registered via ``[project.scripts]`` and starts the server over
stdio (the LSP convention).
"""

from __future__ import annotations

from .server import main

__all__ = ["main"]
