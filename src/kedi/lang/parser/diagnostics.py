"""Structured diagnostics for the tree-sitter-backed Kedi parser.

Compared to the pre-refactor parser — which raised a bare
``ValueError(message)`` with no source location — this module produces
``KediDiagnostic`` records that carry a ``Span`` and a stable code, so
LSP servers and CLI tools can show pointers into the offending source.

``KediParseError`` subclasses ``ValueError`` to preserve backward
compatibility with the many tests that ``pytest.raises(ValueError,
match="...")`` on a specific legacy message — the first line of the
formatted message is the legacy substring; subsequent lines append
the new location info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from ..ast import Span

Severity = Literal["error", "warning"]


@dataclass
class KediDiagnostic:
    """A single parse / structural-validation diagnostic.

    Fields:
        severity: ``"error"`` (parse fails) or ``"warning"`` (parse
            continues; surfaced for tooling).
        message: human-readable description. The first sentence
            should match the legacy parser's message so existing tests
            keep working.
        span: byte / line / column extent of the offending region.
        code: stable identifier (e.g. ``"K001"``) for documentation
            and tooling. Optional.
        snippet: the offending line of source, with no trailing
            newline. Populated by the parser when convenient.
    """

    severity: Severity
    message: str
    span: Span
    code: Optional[str] = None
    snippet: Optional[str] = None

    def format(self) -> str:
        """One-line summary suitable for compiler / CLI output."""
        path = self.span.source_path or "<source>"
        location = f"{path}:{self.span.start_line}:{self.span.start_col}"
        head = f"{location}: {self.severity}: {self.message}"
        if self.code:
            head += f" [{self.code}]"
        if self.snippet:
            caret_col = max(self.span.start_col - 1, 0)
            caret_len = max(self.span.end_col - self.span.start_col, 1)
            head += f"\n    {self.snippet.rstrip()}\n    {' ' * caret_col}{'^' * caret_len}"
        return head


class KediParseError(ValueError):
    """Raised when ``parse_program`` collects one or more error diagnostics.

    Subclasses ``ValueError`` so ``pytest.raises(ValueError, match=...)``
    continues to work for tests written against the pre-refactor parser.
    """

    def __init__(self, diagnostics: List[KediDiagnostic]):
        self.diagnostics = sorted(diagnostics, key=lambda d: d.severity != "error")
        if not diagnostics:
            super().__init__("kedi parse error")
            return
        # The first error's message is the str-representation so
        # legacy `pytest.raises(ValueError, match="<legacy substring>")`
        # invocations keep matching.
        primary = self.diagnostics[0]
        super().__init__(primary.message)
