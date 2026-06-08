from __future__ import annotations

from ..lang.ast import Span

__all__ = ["name_token_span"]


def name_token_span(loc: Span, name: str, *, prefix_len: int) -> Span:
    """Span covering ``name`` after a single-byte prefix (e.g. ``<`` or ``[``)."""
    name_start_col = loc.start_col + prefix_len
    name_len = len(name)
    return Span(
        start_byte=loc.start_byte + prefix_len,
        end_byte=loc.start_byte + prefix_len + name_len,
        start_line=loc.start_line,
        start_col=name_start_col,
        end_line=loc.start_line,
        end_col=name_start_col + name_len,
        source_path=loc.source_path,
    )
