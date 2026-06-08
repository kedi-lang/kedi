"""Shared fixtures for LSP feature edge-case tests."""

from __future__ import annotations

from pathlib import Path

from kedi.lang.ast import Span


def make_edge_span(
    tmp_path: Path,
    *,
    line: int = 1,
    start_col: int = 1,
    end_col: int = 80,
) -> Span:
    return Span(
        start_byte=0,
        end_byte=max(end_col - start_col, 0),
        start_line=line,
        start_col=start_col,
        end_line=line,
        end_col=end_col,
        source_path=str(tmp_path / "x.kedi"),
    )
