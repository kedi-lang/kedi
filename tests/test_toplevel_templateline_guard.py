from __future__ import annotations

import pytest

from kedi.lang.ast import Program, ReturnStmt, TemplateLine, TextSegment
from kedi.lang.compiler import compile_program


def test_toplevel_templateline_is_rejected() -> None:
    """an impossible top-level TemplateLine must not be a silent no-op."""
    prog = Program(
        toplevel_body=[TemplateLine(segments=[TextSegment(text="ignored")])],
        toplevel_return=ReturnStmt(segments=[TextSegment(text="done")]),
    )
    rt = compile_program(prog, adapter=None)

    with pytest.raises(ValueError, match="Top-level TemplateLine is not a valid program node"):
        rt.run_main()
