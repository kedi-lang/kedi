from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest

from marker_pattern.lang import compile_program, parse_program
from tests.utils_llm import llm_retry


def _iter_marker_files() -> Iterable[Path]:
    root = Path(__file__).resolve().parent / "markers"
    return sorted(p for p in root.glob("*.marker") if p.is_file())


requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; real LLM tests are skipped",
)


@pytest.mark.llm
@requires_groq
@pytest.mark.parametrize(
    "marker_path", list(_iter_marker_files()), ids=lambda p: Path(p).name
)
@llm_retry(tries=3, delay_s=0.5)
def test_marker_programs(marker_path: Path) -> None:
    src = marker_path.read_text(encoding="utf-8")
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
