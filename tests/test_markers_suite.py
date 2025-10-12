from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest

from marker.lang import compile_program, parse_program
from marker.agent_adapter.adapters import PydanticAdapter, DSPyAdapter
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
    # Build real adapter like CLI does
    adapter_name = os.environ.get("MARKER_ADAPTER", "pydantic").lower()
    model_name = os.environ.get("MODEL_NAME", os.environ.get("GROQ_MODEL", "groq:qwen/qwen3-32b"))
    adapter = DSPyAdapter(model=model_name) if adapter_name == "dspy" else PydanticAdapter(model=model_name)
    rt = compile_program(parse_program(src), adapter=adapter)
    out = rt.run_main()
    assert isinstance(out, str)
