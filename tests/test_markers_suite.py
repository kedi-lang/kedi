from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Iterable

import pytest

from marker.lang import compile_program, parse_program
from marker.agent_adapter.adapters import PydanticAdapter, DSPyAdapter
from marker.cli import main as cli_main
from marker.utils import parse_args
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


def test_cli_test_and_eval(tmp_path: Path) -> None:
    srcp = tmp_path / "x.marker"
    valp = tmp_path / "x.val.marker"
    srcp.write_text("@f():\n  = ok\n\n= <f()>\n", encoding="utf-8")
    valp.write_text("@test: f:\n  > case: ok:\n    `assert f() == 'ok'`\n\n@eval: f:\n  > metric: s:\n    = `0.5`\n", encoding="utf-8")
    
    try:
        sys.argv = [str(srcp), "--test"]
        print(sys.argv, file=sys.stderr)
        cli_main.main(sys.argv)
    except SystemExit as e:
        assert e.code == 0
    
    try:
        sys.argv = [str(srcp), "--eval"]
        print(sys.argv, file=sys.stderr)
        cli_main.main(sys.argv)
    except SystemExit as e:
        assert e.code == 0

def test_parse_cli_args() -> None:
    args = [
        "--foo", "bar",
        "--retries", "5",
        "--temperature", "0.7",
        "--test",
        "--no-cache",
    ]
    
    assert parse_args(args) == {
        "foo": "bar",
        "retries": 5,
        "temperature": 0.7,
        "test": True,
        "no_cache": True,
    }