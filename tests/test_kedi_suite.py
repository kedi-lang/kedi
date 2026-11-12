from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.cli import main as cli_main
from kedi.lang import compile_program, parse_program
from kedi.utils import parse_args
from tests.mock_adapter import MockAdapter, WordChainMockAdapter
from tests.utils_llm import llm_retry

_FILES_DIR = Path(__file__).resolve().parent / "files"
_SCOPING_KEDI = _FILES_DIR / "scoping.kedi"
_WORD_CHAIN_KEDI = _FILES_DIR / "word_chain.kedi"


def _build_live_adapter() -> PydanticAdapter:
    # The recorded cassettes were captured against this model; keep it so they replay.
    model_name = os.environ.get("MODEL_NAME", os.environ.get("GROQ_MODEL", "groq:qwen/qwen3-32b"))
    return PydanticAdapter(model=model_name)


@pytest.mark.llm
@pytest.mark.vcr
@llm_retry(tries=3, delay_s=0.5)
def test_kedi_program_scoping() -> None:
    src = _SCOPING_KEDI.read_text(encoding="utf-8")
    rt = compile_program(
        parse_program(src, source_path=str(_SCOPING_KEDI)),
        adapter=_build_live_adapter(),
    )
    out = rt.run_main()
    assert isinstance(out, str)


def test_kedi_program_word_chain() -> None:
    src = _WORD_CHAIN_KEDI.read_text(encoding="utf-8")
    rt = compile_program(
        parse_program(src, source_path=str(_WORD_CHAIN_KEDI)),
        adapter=WordChainMockAdapter(),
    )
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.endswith("ee")


def test_cli_test_and_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.app as app_module

    def _mock_build_runtime_adapter(adapter: str, model: str | None) -> MockAdapter:
        return MockAdapter(default_model=model)

    monkeypatch.setattr(app_module, "_build_runtime_adapter", _mock_build_runtime_adapter)

    srcp = tmp_path / "x.kedi"
    valp = tmp_path / "x.val.kedi"
    srcp.write_text("@f():\n  = ok\n\n= <f()>\n", encoding="utf-8")
    valp.write_text(
        "@test: f:\n  > case: ok:\n    `assert f() == 'ok'`\n\n"
        "@eval: f:\n  > data: cases:\n    = `[1, 2, 3]`\n"
        "  > metric: s(cases):\n    = `1.0`\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    test_result = runner.invoke(cli_main, [str(srcp), "--test"])
    assert test_result.exit_code == 0, test_result.output
    assert "f::ok: OK" in test_result.output

    eval_result = runner.invoke(cli_main, [str(srcp), "--eval"])
    assert eval_result.exit_code == 0, eval_result.output
    assert "f::s: 1.0000" in eval_result.output


def test_parse_cli_args() -> None:
    args = [
        "--foo",
        "bar",
        "--retries",
        "5",
        "--temperature",
        "0.7",
        "--test",
        "--no-cache",
    ]

    assert parse_args(args) == {
        "foo": "bar",
        "retries": "5",
        "temperature": "0.7",
        "test": True,
        "no_cache": True,
    }
