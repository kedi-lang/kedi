from __future__ import annotations

from pathlib import Path

from kedi import app
from kedi.tests_runner import (
    EvalMetricResult,
    EvalRunResult,
)
from kedi.tests_runner import TestCaseResult as CaseResult
from kedi.tests_runner import TestRunResult as RunResult


def test_run_evals_cli_forwards_selected_adapter_model(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")

    sentinel = object()
    seen: dict[str, object] = {}

    def fake_codegen(*args, **kwargs):
        return True, None

    def fake_build(adapter: str, model: str | None):
        seen["adapter"] = adapter
        seen["model"] = model
        return sentinel

    def fake_run_evals(src_main: str, *, adapter=None):
        seen["run_adapter"] = adapter
        return [
            EvalRunResult(
                procedure="solve",
                metrics=[EvalMetricResult(name="score", score=1.0, feedback=None)],
            )
        ]

    monkeypatch.setattr(app, "maybe_run_codegen", fake_codegen)
    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)
    monkeypatch.setattr(app, "run_evals", fake_run_evals)

    code, out = app.run_evals_cli(
        str(src),
        adapter="pydantic",
        adapter_model="openrouter/google/gemini-3-flash-preview",
    )

    assert code == 0
    assert "Evals:" in out
    assert seen["adapter"] == "pydantic"
    assert seen["model"] == "openrouter/google/gemini-3-flash-preview"
    assert seen["run_adapter"] is sentinel


def test_run_tests_cli_forwards_selected_adapter_model(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")

    sentinel = object()
    seen: dict[str, object] = {}

    def fake_codegen(*args, **kwargs):
        return True, None

    def fake_build(adapter: str, model: str | None):
        seen["adapter"] = adapter
        seen["model"] = model
        return sentinel

    def fake_run_tests(src_main: str, *, adapter=None):
        seen["run_adapter"] = adapter
        return [RunResult(procedure="solve", cases=[CaseResult(name="basic", success=True)])]

    monkeypatch.setattr(app, "maybe_run_codegen", fake_codegen)
    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)
    monkeypatch.setattr(app, "run_tests", fake_run_tests)

    code, out = app.run_tests_cli(
        str(src),
        adapter="dspy",
        model="openrouter/google/gemini-3-flash-preview",
    )

    assert code == 0
    assert "Tests:" in out
    assert seen["adapter"] == "dspy"
    assert seen["model"] == "openrouter/google/gemini-3-flash-preview"
    assert seen["run_adapter"] is sentinel
