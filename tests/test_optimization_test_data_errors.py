"""optimization test_data load errors must fail the run.

``load_test_data`` returns None for an absent dataset but raises RuntimeError
on a real load failure. ``run_optimization_cli`` must let that surface as a
dataset error instead of silently dropping the validation data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kedi import app
from kedi.lang.ast import (
    EvalDataset,
    EvalMetric,
    EvalSuite,
    OptimizeSpan,
    ProcedureDef,
    ProgramWithValidation,
)


def _patch_optimizer_services(monkeypatch: Any, *, test_error: Exception | None) -> None:
    """Patch the optimizer service collaborators around run_optimization_cli.

    Only the test-data loader behaviour varies: ``test_error`` controls whether
    ``load_test_data`` raises (real load failure) or returns valid data.
    """
    import kedi.optimization as optimization_module
    import kedi.optimization.services as services_module
    import kedi.optimization.services.prompt_loader as prompt_loader_module
    from kedi.optimization.config import OptimizerConfig

    proc = ProcedureDef(
        name="solve",
        params=["question"],
        optimize_spans=[OptimizeSpan(name="prompt")],
    )
    suite = EvalSuite(
        procedure_name="solve",
        datasets={"train": EvalDataset(name="train", code="[]")},
        metrics=[EvalMetric(name="score", code="return 1.0")],
    )
    program = ProgramWithValidation(procedures=[proc], evals=[suite])

    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: SimpleNamespace(text="source", source_map=None),
    )
    monkeypatch.setattr(app, "parse_program", lambda *args, **kwargs: program)
    monkeypatch.setattr(app, "compile_program", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        app, "PydanticAdapter", lambda *, model, defer_model_check: ("pydantic", model)
    )
    monkeypatch.setattr(
        OptimizerConfig,
        "from_cli_options",
        classmethod(lambda cls, **kwargs: cls(**kwargs)),
    )

    class FakeDataLoader:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def load_training_data(self, suite: EvalSuite, dataset_name: str) -> list[str]:
            return ["train"]

        def load_test_data(self, suite: EvalSuite, dataset_name: str) -> list[str] | None:
            if test_error is not None:
                raise test_error
            return None

    class FakeTypeResolver:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def resolve_param_types(self, proc: ProcedureDef) -> dict[str, type[str]]:
            return {"question": str}

    class FakeMetricFactory:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def create_metric(self, metric: EvalMetric, proc: ProcedureDef, name: str) -> object:
            return lambda *args, **kwargs: 1.0

    class FakeResultWriter:
        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def write_optimized_prompts(self, optimized_dict: dict[str, object]) -> Path:
            return self.source_path.with_suffix(".optimized.kedi")

        def write_training_scores(self, train_scores: dict[str, float]) -> None:
            pass

        def format_prompts_for_display(self, optimized_dict: object, path: Path) -> str:
            return "ok"

    class FakePromptLoader:
        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def has_optimized_prompts(self) -> bool:
            return False

        def inject_optimized_prompts(self, prog: ProgramWithValidation) -> None:
            pass

    class FakeOptimizer:
        def optimize_procedure(self, **kwargs: Any) -> tuple[str, float]:
            return "better prompt", 0.5

    monkeypatch.setattr(services_module, "OptimizationDataLoader", FakeDataLoader)
    monkeypatch.setattr(services_module, "TypeResolver", FakeTypeResolver)
    monkeypatch.setattr(services_module, "MetricFactory", FakeMetricFactory)
    monkeypatch.setattr(services_module, "OptimizationResultWriter", FakeResultWriter)
    monkeypatch.setattr(prompt_loader_module, "OptimizedPromptLoader", FakePromptLoader)
    monkeypatch.setattr(optimization_module, "get_optimizer", lambda name: FakeOptimizer())


def test_run_optimization_cli_fails_on_broken_test_data(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    _patch_optimizer_services(
        monkeypatch,
        test_error=RuntimeError("Error loading test data 'cases': boom"),
    )

    code, msg = app.run_optimization_cli(str(source), optimizer="mock")

    assert code == 1
    assert "Test data load error for 'train'" in msg
    assert "boom" in msg


def test_run_optimization_cli_succeeds_when_test_data_absent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    # Absent test data (load_test_data returns None) is fine: the run proceeds.
    _patch_optimizer_services(monkeypatch, test_error=None)

    code, _ = app.run_optimization_cli(str(source), optimizer="mock")

    assert code == 0
