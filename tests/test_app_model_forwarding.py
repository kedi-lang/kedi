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
from kedi.lang.parser.diagnostics import KediParseError
from kedi.tests_runner import EvalMetricResult, EvalRunResult
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


def test_build_runtime_adapter_paths(monkeypatch: Any) -> None:
    seen: list[tuple[str, str | None]] = []

    class FakeDSPyAdapter:
        def __init__(self, *, model: str | None) -> None:
            seen.append(("dspy", model))

        def produce_sync(self) -> None:
            return None

    class FakePydanticAdapter:
        def __init__(self, *, model: str | None) -> None:
            seen.append(("pydantic", model))

        def produce_sync(self) -> None:
            return None

    class FakeLangChainAdapter:
        def __init__(self, *, model: str | None) -> None:
            seen.append(("langchain", model))

        def produce_sync(self) -> None:
            return None

    monkeypatch.setattr(app, "DSPyAdapter", FakeDSPyAdapter)
    monkeypatch.setattr(app, "PydanticAdapter", FakePydanticAdapter)
    monkeypatch.setattr(app, "LangChainAdapter", FakeLangChainAdapter)

    app._build_runtime_adapter("dspy", None)
    app._build_runtime_adapter("pydantic", None)
    app._build_runtime_adapter("langchain", None)

    assert seen == [
        ("dspy", "groq/qwen/qwen3-32b"),
        ("pydantic", None),
        ("langchain", "groq:qwen/qwen3-32b"),
    ]


def test_pydantic_runtime_adapter_does_not_default_to_groq(monkeypatch: Any) -> None:
    seen: list[str | None] = []

    class FakePydanticAdapter:
        def __init__(self, *, model: str | None) -> None:
            seen.append(model)

    monkeypatch.setattr(app, "PydanticAdapter", FakePydanticAdapter)

    app._build_runtime_adapter("pydantic", None)

    assert seen == [None]


def test_run_tests_and_evals_cli_codegen_failure_and_optimize_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    monkeypatch.setattr(app, "maybe_run_codegen", lambda *args, **kwargs: (False, None))
    assert app.run_tests_cli(str(source)) == (1, "Code generation failed")
    assert app.run_evals_cli(str(source)) == (1, "Code generation failed")

    monkeypatch.setattr(app, "run_optimization_cli", lambda *args, **kwargs: (7, "bad opt"))
    assert app.run_evals_cli(str(source), optimize=True) == (7, "bad opt")

    monkeypatch.setattr(app, "run_optimization_cli", lambda *args, **kwargs: (0, "ok opt"))
    monkeypatch.setattr(app, "maybe_run_codegen", lambda *args, **kwargs: (False, "bad codegen"))
    assert app.run_evals_cli(str(source), optimize=True) == (1, "bad codegen")


def _optimized_program() -> ProgramWithValidation:
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
    return ProgramWithValidation(procedures=[proc], evals=[suite])


def _patch_optimizer_services(
    monkeypatch: Any,
    *,
    program: ProgramWithValidation | None = None,
    train_error: Exception | None = None,
    test_error: Exception | None = None,
    optimizer_error: Exception | None = None,
    optimized: Any = "better prompt",
) -> dict[str, Any]:
    import kedi.optimization as optimization_module
    import kedi.optimization.services as services_module
    from kedi.optimization.config import OptimizerConfig

    state: dict[str, Any] = {"deleted": False, "configured": []}
    program = program or _optimized_program()

    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: SimpleNamespace(text="source", source_map=None),
    )
    monkeypatch.setattr(app, "parse_program", lambda *args, **kwargs: program)
    monkeypatch.setattr(app, "compile_program", lambda *args, **kwargs: object())
    monkeypatch.setattr(app, "PydanticAdapter", lambda *, model: ("pydantic", model))
    monkeypatch.setattr(
        OptimizerConfig,
        "from_cli_options",
        classmethod(lambda cls, **kwargs: cls(**kwargs)),
    )

    class FakeDataLoader:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def load_training_data(self, suite: EvalSuite, dataset_name: str) -> list[str]:
            if train_error is not None:
                raise train_error
            return ["train"]

        def load_test_data(self, suite: EvalSuite, dataset_name: str) -> list[str]:
            if test_error is not None:
                raise test_error
            return ["test"]

    class FakeTypeResolver:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def resolve_param_types(self, proc: ProcedureDef) -> dict[str, type[str]]:
            return {"question": str}

    class FakeMetricFactory:
        def __init__(self, rt: object) -> None:
            self.rt = rt

        def create_metric(
            self,
            metric: EvalMetric,
            proc: ProcedureDef,
            dataset_name: str,
        ) -> object:
            state["metric_dataset"] = dataset_name
            return lambda *args, **kwargs: 1.0

    class FakeResultWriter:
        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def delete_existing_optimized_files(self) -> None:
            state["deleted"] = True

        def write_optimized_prompts(self, optimized_dict: dict[str, object]) -> Path:
            state["optimized_dict"] = optimized_dict
            return self.source_path.with_suffix(".optimized.kedi")

        def write_training_scores(self, train_scores: dict[str, float]) -> None:
            state["train_scores"] = train_scores

        def format_prompts_for_display(
            self,
            optimized_dict: dict[str, object],
            optimized_path: Path,
        ) -> str:
            return f"wrote {optimized_path.name}: {optimized_dict}"

    class FakePromptLoader:
        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def has_optimized_prompts(self) -> bool:
            return True

        def inject_optimized_prompts(self, prog: ProgramWithValidation) -> None:
            state["prompt_loader_injected"] = prog

    class FakeOptimizer:
        def optimize_procedure(self, **kwargs: Any) -> tuple[Any, float]:
            state["optimizer_kwargs"] = kwargs
            if optimizer_error is not None:
                raise optimizer_error
            return optimized, 0.5

    monkeypatch.setattr(services_module, "OptimizationDataLoader", FakeDataLoader)
    monkeypatch.setattr(services_module, "TypeResolver", FakeTypeResolver)
    monkeypatch.setattr(services_module, "MetricFactory", FakeMetricFactory)
    monkeypatch.setattr(services_module, "OptimizationResultWriter", FakeResultWriter)
    monkeypatch.setattr(optimization_module, "get_optimizer", lambda name: FakeOptimizer())

    import kedi.optimization.services.prompt_loader as prompt_loader_module

    monkeypatch.setattr(prompt_loader_module, "OptimizedPromptLoader", FakePromptLoader)
    return state


def test_run_optimization_cli_success_and_service_error_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    state = _patch_optimizer_services(monkeypatch)
    code, msg = app.run_optimization_cli(str(source), optimizer="mock")

    assert code == 0
    assert "wrote prog.optimized.kedi" in msg
    assert state["prompt_loader_injected"] is not None
    assert state["metric_dataset"] == "train"
    assert state["optimized_dict"] == {"solve": "better prompt"}
    assert state["train_scores"] == {"solve": 0.5}

    _patch_optimizer_services(monkeypatch, train_error=RuntimeError("bad train"))
    assert app.run_optimization_cli(str(source), optimizer="mock") == (1, "bad train")

    _patch_optimizer_services(monkeypatch, optimizer_error=RuntimeError("bad optimizer"))
    assert app.run_optimization_cli(str(source), optimizer="mock") == (
        1,
        "Optimization error for 'solve': bad optimizer",
    )


def test_run_optimization_cli_validation_adapter_and_fresh_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import kedi.optimization as optimization_module
    import kedi.optimization.services.dspy_config as dspy_config_module

    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: SimpleNamespace(text="source", source_map=None),
    )
    monkeypatch.setattr(
        app,
        "parse_program",
        lambda *args, **kwargs: ProgramWithValidation(procedures=[], evals=[]),
    )
    assert app.run_optimization_cli(str(source)) == (
        0,
        "No procedures with optimize spans found",
    )

    missing_eval = ProgramWithValidation(
        procedures=[
            ProcedureDef(name="solve", params=[], optimize_spans=[OptimizeSpan(name="prompt")])
        ],
        evals=[],
    )
    monkeypatch.setattr(app, "parse_program", lambda *args, **kwargs: missing_eval)
    assert app.run_optimization_cli(str(source))[1] == "Procedure 'solve' needs @eval: solve suite"

    missing_data = ProgramWithValidation(
        procedures=[
            ProcedureDef(name="solve", params=[], optimize_spans=[OptimizeSpan(name="prompt")])
        ],
        evals=[EvalSuite(procedure_name="solve", datasets={})],
    )
    monkeypatch.setattr(app, "parse_program", lambda *args, **kwargs: missing_data)
    assert "needs training data" in app.run_optimization_cli(str(source))[1]

    no_metrics = _optimized_program()
    no_metrics.evals[0].metrics = []
    _patch_optimizer_services(monkeypatch, program=no_metrics)
    assert app.run_optimization_cli(str(source))[1] == (
        "Procedure 'solve' needs at least one metric in eval suite"
    )

    bad_metric_dataset = _optimized_program()
    bad_metric_dataset.evals[0].metrics[0].dataset_name = "missing"
    _patch_optimizer_services(monkeypatch, program=bad_metric_dataset)
    assert "references dataset 'missing'" in app.run_optimization_cli(str(source))[1]

    monkeypatch.setattr(optimization_module, "get_optimizer", lambda name: None)
    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(optimization_module, "get_optimizer", lambda name: None)
    assert app.run_optimization_cli(str(source), optimizer="unknown") == (
        1,
        "Unknown optimizer: unknown",
    )

    class FailingDSPyAdapter:
        def __init__(self, *, model: str) -> None:
            raise ImportError("dspy unavailable")

    monkeypatch.setattr(app, "DSPyAdapter", FailingDSPyAdapter)
    _patch_optimizer_services(monkeypatch)
    assert (
        "DSPy is required for dspy adapter"
        in app.run_optimization_cli(
            str(source),
            adapter="dspy",
        )[1]
    )

    class RaisingDSPyConfigurationService:
        def __init__(self, config: object) -> None:
            raise ImportError("missing dspy")

    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(
        dspy_config_module, "DSPyConfigurationService", RaisingDSPyConfigurationService
    )
    assert (
        "DSPy is required for GEPA optimizer"
        in app.run_optimization_cli(
            str(source),
            optimizer="gepa",
        )[1]
    )

    state = _patch_optimizer_services(monkeypatch)
    checkpoint = app._gepa_checkpoint_root(source)
    checkpoint.mkdir()
    code, _ = app.run_optimization_cli(str(source), optimizer_fresh=True)
    assert code == 0
    assert state["deleted"] is True
    assert not checkpoint.exists()

    state = _patch_optimizer_services(monkeypatch)
    code, _ = app.run_optimization_cli(str(source), optimizer_fresh=True)
    assert code == 0
    assert state["deleted"] is True

    class FalsePromptLoader:
        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def has_optimized_prompts(self) -> bool:
            return False

        def inject_optimized_prompts(self, prog: ProgramWithValidation) -> None:
            raise AssertionError("should not inject")

    import kedi.optimization.services.prompt_loader as prompt_loader_module

    state = _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(prompt_loader_module, "OptimizedPromptLoader", FalsePromptLoader)
    code, _ = app.run_optimization_cli(str(source))
    assert code == 0
    assert "prompt_loader_injected" not in state

    metric_dataset = _optimized_program()
    metric_dataset.evals[0].metrics[0].dataset_name = "train"
    _patch_optimizer_services(monkeypatch, program=metric_dataset)
    assert app.run_optimization_cli(str(source))[0] == 0


def test_dspy_adapter_error_paths(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    class FailingDSPyAdapter:
        def __init__(self, *, model: str) -> None:
            raise ImportError("dspy unavailable")

    class FakeDSPyConfigurationService:
        def __init__(self, config: object) -> None:
            self.config = config

        def configure_main_lm(self, model: str) -> None:
            assert model == "openai/gpt-4o"

    import kedi.optimization.services.dspy_config as dspy_config_module

    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(
        dspy_config_module, "DSPyConfigurationService", FakeDSPyConfigurationService
    )
    monkeypatch.setattr(app, "DSPyAdapter", FailingDSPyAdapter)
    assert (
        "DSPy is required for GEPA optimizer"
        in app.run_optimization_cli(str(source), optimizer="gepa")[1]
    )

    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(app, "DSPyAdapter", FailingDSPyAdapter)
    assert (
        "DSPy is required for dspy adapter"
        in app.run_optimization_cli(str(source), optimizer="mock", adapter="dspy")[1]
    )


def test_run_optimization_cli_gepa_and_dspy_success_edges(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import kedi.optimization.services.dspy_config as dspy_config_module

    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    class FakeDSPyConfigurationService:
        def __init__(self, config: object) -> None:
            self.config = config

        def configure_main_lm(self, model: str) -> None:
            assert model == "openai/gpt-4o"

    class FakeDSPyAdapter:
        def __init__(self, *, model: str) -> None:
            assert model in {"openai/gpt-4o", "custom-dspy-model"}

    class FakeGEPAOptimizer:
        def __init__(
            self,
            *,
            config: object,
            runtime: object,
            source_path: Path,
            reporter: object | None = None,
        ) -> None:
            self.config = config
            self.runtime = runtime
            self.source_path = source_path
            self.reporter = reporter

        def optimize_procedure(self, **kwargs: Any) -> tuple[str, float]:
            return "gepa prompt", 0.75

    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(
        dspy_config_module, "DSPyConfigurationService", FakeDSPyConfigurationService
    )
    monkeypatch.setattr(app, "DSPyAdapter", FakeDSPyAdapter)
    monkeypatch.setitem(
        __import__("sys").modules,
        "kedi.optimization.gepa",
        SimpleNamespace(GEPAOptimizer=FakeGEPAOptimizer),
    )

    code, msg = app.run_optimization_cli(str(source), optimizer="gepa")
    assert code == 0
    assert "gepa prompt" in msg

    _patch_optimizer_services(monkeypatch)
    monkeypatch.setattr(app, "DSPyAdapter", FakeDSPyAdapter)
    code, msg = app.run_optimization_cli(
        str(source),
        optimizer="mock",
        adapter="dspy",
        adapter_model="custom-dspy-model",
    )
    assert code == 0
    assert "better prompt" in msg


def test_run_optimization_cli_missing_dataset_after_truthy_validation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    class TruthyEmptyDatasets(dict[str, EvalDataset]):
        def __bool__(self) -> bool:
            return True

    program = _optimized_program()
    program.evals[0].datasets = TruthyEmptyDatasets()
    _patch_optimizer_services(monkeypatch, program=program)

    assert app.run_optimization_cli(str(source)) == (
        1,
        "Procedure 'solve' needs training data",
    )


def test_maybe_codegen_and_run_program_cli_paths(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: SimpleNamespace(text="source", source_map=None),
    )
    monkeypatch.setattr(
        app,
        "parse_program",
        lambda *args, **kwargs: ProgramWithValidation(procedures=[], evals=[]),
    )
    assert app.maybe_run_codegen(str(source)) == (True, None)

    needs_codegen = ProgramWithValidation(
        procedures=[ProcedureDef(name="generated", params=[], ai_prompt="build it")],
    )
    monkeypatch.setattr(app, "parse_program", lambda *args, **kwargs: needs_codegen)
    monkeypatch.setattr(app, "run_codegen", lambda *args, **kwargs: False)
    assert app.maybe_run_codegen(str(source)) == (
        False,
        "Code generation failed to produce passing tests.",
    )
    monkeypatch.setattr(app, "run_codegen", lambda *args, **kwargs: True)
    assert app.maybe_run_codegen(str(source), agent="mock", retries=1, use_cache=False) == (
        True,
        None,
    )

    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: (_ for _ in ()).throw(RuntimeError("bad load")),
    )
    assert app.maybe_run_codegen(str(source)) == (False, "bad load")
    monkeypatch.setattr(
        app,
        "load_kedi_source_bundle",
        lambda source_path: SimpleNamespace(text="source", source_map=None),
    )
    monkeypatch.setattr(
        app,
        "parse_program",
        lambda *args, **kwargs: (_ for _ in ()).throw(KediParseError([])),
    )
    try:
        app.maybe_run_codegen(str(source))
    except KediParseError:
        pass
    else:
        raise AssertionError("KediParseError should be re-raised")

    import kedi.optimization.services.program_loader as program_loader_module

    class FakeRuntime:
        def __init__(self, outputs: list[Any]) -> None:
            self.outputs = outputs
            self.initial_globals: dict[str, Any] | None = None

        def set_initial_globals(self, values: dict[str, Any]) -> None:
            self.initial_globals = values

        def run_main(self) -> Any:
            value = self.outputs.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    # run_program_cli runs the program exactly once with a lazily-built adapter.
    runtimes = [
        FakeRuntime([None]),
        FakeRuntime(["with adapter"]),
    ]
    monkeypatch.setattr(
        program_loader_module,
        "load_program_with_optimized_prompts",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(app, "compile_program", lambda *args, **kwargs: runtimes.pop(0))
    monkeypatch.setattr(app, "_build_runtime_adapter", lambda adapter, model: object())
    monkeypatch.setenv("KEDI_DEBUG_OPTIMIZED", "1")

    assert app.run_program_cli(str(source), adapter="pydantic", model=None) == (0, "")
    assert app.run_program_cli(str(source), adapter="pydantic", model=None) == (
        0,
        "with adapter",
    )

    monkeypatch.setenv("KEDI_DEBUG_OPTIMIZED", "0")
    runtimes = [FakeRuntime([RuntimeError("boom")])]
    try:
        app.run_program_cli(str(source), adapter="pydantic", model=None)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("run_main errors should propagate")
