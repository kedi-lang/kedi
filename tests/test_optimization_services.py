"""Comprehensive tests for optimization services."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import dspy
import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import EvalSuite, ProcedureDef
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.services import (
    DSPyConfigurationService,
    DSPyExampleFormatter,
    DSPyModuleBuilder,
    DSPyPromptExtractor,
    DSPyScoreCalculator,
    GEPAMetricWrapper,
    MetricFactory,
    OptimizationDataLoader,
    OptimizationResultWriter,
    StructuredTypeHandler,
    TypeResolver,
)
from kedi.optimization.services.program_loader import load_program_with_optimized_prompts
from kedi.optimization.utils import extract_output_fields
from tests.optimization_helpers import parse_validation_program

# ==================== OptimizationDataLoader Tests ====================


def test_data_loader_single_param():
    """Test loading data with single parameter."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [(1, 2), (2, 4), (3, 6)]
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]
    train_data = loader.load_training_data(eval_suite, "cases")

    assert len(train_data) == 3
    assert train_data[0] == (1, 2)
    assert train_data[1] == (2, 4)
    assert train_data[2] == (3, 6)


def test_data_loader_multi_param():
    """Test loading data with multiple parameters."""
    src = """
@add(a: int, b: int) -> int:
  = `a + b`

@eval: add:
  > data: cases:
    = ```
    return [((1, 2), 3), ((2, 3), 5), ((3, 4), 7)]
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]
    train_data = loader.load_training_data(eval_suite, "cases")

    assert len(train_data) == 3
    assert train_data[0] == ((1, 2), 3)


def test_data_loader_test_data():
    """Test loading test data separately."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [(1, 2)]
    ```
  > test_data: test_cases:
    = ```
    return [(4, 8), (5, 10)]
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    test_data = loader.load_test_data(eval_suite, "cases")

    assert len(train_data) == 1
    assert test_data is None  # test_data uses different name

    test_data = loader.load_test_data(eval_suite, "test_cases")
    assert test_data is not None
    assert len(test_data) == 2


def test_data_loader_dict_format():
    """Test loading data from dictionary."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return {"a": 2, "b": 4, "c": 6}
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]
    train_data = loader.load_training_data(eval_suite, "cases")

    assert len(train_data) == 3
    assert all(isinstance(item, tuple) for item in train_data)


def test_data_loader_errors_inline_data_and_tuple_normalization():
    src = """
@test(x: int) -> int:
  = `x`

@eval: test:
  > data: inline_cases:
    = `[(1, 2), (3, 4, 5), 6]`
  > data: broken:
    = `1 / 0`
  > test_data: inline_tests:
    = `[(7, 8)]`
  > test_data: broken_tests:
    = `1 / 0`
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)
    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    assert loader.load_training_data(eval_suite, "inline_cases") == [
        (1, 2),
        (3, None),
        (6, None),
    ]
    assert loader.load_training_data(eval_suite) == [(1, 2), (3, None), (6, None)]
    assert loader.load_test_data(eval_suite, "inline_tests") == [(7, 8)]
    assert loader.load_test_data(EvalSuite("test"), "missing") is None

    with pytest.raises(ValueError, match="No training datasets"):
        loader.load_training_data(EvalSuite("test"))
    with pytest.raises(ValueError, match="Dataset 'missing' not found"):
        loader.load_training_data(eval_suite, "missing")
    with pytest.raises(RuntimeError, match="Error loading training data 'broken'"):
        loader.load_training_data(eval_suite, "broken")
    with pytest.raises(RuntimeError, match="Error loading test data 'broken_tests'"):
        loader.load_test_data(eval_suite, "broken_tests")
    with pytest.raises(ValueError, match="must return an iterable"):
        loader._convert_to_list("abc", "letters")
    with pytest.raises(ValueError, match="must return an iterable"):
        loader._convert_to_list(123, "number")


# ==================== MetricFactory Tests ====================


def test_metric_factory_simple_metric():
    """Test creating a simple metric function."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [(1, 2)]
    ```
  > metric: check(cases):
    = ```
    # cases is the input value, expected is available separately
    result = test(cases)
    return (1.0 if result == expected else 0.0, None)
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    factory = MetricFactory(rt)
    eval_suite = prog.evals[0]
    proc = prog.procedures[0]
    metric_def = eval_suite.metrics[0]

    metric_fn = factory.create_metric(metric_def, proc, "cases")

    # Test metric function - pass the input value and expected output
    score, feedback = metric_fn(1, 2)
    assert score == 1.0
    assert feedback is None

    score, feedback = metric_fn(1, 3)
    assert score == 0.0


def test_metric_factory_with_feedback():
    """Test metric that returns feedback."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [(1, 2)]
    ```
  > metric: check(cases):
    = ```
    result = test(cases)
    if result == expected:
        return (1.0, None)
    else:
        return (0.0, f"Expected {expected}, got {result}")
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    factory = MetricFactory(rt)
    eval_suite = prog.evals[0]
    proc = prog.procedures[0]
    metric_def = eval_suite.metrics[0]

    metric_fn = factory.create_metric(metric_def, proc, "cases")

    score, feedback = metric_fn(1, 3)
    assert score == 0.0
    assert feedback == "Expected 3, got 2"


def test_metric_factory_bool_return():
    """Test metric that returns boolean."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [(1, 2)]
    ```
  > metric: check(cases):
    = ```
    result = test(cases)
    return result == expected
    ```
"""
    prog = parse_validation_program(src)
    rt = compile_program(prog, adapter=None)

    factory = MetricFactory(rt)
    eval_suite = prog.evals[0]
    proc = prog.procedures[0]
    metric_def = eval_suite.metrics[0]

    metric_fn = factory.create_metric(metric_def, proc, "cases")

    score, feedback = metric_fn(1, 2)
    assert score == 1.0

    score, feedback = metric_fn(1, 3)
    assert score == 0.0


# ==================== TypeResolver Tests ====================


def test_type_resolver_basic_types():
    """Test resolving basic Python types."""
    src = """
@test(x: int, y: str) -> float:
  = `float(x) + len(y)`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    resolver = TypeResolver(rt)
    proc = prog.procedures[0]
    param_types = resolver.resolve_param_types(proc)

    assert "x" in param_types
    assert param_types["x"] is int


def test_type_resolver_no_annotations():
    """Test resolver handles missing type annotations."""
    src = """
@test(x, y) -> float:
  = `float(x) + len(y)`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    resolver = TypeResolver(rt)
    proc = prog.procedures[0]
    param_types = resolver.resolve_param_types(proc)

    # Should return empty dict if no annotations
    assert param_types == {}


def test_type_resolver_outputs_defaults_typing_constructs_and_failures():
    src = """
@test(items: list[int], missing: MissingType, raw) -> str:
  = ok
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)
    resolver = TypeResolver(rt)
    proc = prog.procedures[0]

    param_types = resolver.resolve_param_types(proc)
    assert "items" in param_types
    assert "missing" not in param_types
    assert "raw" not in param_types

    assert resolver.resolve_output_types([]) == {}
    outputs = resolver.resolve_output_types(
        [
            ("plain", None),
            ("count", "int"),
            ("items", "list[int]"),
            ("unknown", "MissingType"),
        ]
    )
    assert outputs["plain"] is str
    assert outputs["count"] is int
    assert getattr(outputs["items"], "__origin__", None) is list
    assert outputs["unknown"] is str


def test_type_resolver_direct_fallback_paths(monkeypatch):
    rt = compile_program(parse_program("= ok"), adapter=None)
    resolver = TypeResolver(rt)

    assert resolver.resolve_param_types(ProcedureDef("plain", ["x"])) == {}

    def fail_resolve(self: Any, runtime: Any, env: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("kedi.optimization.services.type_resolver.KediType.resolve", fail_resolve)

    proc = ProcedureDef("broken", ["x"], param_type_annotations={"x": "int"})
    assert resolver.resolve_param_types(proc) == {}
    assert resolver.resolve_output_types([("value", "int")]) == {"value": str}

    def resolve_to_value(self: Any, runtime: Any, env: Any) -> object:
        return object()

    monkeypatch.setattr(
        "kedi.optimization.services.type_resolver.KediType.resolve",
        resolve_to_value,
    )

    assert resolver.resolve_output_types([("value", "int")]) == {"value": str}


# ==================== OptimizationResultWriter Tests ====================


def test_result_writer_create_files():
    """Test writing optimized prompts and scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text("# Test file")

        writer = OptimizationResultWriter(source_path)

        # Write optimized prompts
        optimized_prompts = {
            "procedure1": {
                "span1": "Optimized prompt 1",
                "span2": "Optimized prompt 2",
            }
        }
        optimized_path = writer.write_optimized_prompts(optimized_prompts)

        assert optimized_path.exists()
        content = optimized_path.read_text()
        assert "Optimized prompt 1" in content

        # Write scores
        train_scores = {"procedure1": 0.95}
        writer.write_training_scores(train_scores)

        scores_path = source_path.with_suffix(source_path.suffix + ".optimized_scores.json")
        assert scores_path.exists()


def test_result_writer_merges_scores_and_recovers_from_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text("# Test")
        scores_path = source_path.with_suffix(source_path.suffix + ".optimized_scores.json")
        scores_path.write_text("{bad json", encoding="utf-8")

        writer = OptimizationResultWriter(source_path)
        writer.write_training_scores({"fresh": 0.5})

        assert json.loads(scores_path.read_text(encoding="utf-8")) == {"fresh": 0.5}

        writer.write_training_scores({"other": 0.75})

        assert json.loads(scores_path.read_text(encoding="utf-8")) == {
            "fresh": 0.5,
            "other": 0.75,
        }


def test_result_writer_format_display():
    """Test formatting prompts for display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text("# Test")

        writer = OptimizationResultWriter(source_path)

        optimized_prompts = {"proc1": {"span1": "Prompt 1\nLine 2"}}

        formatted = writer.format_prompts_for_display(
            optimized_prompts,
            source_path.with_suffix(source_path.suffix + ".optimized.json"),
        )

        assert "proc1" in formatted
        assert "Prompt 1" in formatted


def test_result_writer_delete_existing():
    """Test deleting existing optimized files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text("# Test")

        writer = OptimizationResultWriter(source_path)

        # Create existing files with new JSON format
        opt_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
        scores_path = source_path.with_suffix(source_path.suffix + ".optimized_scores.json")
        opt_path.write_text("{}")
        scores_path.write_text("{}")

        assert opt_path.exists()
        assert scores_path.exists()

        writer.delete_existing_optimized_files()

        assert not opt_path.exists()
        assert not scores_path.exists()

        writer.delete_existing_optimized_files()


def test_program_loader_preserves_output_fields_with_optimized_prefix():
    """Loading optimized prompts must not drop later template lines.

    Multi-line optimize spans often put output placeholders on later lines.
    Those placeholders must survive prompt injection so DSPy can still
    extract a valid signature for GEPA.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text(
            """
@solve(x: str) -> int:
  > optimize: solver:
    Input: <x>
    Reasoning: [reasoning: str]
    Answer: [answer: int]
  = `answer`
""".strip()
        )

        optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
        optimized_path.write_text(
            """
{
  "solve": {
    "solver": "Use careful reasoning before answering."
  }
}
""".strip()
        )

        program = load_program_with_optimized_prompts(source_path)
        proc = next(p for p in program.procedures if p.name == "solve")
        span = proc.optimize_spans[0]

        assert len(span.template_lines) == 3
        assert (
            span.template_lines[0].optimized_prompt_prefix
            == "Use careful reasoning before answering."
        )

        outputs = extract_output_fields(span)
        assert outputs == [("reasoning", "str"), ("answer", "int")]


def test_program_loader_normalizes_polluted_optimized_prompt():
    """Historical full-instruction saves should be normalized back to prefix form."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text(
            """
@solve(x: str) -> int:
  > optimize: solver:
    Input: <x>
    Reasoning: [reasoning: str]
    Answer: [answer: int]
  = `answer`
""".strip()
        )

        base = "Input: <x>\nReasoning: [reasoning: str]\nAnswer: [answer: int]"
        polluted = "Use exact IDs only.\n" + base + "\n" + base

        optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
        optimized_path.write_text(
            json.dumps({"solve": {"solver": polluted}}),
            encoding="utf-8",
        )

        program = load_program_with_optimized_prompts(source_path)
        proc = next(p for p in program.procedures if p.name == "solve")
        span = proc.optimize_spans[0]

        assert span.template_lines[0].optimized_prompt_prefix == "Use exact IDs only."


def test_program_loader_debug_output_shows_before_and_after_injection() -> None:
    from kedi.reporter import BufferReporter

    reporter = BufferReporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text(
            """
@solve(x: str) -> int:
  > optimize: solver:
    Input: <x>
    Answer: [answer: int]
  = `answer`
""".strip()
        )
        optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
        optimized_path.write_text(
            json.dumps({"solve": {"solver": "Use a direct numeric answer."}}),
            encoding="utf-8",
        )

        program = load_program_with_optimized_prompts(source_path, debug=True, reporter=reporter)

        output = reporter.getvalue()
        assert "[DEBUG] Checking for optimized prompts: True" in output
        assert "[DEBUG] BEFORE injection - solve::solver:" in output
        assert "[DEBUG] AFTER injection - solve::solver:" in output
        assert program.procedures[0].optimize_spans[0].template_lines[
            0
        ].optimized_prompt_prefix == ("Use a direct numeric answer.")


def test_program_loader_debug_output_without_optimized_prompts() -> None:
    from kedi.reporter import BufferReporter

    reporter = BufferReporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text(
            """
@solve(x: str) -> str:
  = <x>
""".strip()
        )

        load_program_with_optimized_prompts(source_path, debug=True, reporter=reporter)

        output = reporter.getvalue()
        assert "[DEBUG] Checking for optimized prompts: False" in output
        assert "BEFORE injection" not in output
        assert "AFTER injection" not in output


def test_program_loader_debug_output_with_stale_optimized_prompts() -> None:
    from kedi.reporter import BufferReporter

    reporter = BufferReporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test.kedi"
        source_path.write_text(
            """
@solve(x: str) -> str:
  = <x>
""".strip()
        )
        optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
        optimized_path.write_text(
            json.dumps({"solve": {"old_span": "Use stale prompt."}}),
            encoding="utf-8",
        )

        load_program_with_optimized_prompts(source_path, debug=True, reporter=reporter)

        output = reporter.getvalue()
        assert "[DEBUG] Checking for optimized prompts: True" in output
        assert "[DEBUG] Optimized path:" in output
        assert "BEFORE injection" not in output
        assert "AFTER injection" not in output


# ==================== StructuredTypeHandler Tests ====================


def test_type_handler_simple_signature():
    """Test creating simple DSPy signature."""
    src = """
@test(x: int) -> int:
  = `x * 2`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    handler = StructuredTypeHandler(rt)

    inputs = ["x"]
    outputs = [("result", "int")]

    signature = handler.create_dspy_signature_sync(inputs, outputs, use_string_signature=True)

    assert isinstance(signature, str)
    assert "x" in signature
    assert "result" in signature


def test_type_handler_multi_output():
    """Test signature with multiple outputs."""
    src = "@test(x: int) -> int:\n  = `x * 2`"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    handler = StructuredTypeHandler(rt)

    inputs = ["x"]
    outputs = [("result1", "int"), ("result2", "str")]

    signature = handler.create_dspy_signature_sync(inputs, outputs, use_string_signature=True)

    assert "result1" in signature
    assert "result2" in signature


# ==================== DSPy-Dependent Service Tests ====================


def test_dspy_module_builder_basic():
    """Test building a basic DSPy module."""
    src = """
@test(x: int) -> int:
  > optimize: compute:
    Compute the result: <x>
    [result: int]
  = `result`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    builder = DSPyModuleBuilder(runtime=rt)
    proc = prog.procedures[0]
    span = proc.optimize_spans[0]

    module = builder.build_module(span, ["x"], {"x": int})

    assert module is not None
    assert hasattr(module, "predictor")


def test_dspy_example_formatter_single_param():
    """Test formatting examples with single parameter."""
    from kedi.lang import compile_program, parse_program

    # Create a minimal runtime for the formatter
    src = "@dummy(x: int) -> int:\n  = `x`"
    rt = compile_program(parse_program(src), adapter=None)
    formatter = DSPyExampleFormatter(runtime=rt)

    data = [(1, 2), (2, 4), (3, 6)]
    examples = formatter.format_examples(data, ["x"])

    assert len(examples) == 3
    assert examples[0].x == 1
    assert examples[0]._expected == 2


def test_dspy_example_formatter_multi_param():
    """Test formatting examples with multiple parameters."""
    from kedi.lang import compile_program, parse_program

    # Create a minimal runtime for the formatter
    src = "@dummy(a: int, b: int) -> int:\n  = `a + b`"
    rt = compile_program(parse_program(src), adapter=None)
    formatter = DSPyExampleFormatter(runtime=rt)

    data = [((1, 2), 3), ((2, 3), 5)]
    examples = formatter.format_examples(data, ["a", "b"])

    assert len(examples) == 2
    assert examples[0].a == 1
    assert examples[0].b == 2
    assert examples[0]._expected == 3


def test_gepa_metric_wrapper_basic():
    """Test wrapping a basic metric."""
    wrapper = GEPAMetricWrapper()

    def simple_metric(input_val, expected):
        return (1.0 if input_val * 2 == expected else 0.0, None)

    gepa_metric = wrapper.wrap_metric(simple_metric, ["x"])

    # Create mock Example and Prediction
    gold = dspy.Example(x=2, _expected=4).with_inputs("x")
    pred = dspy.Prediction(result=4)

    score = gepa_metric(gold, pred)

    assert score == 1.0 or (hasattr(score, "score") and score.score == 1.0)


def test_gepa_metric_wrapper_string_extraction():
    """Test metric wrapper with properly typed DSPy outputs."""
    wrapper = GEPAMetricWrapper()

    # Metric compares procedure output with expected output
    def simple_metric(input_val, expected):
        # In real usage, metric calls procedure and compares its output
        # For this test, we simulate by comparing expected with itself
        # (procedure override returns pred_output which equals expected)
        return (1.0 if expected == expected else 0.0, None)

    gepa_metric = wrapper.wrap_metric(simple_metric, ["x"])

    gold = dspy.Example(x=2, _expected=4).with_inputs("x")
    # DSPy returns properly typed outputs when using structured signatures
    pred = dspy.Prediction(result=4)

    result = gepa_metric(gold, pred)
    # pred_output (4) equals expected_output (4), so metric should return 1.0
    assert (hasattr(result, "score") and result.score == 1.0) or result == 1.0


def test_dspy_prompt_extractor():
    """Test extracting prompts from optimized module."""
    extractor = DSPyPromptExtractor()

    src = """
@test(x: int) -> int:
  > optimize: compute:
    Original prompt: <x>
    [result: int]
  = `result`
"""
    prog = parse_program(src)
    span = prog.procedures[0].optimize_spans[0]

    # Create a mock optimized module
    class MockModule(dspy.Module):
        def __init__(self):
            super().__init__()
            sig = cast(Any, dspy).Signature("x -> result")
            self.predictor = dspy.Predict(sig.with_instructions("Optimized prompt: <x>"))

    module = MockModule()
    prompts = extractor.extract_prompts(module, [span])

    assert span.name in prompts
    assert prompts[span.name] == "Optimized prompt: <x>"


def test_dspy_prompt_extractor_strips_repeated_seed_suffix():
    """Extractor should persist only the optimized prefix, not prefix+base(+base)."""
    extractor = DSPyPromptExtractor()

    src = """
@test(x: int) -> int:
  > optimize: compute:
    Original prompt: <x>
    [result: int]
  = `result`
"""
    prog = parse_program(src)
    span = prog.procedures[0].optimize_spans[0]
    base = "Original prompt: <x>\n[result: int]"

    class MockModule(dspy.Module):
        def __init__(self):
            super().__init__()
            sig = cast(Any, dspy).Signature("x -> result")
            self.predictor = dspy.Predict(
                sig.with_instructions(f"Use the exact identifier.\n{base}\n{base}")
            )

    module = MockModule()
    prompts = extractor.extract_prompts(module, [span])

    assert prompts[span.name] == "Use the exact identifier."


def test_dspy_score_calculator():
    """Test calculating training scores."""
    calculator = DSPyScoreCalculator()

    # Create simple module
    class SimpleModule(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(result=kwargs.get("x", 0) * 2)

    module = SimpleModule()

    # Create examples
    examples = [
        dspy.Example(x=1, _expected=2).with_inputs("x"),
        dspy.Example(x=2, _expected=4).with_inputs("x"),
    ]

    # Simple metric
    def metric(gold, pred):
        expected = gold._expected
        actual = pred.result if hasattr(pred, "result") else 0
        return 1.0 if expected == actual else 0.0

    score = calculator.calculate_score(module, examples, metric, sample_size=2)

    # Both examples should be correct
    assert score == 1.0


def test_dspy_config_service_basic():
    """Test DSPy configuration service."""
    config = OptimizerConfig(
        optimizer_model="openai/gpt-4o",
        optimizer_model_max_tokens=2000,
    )

    service = DSPyConfigurationService(config)

    # Should not raise
    try:
        service.configure_main_lm("openai/gpt-4o")
        # Verify configured
        assert dspy.settings.lm is not None
    except Exception as e:
        # Skip if model requires actual API keys
        if "API key" not in str(e) and "provider" not in str(e).lower():
            raise
