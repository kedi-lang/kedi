"""Keep the published example executable and its scoring independently checked."""

from itertools import combinations
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program
from kedi.optimization.mock import MockOptimizer
from kedi.optimization.services import MetricFactory, OptimizationDataLoader, TypeResolver
from kedi.optimization.services.dspy_example_formatter import DSPyExampleFormatter
from kedi.optimization.services.dspy_module_builder import DSPyModuleBuilder
from kedi.tests_runner import run_evals

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "knapsack" / "main.kedi"


@pytest.fixture
def program():
    return parse_program(EXAMPLE.read_text(), source_path=str(EXAMPLE))


def test_demo_returns_typed_selection(program):
    model = TestModel(call_tools=[], custom_output_args={"selected": ["C", "D"]})
    runtime = compile_program(program, adapter=PydanticAdapter(model))
    assert runtime.run_main() == ["C", "D"]
    assert model.last_model_request_parameters is not None
    schema = model.last_model_request_parameters.output_tools[0].parameters_json_schema
    assert schema["properties"]["selected"]["items"]["type"] == "string"


def test_all_dataset_targets_are_optimal(program):
    runtime = compile_program(program, adapter=None)
    loader = OptimizationDataLoader(runtime)
    training = loader.load_training_data(program.evals[0], "problems")
    validation = loader.load_test_data(program.evals[0], "problems")
    assert len(training) == 4
    assert validation is not None and len(validation) == 2
    assert all(row not in training for row in validation)
    for (items, capacity), expected in training + validation:
        optimum = max(
            sum(value for weight, value in subset)
            for size in range(len(items) + 1)
            for subset in combinations(items.values(), size)
            if sum(weight for weight, value in subset) <= capacity
        )
        assert expected["value"] == optimum


@pytest.mark.parametrize(
    ("selected", "score", "feedback"),
    [
        (["C", "D"], 1.0, "value 18"),
        (["D", "C"], 1.0, "value 18"),
        (["D", "D"], 0.0, "at most once"),
        (["unknown"], 0.0, "Only these items"),
        (["C, D"], 0.0, "Unknown item"),
        (["['C', 'D']"], 0.0, "Unknown item"),
        (["A", "C", "D"], 0.0, "exceeds capacity"),
        (["A", "B", "D"], 0.0, "value 17"),
        ([], 0.0, "value 0"),
    ],
)
def test_metric_scores_feasibility_and_optimality(program, selected, score, feedback):
    runtime = compile_program(
        program,
        adapter=PydanticAdapter(
            TestModel(call_tools=[], custom_output_args={"selected": selected})
        ),
    )
    metric = MetricFactory(runtime).create_metric(
        program.evals[0].metrics[0], program.procedures[0], "problems"
    )
    actual, message = metric(
        ({"A": (2, 3), "B": (3, 4), "C": (4, 8), "D": (5, 10)}, 10), {"value": 18}
    )
    assert actual == score
    assert feedback in message


@pytest.mark.parametrize("selected", [["A", "B"], ["A", "C"], ["B", "C"]])
def test_metric_accepts_tied_optima(program, selected):
    runtime = compile_program(
        program,
        adapter=PydanticAdapter(
            TestModel(call_tools=[], custom_output_args={"selected": selected})
        ),
    )
    metric = MetricFactory(runtime).create_metric(
        program.evals[0].metrics[0], program.procedures[0], "problems"
    )
    score, _ = metric(({"A": (1, 1), "B": (1, 1), "C": (1, 1)}, 2), {"value": 2})
    assert score == 1.0


@pytest.mark.parametrize("items", [{}, {"A": (2, 3)}])
def test_metric_accepts_empty_optimum(program, items):
    runtime = compile_program(
        program,
        adapter=PydanticAdapter(TestModel(call_tools=[], custom_output_args={"selected": []})),
    )
    metric = MetricFactory(runtime).create_metric(
        program.evals[0].metrics[0], program.procedures[0], "problems"
    )
    assert metric((items, 0), {"value": 0})[0] == 1.0


def test_eval_uses_validation_cases_without_running_demo():
    answers = iter([["G", "I"], ["B", "C"]])
    requests = []

    def respond(messages, info):
        requests.append(messages)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"selected": next(answers)})]
        )

    results = run_evals(EXAMPLE.read_text(), adapter=PydanticAdapter(FunctionModel(respond)))
    assert len(results) == 1
    assert results[0].procedure == "solve_knapsack"
    assert results[0].metrics[0].name == "optimality"
    assert results[0].metrics[0].score == 1.0
    assert len(requests) == 2
    assert '{"G":[5,10]' in requests[0][-1].parts[0].content[0]
    assert "The selected item names are [selected" in str(requests[0])


def test_optimizer_can_consume_example_types_data_and_metric(program):
    runtime = compile_program(
        program,
        adapter=PydanticAdapter(TestModel(call_tools=[], custom_output_args={"selected": []})),
    )
    proc = program.procedures[0]
    suite = program.evals[0]
    loader = OptimizationDataLoader(runtime)
    module = DSPyModuleBuilder(runtime).build_module(
        proc.optimize_spans[0], proc.params, TypeResolver(runtime).resolve_param_types(proc)
    )
    assert module.predictor.signature.output_fields["selected"].annotation == list[str]
    examples = DSPyExampleFormatter(runtime).format_examples(
        loader.load_training_data(suite, "problems"), proc.params, proc.optimize_spans[0], proc.name
    )
    assert len(examples) == 4
    assert "'A': (2, 3)" in examples[0]["items"]
    assert examples[0]["capacity"] == "10"
    prompts, score = MockOptimizer().optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=TypeResolver(runtime).resolve_param_types(proc),
        spans=proc.optimize_spans,
        train_data=loader.load_training_data(suite, "problems"),
        test_data=loader.load_test_data(suite, "problems"),
        metric=MetricFactory(runtime).create_metric(suite.metrics[0], proc, "problems"),
    )
    assert set(prompts) == {"selection"}
    assert "[selected: list[str]]" in prompts["selection"]
    assert score == 0.0
