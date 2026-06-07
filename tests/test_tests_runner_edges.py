from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from kedi import tests_runner
from kedi.tests_runner import _load_program_with_validations, run_evals, run_tests
from kedi.utils import AttrDict


def test_load_program_with_validations_accepts_file_path(tmp_path: Path) -> None:
    source = tmp_path / "plain.kedi"
    source.write_text("@id(x):\n  = <x>\n", encoding="utf-8")

    runtime = _load_program_with_validations(str(source), adapter=None)

    assert runtime._get_proc("id")("ok") == "ok"


def test_run_tests_file_path_block_inline_failure_and_loader_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tests.kedi"
    source.write_text(
        "@id(x):\n"
        "  = <x>\n\n"
        "@test: id:\n"
        "  > case: block_ok:\n"
        "    ```\n"
        "    assert id('x') == 'x'\n"
        "    ```\n"
        "  > case: inline_bad:\n"
        "    `assert False, 'boom'`\n",
        encoding="utf-8",
    )

    results = run_tests(str(source), adapter=None)
    assert [case.success for case in results[0].cases] == [True, False]
    assert "boom" in (results[0].cases[1].error or "")

    import kedi.optimization.services.program_loader as program_loader

    monkeypatch.setattr(
        program_loader,
        "load_program_with_optimized_prompts",
        lambda _: (_ for _ in ()).throw(RuntimeError("optimized load failed")),
    )

    fallback_results_without_globals = run_tests(str(source), adapter=None)
    assert [case.success for case in fallback_results_without_globals[0].cases] == [True, False]

    fallback_results = run_tests(
        str(source),
        adapter=None,
        runtime_globals={"args": AttrDict({}, default=None)},
    )
    assert [case.success for case in fallback_results[0].cases] == [True, False]


def test_run_evals_file_path_loader_fallback_and_metric_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evals.kedi"
    source.write_text(
        "@identity(x):\n"
        "  = `x`\n\n"
        "@eval: identity:\n"
        "  > data: mapping:\n"
        "    = `{'train': 1}`\n"
        "  > test_data: mapping:\n"
        "    = ```\n"
        "    return {'test': 2}\n"
        "    ```\n"
        "  > metric: mapping_score(mapping):\n"
        "    = ```\n"
        "    key, value = mapping\n"
        "    return (1.0, f'{key}:{value}')\n"
        "    ```\n",
        encoding="utf-8",
    )

    import kedi.optimization.services.program_loader as program_loader

    monkeypatch.setattr(
        program_loader,
        "load_program_with_optimized_prompts",
        lambda _: (_ for _ in ()).throw(RuntimeError("optimized load failed")),
    )

    result_without_globals = run_evals(str(source), adapter=None)[0]
    metrics_without_globals = {metric.name: metric for metric in result_without_globals.metrics}
    assert metrics_without_globals["mapping_score"].score == 1.0

    result = run_evals(
        str(source),
        adapter=None,
        runtime_globals={"args": AttrDict({}, default=None)},
    )[0]
    metrics = {metric.name: metric for metric in result.metrics}

    assert metrics["mapping_score"].score == 1.0
    assert metrics["mapping_score"].feedback == "test:2"


def test_run_tests_and_evals_source_strings_receive_runtime_globals() -> None:
    cli_args = {"args": AttrDict({"id": "7"}, default=None)}
    test_source = (
        "@test: args:\n  > case: cli:\n    `assert args.id == '7' and args.missing is None`\n"
    )
    test_results = run_tests(test_source, adapter=None, runtime_globals=cli_args)
    assert test_results[0].cases[0].success

    eval_source = (
        "@eval: args:\n"
        "  > metric: cli:\n"
        "    = `1.0 if args.id == '7' and args.missing is None else 0.0`\n"
    )
    eval_results = run_evals(eval_source, adapter=None, runtime_globals=cli_args)
    assert eval_results[0].metrics[0].score == 1.0


def test_run_evals_legacy_metric_tuple_shapes_and_dataset_example_errors() -> None:
    legacy_tuple = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > metric: legacy_tuple:\n"
        "    = `(0.5, 'legacy feedback')`\n"
    )
    result = run_evals(legacy_tuple, adapter=None)[0].metrics[0]
    assert result.score == 0.5
    assert result.feedback == "legacy feedback"

    legacy_bad_tuple = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > metric: legacy_bad_tuple:\n"
        "    = `(1.0, 'bad', 'tuple')`\n"
    )
    result = run_evals(legacy_bad_tuple, adapter=None)[0].metrics[0]
    assert result.score == 0.0
    assert "Metric tuple result" in (result.feedback or "")

    dataset_errors = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > data: tuple_cases:\n"
        "    = `[((1, 2), 3), ('solo', None), 'pair']`\n"
        "  > metric: tuple_score(tuple_cases):\n"
        "    = ```\n"
        "    if tuple_cases == 'pair':\n"
        "        raise ValueError('pair exploded')\n"
        "    return True\n"
        "    ```\n"
    )
    result = run_evals(dataset_errors, adapter=None)[0].metrics[0]
    assert result.score == pytest.approx(2 / 3)
    assert "pair exploded" in (result.feedback or "")


def test_run_evals_rejects_non_iterable_test_and_training_data() -> None:
    bad_test_data = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > test_data: cases:\n"
        "    = `1`\n"
        "  > metric: score(cases):\n"
        "    = `1.0`\n"
    )
    with pytest.raises(RuntimeError, match="Error evaluating test_data 'cases'"):
        run_evals(bad_test_data, adapter=None)

    bad_training_data = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > data: cases:\n"
        "    = `'not iterable for data'`\n"
        "  > metric: score(cases):\n"
        "    = `1.0`\n"
    )
    with pytest.raises(RuntimeError, match="Error evaluating training data 'cases'"):
        run_evals(bad_training_data, adapter=None)


def test_run_tests_and_evals_treat_long_or_source_like_strings_as_source() -> None:
    source = "@id(x):\n  = <x>\n"
    assert run_tests(source, adapter=None) == []
    assert run_evals(source, adapter=None) == []

    long_source = "x" * 501
    with pytest.raises(Exception):
        tests_runner.run_tests(long_source, adapter=None)


def test_run_tests_and_evals_path_detection_exceptions_and_non_string_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyRuntime:
        _validation_tests: list[Any] = []
        _validation_evals: list[Any] = []

    def fake_load(src_main: object, *, adapter: object) -> EmptyRuntime:
        return EmptyRuntime()

    def raise_exists(self: Path) -> bool:
        raise OSError("bad path")

    monkeypatch.setattr(tests_runner, "_load_program_with_validations", fake_load)
    monkeypatch.setattr(Path, "exists", raise_exists)

    assert run_tests("not-a-real-path", adapter=None) == []
    assert run_evals("not-a-real-path", adapter=None) == []
    assert run_tests(cast(Any, 123), adapter=None) == []
    assert run_evals(cast(Any, 123), adapter=None) == []


def test_run_tests_and_evals_path_detection_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyRuntime:
        _validation_tests: list[Any] = []
        _validation_evals: list[Any] = []

    monkeypatch.setattr(
        tests_runner,
        "_load_program_with_validations",
        lambda src_main, *, adapter: EmptyRuntime(),
    )

    assert run_tests("missing-file.kedi", adapter=None) == []
    assert run_evals("missing-file.kedi", adapter=None) == []


def test_run_evals_legacy_block_metric_and_dataset_expected_shapes() -> None:
    legacy_block = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > metric: legacy_block:\n"
        "    = ```\n"
        "    return 0.25\n"
        "    ```\n"
    )
    assert run_evals(legacy_block, adapter=None)[0].metrics[0].score == 0.25

    expected_dict = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > data: cases:\n"
        "    = `[('input', {'answer': 'input'})]`\n"
        "  > metric: score(cases):\n"
        "    = ```\n"
        "    return 1.0 if expected['answer'] == cases else 0.0\n"
        "    ```\n"
    )
    assert run_evals(expected_dict, adapter=None)[0].metrics[0].score == 1.0

    dataset_bad_tuple = (
        "@id(x):\n"
        "  = `x`\n\n"
        "@eval: id:\n"
        "  > data: cases:\n"
        "    = `['x']`\n"
        "  > metric: score(cases):\n"
        "    = `(1.0, 'bad', 'tuple')`\n"
    )
    result = run_evals(dataset_bad_tuple, adapter=None)[0].metrics[0]
    assert result.score == 0.0
    assert "Metric tuple result" in (result.feedback or "")
