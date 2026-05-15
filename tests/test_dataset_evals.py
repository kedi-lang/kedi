"""Tests for dataset-aware evaluation metrics."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program
from kedi.tests_runner import run_evals
from tests.mock_adapter import MockAdapter


def test_eval_with_dataset(monkeypatch):
    """Test eval with dataset iteration."""
    src = """
@prime_factors(n: int) -> list[int]:
  = ```
  factors = []
  d = 2
  while n > 1:
      while n % d == 0:
          factors.append(d)
          n //= d
      d += 1
  return factors
  ```

@eval: prime_factors:
  > data: test_cases:
    = ```
    return {6:[2,3], 28:[2,2,7], 35:[5,7]}.items()
    ```
  > metric: correctness(test_cases):
    = ```
    k, v = test_cases
    return prime_factors(k) == v
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert len(evals) == 1
    assert evals[0].procedure == "prime_factors"
    assert len(evals[0].metrics) == 1
    assert evals[0].metrics[0].name == "correctness"
    # All 3 cases should pass
    assert evals[0].metrics[0].score == 1.0


def test_eval_with_train_and_test(monkeypatch):
    """Test eval with both train and test datasets."""
    src = """
@double(n: int) -> int:
  = `n * 2`

@eval: double:
  > data: inputs:
    = ```
    return [1, 2, 3]
    ```
  > test_data: inputs:
    = ```
    return [4, 5]
    ```
  > metric: accuracy(inputs):
    = ```
    expected = inputs * 2
    actual = double(inputs)
    return 1.0 if actual == expected else 0.0
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert len(evals) == 1
    assert len(evals[0].metrics) == 1
    # Test data is preferred, so metric should use test_data [4, 5]
    metric = evals[0].metrics[0]
    assert metric.name == "accuracy"
    assert metric.score == 1.0  # Should score perfectly on test data


def test_eval_metric_with_bool_return(monkeypatch):
    """Test metric that returns bool."""
    src = """
@is_even(n: int) -> bool:
  = `n % 2 == 0`

@eval: is_even:
  > data: numbers:
    = ```
    return [2, 4, 6, 8]
    ```
  > metric: all_even(numbers):
    = ```
    return is_even(numbers)
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert evals[0].metrics[0].score == 1.0  # All should be even


def test_eval_metric_with_tuple_return(monkeypatch):
    """Test metric that returns (score, feedback) tuple."""
    src = """
@add(a: int, b: int) -> int:
  = `a + b`

@eval: add:
  > data: pairs:
    = ```
    return [(1, 2), (3, 4)]
    ```
  > metric: check(pairs):
    = ```
    a, b = pairs
    result = add(a, b)
    expected = a + b
    if result == expected:
        return (1.0, "correct")
    else:
        return (0.0, f"expected {expected}, got {result}")
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert evals[0].metrics[0].score == 1.0
    assert "correct" in (evals[0].metrics[0].feedback or "")


def test_eval_empty_dataset(monkeypatch):
    """Test eval with empty dataset."""
    src = """
@foo(x: int) -> int:
  = `x + 1`

@eval: foo:
  > data: empty_set:
    = ```
    return []
    ```
  > metric: test(empty_set):
    = ```
    return 1.0
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert evals[0].metrics[0].score == 0.0
    assert evals[0].metrics[0].feedback == "no examples"


def test_eval_empty_test_data(monkeypatch):
    """Test eval with empty test_data."""
    src = """
@double(n: int) -> int:
  = `n * 2`

@eval: double:
  > data: train:
    = ```
    return [1, 2, 3]
    ```
  > test_data: train:
    = ```
    return []
    ```
  > metric: accuracy(train):
    = ```
    expected = train * 2
    actual = double(train)
    return 1.0 if actual == expected else 0.0
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert len(evals[0].metrics) == 1
    # Test data is preferred (empty in this case), so metric should use test_data
    test_metric = evals[0].metrics[0]
    assert test_metric.name == "accuracy"
    assert test_metric.score == 0.0
    assert test_metric.feedback == "no examples"


def test_eval_unknown_dataset_error(monkeypatch):
    """Test that referencing unknown dataset raises error."""
    src = """
@foo(x: int) -> int:
  = `x + 1`

@eval: foo:
  > data: my_data:
    = ```
    return [1, 2, 3]
    ```
  > metric: test(wrong_name):
    = ```
    return 1.0
    ```
"""
    with pytest.raises(ValueError, match="no data.*with this name found"):
        run_evals(src, adapter=MockAdapter())


def test_eval_non_iterable_data_error(monkeypatch):
    """Test that non-iterable data raises error."""
    # Strings are iterable in Python, so this won't fail on iterability check
    # But let's test with a non-iterable value
    src2 = """
@foo(x: int) -> int:
  = `x + 1`

@eval: foo:
  > data: bad_data:
    = ```
    return 42  # Not iterable
    ```
  > metric: test(bad_data):
    = ```
    return 1.0
    ```
"""
    # The error should be raised when trying to evaluate the dataset
    # But since we fall back to training data, the error will occur there
    with pytest.raises(
        (RuntimeError, ValueError),
        match="must return an iterable|no data.*with this name",
    ):
        run_evals(src2, adapter=MockAdapter())


def test_eval_dict_dataset_conversion(monkeypatch):
    """Test that dict datasets are converted to items()."""
    src = """
@identity(x):
  = `<x>`

@eval: identity:
  > data: mapping:
    = ```
    return {"a": 1, "b": 2}
    ```
  > metric: check(mapping):
    = ```
    k, v = mapping
    return k in ["a", "b"] and v in [1, 2]
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    # Should iterate over dict items
    assert evals[0].metrics[0].score == 1.0


def test_eval_parsing_unknown_directive(monkeypatch):
    """Test that unknown directives in eval raise error."""
    src = """
@foo():
  = ok

@eval: foo:
  > unknown: directive:
    = `1.0`
"""
    with pytest.raises(ValueError, match="Unknown directive"):
        parse_program(src)


def test_eval_metric_without_dataset(monkeypatch):
    """Test that metric without dataset name is still supported (legacy)."""
    src = """
@foo():
  = ok

@eval: foo:
  > metric: simple:
    = `1.0`
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert evals[0].metrics[0].score == 1.0
    assert evals[0].metrics[0].name == "simple"


def test_eval_multiple_datasets(monkeypatch):
    """Test eval with multiple datasets (only one metric allowed per procedure)."""
    # This test is now invalid - we only allow one metric per procedure
    # Testing with single metric on one dataset
    src = """
@add(a: int, b: int) -> int:
  = `a + b`

@eval: add:
  > data: small:
    = ```
    return [(1, 2)]
    ```
  > metric: small_test(small):
    = ```
    a, b = small
    return add(a, b) == 3
    ```
"""
    evals = run_evals(src, adapter=MockAdapter())
    assert len(evals[0].metrics) == 1
    assert evals[0].metrics[0].score == 1.0
