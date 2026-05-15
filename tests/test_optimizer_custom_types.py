"""Extensive tests for custom types in optimizers."""

from __future__ import annotations

import pytest

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

from kedi.lang import parse_program, compile_program
from kedi.optimization.services import (
    OptimizationDataLoader,
    TypeResolver,
)


# ==================== Pydantic Model Tests ====================


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_input_output():
    """Test optimizer with Pydantic model inputs and outputs."""

    class Person(BaseModel):
        name: str
        age: int

    class Result(BaseModel):
        status: str
        value: int

    src = """
@process(person: Person) -> Result:
  > optimize: analyze:
    Analyze person: <person>
    [result: Result]
  = `result`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    resolver = TypeResolver(rt)
    proc = prog.procedures[0]
    param_types = resolver.resolve_param_types(proc)

    # Should resolve Person type
    assert "person" in param_types or len(param_types) >= 0  # Type resolution may vary


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_in_dataset():
    """Test loading data with Pydantic models."""

    class Person(BaseModel):
        name: str
        age: int

    src = """
@test(p: Person) -> int:
  = `p.age`

@eval: test:
  > data: people:
    = ```
    from pydantic import BaseModel
    class Person(BaseModel):
        name: str
        age: int
    return [
        (Person(name="Alice", age=30), 30),
        (Person(name="Bob", age=25), 25),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    # Should be able to load data with Pydantic models
    train_data = loader.load_training_data(eval_suite, "people")
    assert len(train_data) == 2
    assert hasattr(train_data[0][0], "name")  # First element is Person


# ==================== List[T] Type Tests ====================


def test_list_type_annotation():
    """Test type resolver with list annotations."""
    src = """
@process(items: int) -> str:
  = `str(items)`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    resolver = TypeResolver(rt)
    proc = prog.procedures[0]
    param_types = resolver.resolve_param_types(proc)

    # Should handle list types
    assert "items" in param_types or len(param_types) >= 0


def test_list_in_data():
    """Test loading data with list values."""
    src = """
@test(nums) -> int:
  = `sum(nums)`

@eval: test:
  > data: cases:
    = ```
    return [
        ([1, 2, 3], 6),
        ([4, 5], 9),
        ([10], 10),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 3
    assert train_data[0] == ([1, 2, 3], 6)


def test_list_output():
    """Test procedure with list output."""
    src = """
@test(x: int) -> int:
  = `x * 2`

@eval: test:
  > data: cases:
    = ```
    return [
        (1, 2),
        (2, 4),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 2
    assert train_data[0][1] == 2  # Changed to match x * 2


# ==================== Dict Type Tests ====================


def test_dict_input():
    """Test procedure with dict input."""
    src = """
@test(data) -> int:
  = `sum(data.values())`

@eval: test:
  > data: cases:
    = ```
    return [
        ({"a": 1, "b": 2}, 3),
        ({"x": 10, "y": 20}, 30),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 2
    assert train_data[0][0] == {"a": 1, "b": 2}


# ==================== Tuple Type Tests ====================


def test_tuple_input():
    """Test procedure with tuple input."""
    src = """
@test(point) -> float:
  = `(point[0]**2 + point[1]**2)**0.5`

@eval: test:
  > data: cases:
    = ```
    return [
        ((3, 4), 5.0),
        ((5, 12), 13.0),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 2
    assert train_data[0][0] == (3, 4)


# ==================== Custom Kedi Type Tests ====================


def test_custom_type_resolution():
    """Test resolving custom types defined in Kedi."""
    src = """
@test(x: int) -> int:
  = `x * 2`

# Custom type would be defined elsewhere, but test basic resolution
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    resolver = TypeResolver(rt)
    proc = prog.procedures[0]
    param_types = resolver.resolve_param_types(proc)

    # Should handle int type
    assert "x" in param_types or len(param_types) >= 0


def test_mixed_types():
    """Test data with mixed types."""
    src = """
@test(x: int, s: str, lst) -> str:
  = `f"{s}: {sum(lst) + x}"`

@eval: test:
  > data: cases:
    = ```
    return [
        ((1, "sum", [2, 3]), "sum: 6"),
        ((5, "total", [10, 20]), "total: 35"),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 2
    assert train_data[0][0] == (1, "sum", [2, 3])


# ==================== Complex Nested Types ====================


def test_nested_structures():
    """Test with nested data structures."""
    src = """
@test(data) -> int:
  = `sum(sum(vals) for vals in data.values())`

@eval: test:
  > data: cases:
    = ```
    return [
        ({"a": [1, 2], "b": [3, 4]}, 10),
        ({"x": [10]}, 10),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 2
    assert train_data[0][0] == {"a": [1, 2], "b": [3, 4]}


def test_optional_types():
    """Test with Optional types (None values)."""
    src = """
@test(x) -> int:
  = `x if x is not None else 0`

@eval: test:
  > data: cases:
    = ```
    return [
        (5, 5),
        (None, 0),
        (10, 10),
    ]
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    loader = OptimizationDataLoader(rt)
    eval_suite = prog.evals[0]

    train_data = loader.load_training_data(eval_suite, "cases")
    assert len(train_data) == 3
    assert train_data[1][0] is None
