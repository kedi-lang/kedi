"""Comprehensive tests for inline Python type annotations in Marker DSL.

Tests backtick-wrapped type expressions that are evaluated at runtime.
"""
import pytest
from tests.utils_llm import run_without_llm


def _run(src: str):
    """Helper to run marker program without LLM."""
    return run_without_llm(src)


class TestBasicBacktickTypes:
    """Test basic Python types with backtick syntax."""
    
    def test_int_backtick_type(self):
        """Test `int` type annotation."""
        src = """
[x: `int`] = `42`
= <`str(x)`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_str_backtick_type(self):
        """Test `str` type annotation."""
        src = """
[x: `str`] = `"hello"`
= <x>
"""
        result = _run(src)
        assert result == "hello"
    
    def test_float_backtick_type(self):
        """Test `float` type annotation."""
        src = """
[x: `float`] = `3.14`
= <`str(x)`>
"""
        result = _run(src)
        assert result == "3.14"
    
    def test_bool_backtick_type(self):
        """Test `bool` type annotation."""
        src = """
[x: `bool`] = `True`
= <`'yes' if x else 'no'`>
"""
        result = _run(src)
        assert result == "yes"


class TestComplexBacktickTypes:
    """Test complex types with backtick syntax."""
    
    def test_list_int_backtick_type(self):
        """Test `list[int]` type annotation."""
        src = """
[x: `list[int]`] = `[1, 2, 3]`
= <`str(sum(x))`>
"""
        result = _run(src)
        assert result == "6"
    
    def test_list_str_backtick_type(self):
        """Test `list[str]` type annotation."""
        src = """
[x: `list[str]`] = `["a", "b", "c"]`
= <`str(len(x))`>
"""
        result = _run(src)
        assert result == "3"
    
    def test_nested_list_backtick_type(self):
        """Test `list[list[int]]` type annotation."""
        src = """
[x: `list[list[int]]`] = `[[1, 2], [3, 4]]`
= <`str(x[1][1])`>
"""
        result = _run(src)
        assert result == "4"


class TestCustomTypesFromPrelude:
    """Test custom types defined in prelude with backtick syntax."""
    
    def test_prelude_custom_type(self):
        """Test custom type from prelude using backtick."""
        src = """
```
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```

[p: `Point`] = `Point(x=10, y=20)`
= <`str(p.x + p.y)`>
"""
        result = _run(src)
        assert result == "30"
    
    def test_prelude_list_custom_type(self):
        """Test list of custom type from prelude."""
        src = """
```
from dataclasses import dataclass

@dataclass
class Item:
    name: str
    value: int
```
        
[items: `list[Item]`] = `[Item(name="A", value=1), Item(name="B", value=2)]`
= <`str(len(items))`>
"""
        result = _run(src)
        assert result == "2"


class TestCustomTypesFromDSL:
    """Test custom types defined with ~ syntax using backtick."""
    
    def test_dsl_custom_type(self):
        """Test DSL-defined custom type with backtick."""
        src = """
~Person(name, age: int)

[p: `Person`] = `Person(name="Alice", age=30)`
= <`p.name + ':' + str(p.age)`>
"""
        result = _run(src)
        assert result == "Alice:30"
    
    def test_dsl_list_custom_type(self):
        """Test list of DSL-defined custom type."""
        src = """
~Book(title, pages: int)

[books: `list[Book]`] = `[Book(title="Book1", pages=100), Book(title="Book2", pages=200)]`
= <`str(len(books))`>
"""
        result = _run(src)
        assert result == "2"


class TestMixedUsage:
    """Test mixing backtick and non-backtick type annotations."""
    
    def test_mixed_parameter_types(self):
        """Test procedure with mixed parameter type annotations."""
        src = """
@process(x: int, y: `int`) -> int:
    = `x + y`

= <`str(process(10, 20))`>
"""
        result = _run(src)
        assert result == "30"
    
    def test_mixed_variable_types(self):
        """Test mixed variable type annotations."""
        src = """
[x: int] = `10`
[y: `int`] = `20`
= <`str(x + y)`>
"""
        result = _run(src)
        assert result == "30"
    
    def test_mixed_return_types(self):
        """Test procedure with mixed return type annotations."""
        src = """
@get_int() -> int:
    = `42`

@get_int_backtick() -> `int`:
    = `42`

= <`str(get_int() + get_int_backtick())`>
"""
        result = _run(src)
        assert result == "84"


class TestParameterAnnotations:
    """Test backtick type annotations in procedure parameters."""
    
    def test_int_parameter_backtick(self):
        """Test int parameter with backtick annotation."""
        src = """
@double(x: `int`) -> int:
    = `x * 2`

= <`str(double(21))`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_list_parameter_backtick(self):
        """Test list parameter with backtick annotation."""
        src = """
@sum_list(nums: `list[int]`) -> int:
    = `sum(nums)`

= <`str(sum_list([1, 2, 3, 4]))`>
"""
        result = _run(src)
        assert result == "10"
    
    def test_custom_type_parameter_backtick(self):
        """Test custom type parameter with backtick annotation."""
        src = """
~Person(name, age: int)

@greet(p: `Person`):
    = Hello <`p.name`>!

= <`greet(Person(name="Bob", age=25))`>
"""
        result = _run(src)
        assert result == "Hello Bob!"


class TestReturnTypeAnnotations:
    """Test backtick type annotations in procedure return types."""
    
    def test_int_return_backtick(self):
        """Test int return type with backtick annotation."""
        src = """
@get_number() -> `int`:
    = `42`

= <`str(get_number())`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_list_return_backtick(self):
        """Test list return type with backtick annotation."""
        src = """
@get_numbers() -> `list[int]`:
    = `[1, 2, 3, 4, 5]`

= <`str(len(get_numbers()))`>
"""
        result = _run(src)
        assert result == "5"
    
    def test_custom_type_return_backtick(self):
        """Test custom type return with backtick annotation."""
        src = """
~Point(x: int, y: int)

@create_point() -> `Point`:
    = `Point(x=10, y=20)`

= <`str(create_point().x + create_point().y)`>
"""
        result = _run(src)
        assert result == "30"


class TestOutputAnnotations:
    """Test backtick type annotations in output specifications."""
    
    def test_output_int_backtick(self):
        """Test output with int backtick annotation - should work with LLM."""
        src = """
[result: `int`] = The answer is 42
= <`str(result)`>
"""
        # This test expects the LLM to return an integer, but our mock returns strings
        # So we expect this to fail with type validation error
        with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
            _run(src)
    
    def test_output_list_backtick(self):
        """Test output with list backtick annotation - should work with LLM."""
        src = """
[items: `list[str]`] = List three fruits
= <`str(len(items))`>
"""
        # This test expects the LLM to return a list, but our mock returns strings
        # So we expect this to fail with type validation error
        with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
            _run(src)


class TestTypeSafetyValidation:
    """Test that type safety guarantees hold for backtick types."""
    
    def test_int_backtick_type_mismatch(self):
        """Test that int backtick type rejects string values."""
        src = """
[x: `int`] = `"not_a_number"`
= <`str(x)`>
"""
        with pytest.raises(TypeError, match="Annotation type mismatch"):
            _run(src)
    
    def test_list_int_backtick_type_mismatch(self):
        """Test that list[int] backtick type rejects mixed elements."""
        src = """
[x: `list[int]`] = `[1, "2", 3]`
= <`str(len(x))`>
"""
        with pytest.raises(TypeError):
            _run(src)
    
    def test_custom_type_backtick_validation(self):
        """Test custom type backtick validation."""
        src = """
~Person(name, age: int)

[p: `Person`] = `Person(name="Alice", age="not_an_int")`
= <`p.name`>
"""
        with pytest.raises(RuntimeError, match="ValidationError"):
            _run(src)
    
    def test_parameter_backtick_type_validation(self):
        """Test parameter backtick type validation."""
        src = """
@process(x: `int`) -> int:
    = `x * 2`

= <`str(process("not_an_int"))`>
"""
        with pytest.raises(RuntimeError):
            _run(src)


class TestErrorCases:
    """Test error cases for backtick type annotations."""
    
    def test_undefined_type_backtick(self):
        """Test backtick with undefined type."""
        src = """
[x: `UndefinedType`] = `42`
= <`str(x)`>
"""
        with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
            _run(src)
    
    def test_invalid_expression_backtick(self):
        """Test backtick with invalid Python expression."""
        src = """
[x: `int +`] = `42`
= <`str(x)`>
"""
        with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
            _run(src)
    
    def test_non_type_expression_backtick(self):
        """Test backtick with expression that doesn't evaluate to a type."""
        src = """
[x: `42`] = `42`
= <`str(x)`>
"""
        with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
            _run(src)


class TestTypeDefinitionBacktick:
    """Test backtick type annotations in custom type definitions."""
    
    def test_type_field_backtick(self):
        """Test custom type field with backtick type annotation."""
        src = """
~Person(name, age: `int`)

[p: Person] = `Person(name="Alice", age=30)`
= <`str(p.age)`>
"""
        result = _run(src)
        assert result == "30"
    
    def test_type_field_list_backtick(self):
        """Test custom type field with list backtick type annotation."""
        src = """
~Team(name, scores: `list[int]`)

[t: Team] = `Team(name="Eagles", scores=[10, 20, 30])`
= <`str(sum(t.scores))`>
"""
        result = _run(src)
        assert result == "60"
    
    def test_mixed_type_field_annotations(self):
        """Test custom type with mixed field type annotations."""
        src = """
~Mixed(name, count: int, items: `list[str]`)

[m: Mixed] = `Mixed(name="Test", count=2, items=["a", "b"])`
= <`str(m.count + len(m.items))`>
"""
        result = _run(src)
        assert result == "4"


class TestInteroperability:
    """Test that backtick and non-backtick types work interchangeably."""
    
    def test_function_call_interop(self):
        """Test calling functions with different annotation styles."""
        src = """
@func1(x: int) -> int:
    = `x * 2`

@func2(x: `int`) -> `int`:
    = `x * 2`

= <`str(func1(10) + func2(10))`>
"""
        result = _run(src)
        assert result == "40"
    
    def test_variable_assignment_interop(self):
        """Test variable assignment with different annotation styles."""
        src = """
[x: int] = `10`
[y: `int`] = `20`
[z: int] = `30`
= <`str(x + y + z)`>
"""
        result = _run(src)
        assert result == "60"
    
    def test_custom_type_interop(self):
        """Test custom types with different annotation styles."""
        src = """
~Person(name, age: int)

[p1: Person] = `Person(name="Alice", age=30)`
[p2: `Person`] = `Person(name="Bob", age=25)`
= <`str(p1.age + p2.age)`>
"""
        result = _run(src)
        assert result == "55"
