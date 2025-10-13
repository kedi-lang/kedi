"""Comprehensive type validation tests for Marker DSL.

Tests type checking for parameters, variables, return values, and complex types.
"""
import pytest
from pydantic_core import ValidationError
from tests.utils_llm import run_without_llm


def _run(src: str):
    """Helper to run marker program without LLM."""
    return run_without_llm(src)


class TestCustomTypeFieldValidation:
    """Tests for custom type field validation (via Pydantic)."""
    
    def test_custom_type_with_wrong_int_field(self):
        """Test custom type with string instead of int."""
        src = """
~Person(name, age:int)

[p: Person] = `Person(name='Alice', age='not_an_int')`

= <`p.name`>
"""
        with pytest.raises(RuntimeError, match="ValidationError"):
            _run(src)
    
    def test_custom_type_with_correct_fields(self):
        """Test custom type with correct field types."""
        src = """
~Person(name, age:int)

[p: Person] = `Person(name='Alice', age=30)`

= <`p.name + ':' + str(p.age)`>
"""
        result = _run(src)
        assert result == "Alice:30"
    
    def test_custom_type_with_list_field_wrong_elements(self):
        """Test custom type with list field containing wrong element types."""
        src = """
~Team(name, scores:list[int])

[t: Team] = `Team(name='Eagles', scores=[10, 'twenty', 30])`

= <`t.name`>
"""
        with pytest.raises(RuntimeError, match="ValidationError"):
            _run(src)
    
    def test_custom_type_with_list_field_correct(self):
        """Test custom type with list field containing correct elements."""
        src = """
~Team(name, scores:list[int])

[t: Team] = `Team(name='Eagles', scores=[10, 20, 30])`

= <`str(sum(t.scores))`>
"""
        result = _run(src)
        assert result == "60"


class TestParameterTypeValidation:
    """Tests for procedure parameter type validation."""
    
    def test_int_parameter_with_string_value(self):
        """Test int parameter receives string - should fail."""
        src = """
@process(value: int) -> int:
    = `value * 2`

= <`process('not_an_int')`>
"""
        # Inline call -> wrapped RuntimeError
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_int_parameter_with_correct_value(self):
        """Test int parameter with correct type."""
        src = """
@process(value: int) -> int:
    = `value * 2`

= <`str(process(21))`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_list_int_parameter_with_mixed_elements(self):
        """Test list[int] parameter with mixed element types."""
        src = """
@sum_numbers(numbers: list[int]) -> int:
    = `sum(numbers)`

= <`sum_numbers([1, '2', 3])`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_list_int_parameter_with_correct_elements(self):
        """Test list[int] parameter with correct types."""
        src = """
@sum_numbers(numbers: list[int]) -> int:
    = `sum(numbers)`

= <`str(sum_numbers([1, 2, 3, 4, 5]))`>
"""
        result = _run(src)
        assert result == "15"
    
    def test_custom_type_parameter_with_correct_value(self):
        """Test custom type parameter validation."""
        src = """
~Person(name, age:int)

@greet(p: Person):
    = Hello <`p.name`>!

= <`greet(Person(name='Bob', age=25))`>
"""
        result = _run(src)
        assert result == "Hello Bob!"


class TestVariableTypeValidation:
    """Tests for variable assignment type validation."""
    
    def test_int_variable_with_string_value(self):
        """Test assigning string to int variable."""
        src = """
[num: int] = `"not_a_number"`

= <`num`>
"""
        with pytest.raises(TypeError, match="Annotation type mismatch"):
            _run(src)
    
    def test_int_variable_with_correct_value(self):
        """Test assigning int to int variable."""
        src = """
[num: int] = `42`

= <`str(num)`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_list_int_variable_with_mixed_elements(self):
        """Test list[int] variable with mixed element types."""
        src = """
[numbers: list[int]] = `[1, "2", 3]`

= <`str(len(numbers))`>
"""
        # Top-level backtick assignment raises direct TypeError
        with pytest.raises(TypeError):
            _run(src)
    
    def test_list_int_variable_with_correct_elements(self):
        """Test list[int] variable with correct types."""
        src = """
[numbers: list[int]] = `[1, 2, 3]`

= <`str(sum(numbers))`>
"""
        result = _run(src)
        assert result == "6"
    
    def test_native_non_string_requires_annotation(self):
        """Test that native non-string values require type annotation."""
        src = """
[value] = `123`

= <`value`>
"""
        with pytest.raises(TypeError, match="Native non-string assignment requires explicit type annotation"):
            _run(src)


class TestReturnTypeValidation:
    """Tests for procedure return type validation."""
    
    def test_return_int_without_annotation(self):
        """Test that returning int without annotation fails."""
        src = """
@get_number():
    = `42`

= <`get_number()`>
"""
        with pytest.raises(RuntimeError, match="Return value must be str"):
            _run(src)
    
    def test_return_int_with_annotation(self):
        """Test returning int with proper annotation."""
        src = """
@get_number() -> int:
    = `42`

= <`str(get_number())`>
"""
        result = _run(src)
        assert result == "42"
    
    def test_return_list_int_with_mixed_elements(self):
        """Test returning list[int] with wrong element types."""
        src = """
@get_numbers() -> list[int]:
    [numbers: list[int]] = `[1, 2, "three", 4]`
    = `numbers`

= <`get_numbers()`>
"""
        # Inside inline call -> wrapped as RuntimeError
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_return_list_int_with_correct_elements(self):
        """Test returning list[int] with correct types."""
        src = """
@get_numbers() -> list[int]:
    [numbers: list[int]] = `[1, 2, 3, 4]`
    = `numbers`

= <`str(get_numbers()[2])`>
"""
        result = _run(src)
        assert result == "3"


class TestListCustomTypeValidation:
    """Tests specifically for list[CustomType] validation."""
    
    def test_list_custom_type_all_valid(self):
        """Test list[CustomType] with all valid elements."""
        src = """
~Book(title, pages:int)

    [books: list[Book]] = `[Book(title="Book1", pages=100), Book(title="Book2", pages=200)]`

= <`str(len(books))`>
"""
        result = _run(src)
        assert result == "2"
    
    def test_list_custom_type_with_string_element(self):
        """Test list[CustomType] with string element."""
        src = """
~Book(title, pages:int)

[books: list[Book]] = ```
[Book(title="Book1", pages=100), "not_a_book", Book(title="Book3", pages=300)]
```

= <`str(len(books))`>
"""
        with pytest.raises(TypeError):
            _run(src)
    
    def test_list_custom_type_with_dict_element(self):
        """Test list[CustomType] with dict element (should fail)."""
        src = """
~Book(title, pages:int)

[books: list[Book]] = ```
[Book(title="Book1", pages=100), {"title": "Book2", "pages": 200}]
```

= <`str(len(books))`>
"""
        with pytest.raises(TypeError):
            _run(src)
    
    def test_list_custom_type_with_none_element(self):
        """Test list[CustomType] with None element."""
        src = """
~Book(title, pages:int)

[books: list[Book]] = ```
[Book(title="Book1", pages=100), None]
```

= <`str(len(books))`>
"""
        with pytest.raises(TypeError):
            _run(src)
    
    def test_list_custom_type_with_wrong_custom_type(self):
        """Test list[CustomType] with different custom type."""
        src = """
~Book(title, pages:int)
~Magazine(title, issue:int)

[books: list[Book]] = ```
[Book(title="Book1", pages=100), Magazine(title="Mag1", issue=5)]
```

= <`str(len(books))`>
"""
        with pytest.raises(TypeError):
            _run(src)


class TestEdgeCases:
    """Test edge cases in type validation."""
    
    def test_empty_list_with_type_annotation(self):
        """Test empty list with type annotation is valid."""
        src = """
[numbers: list[int]] = `[]`

= <`str(len(numbers))`>
"""
        result = _run(src)
        assert result == "0"
    
    def test_nested_list_type(self):
        """Test nested list[list[int]] type."""
        src = """
[matrix: list[list[int]]] = `[[1, 2], [3, 4]]`

= <`str(matrix[1][1])`>
"""
        result = _run(src)
        assert result == "4"
    
    def test_nested_list_with_wrong_inner_type(self):
        """Test nested list with wrong inner element type."""
        src = """
[matrix: list[list[int]]] = `[[1, "2"], [3, 4]]`

= <`str(len(matrix))`>
"""
        with pytest.raises(TypeError):
            _run(src)
    
    def test_bool_type_validation(self):
        """Test boolean type validation."""
        src = """
@check_flag(flag: bool):
    = <`'yes' if flag else 'no'`>

= <`check_flag(True)`>
"""
        result = _run(src)
        assert result == "yes"
    
    def test_bool_type_with_string_fails(self):
        """Test bool type with string value fails."""
        src = """
@check_flag(flag: bool):
    return <`'yes' if flag else 'no'`>

= <`check_flag('true')`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_float_type_validation(self):
        """Test float type validation."""
        src = """
@double(val: float) -> float:
    = `val * 2.0`

= <`str(double(3.14))`>
"""
        result = _run(src)
        assert result == "6.28"
    
    def test_float_type_with_string_fails(self):
        """Test float type with string fails."""
        src = """
@double(val: float) -> float:
    = `val * 2.0`

= <`double('3.14')`>
"""
        with pytest.raises(RuntimeError):
            _run(src)


class TestParameterListElementValidation:
    """Test parameter validation catches wrong list element types."""
    
    def test_list_int_first_element_wrong(self):
        """Test list[int] with first element wrong type."""
        src = """
@process(nums: list[int]) -> int:
    = `sum(nums)`

= <`process(['1', 2, 3])`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_list_int_middle_element_wrong(self):
        """Test list[int] with middle element wrong type."""
        src = """
@process(nums: list[int]) -> int:
    = `sum(nums)`

= <`process([1, 2, '3', 4, 5])`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_list_int_last_element_wrong(self):
        """Test list[int] with last element wrong type."""
        src = """
@process(nums: list[int]) -> int:
    = `sum(nums)`

= <`process([1, 2, 3, '4'])`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
    
    def test_list_int_multiple_elements_wrong(self):
        """Test list[int] with multiple wrong elements."""
        src = """
@process(nums: list[int]) -> int:
    = `sum(nums)`

= <`process(['1', 2, '3', 4, '5'])`>
"""
        with pytest.raises(RuntimeError):
            _run(src)
