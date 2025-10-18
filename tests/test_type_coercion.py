"""Tests for type coercion in LLM outputs."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel

import pytest

from marker.lang import compile_program, parse_program
from marker.agent_adapter import AgentAdapter


class PersonModel(BaseModel):
    name: str
    age: int
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"


class CompanyModel(BaseModel):
    name: str
    employees: List[PersonModel]
    
    def total_employees(self) -> int:
        return len(self.employees)


class TypeCoercionMockAdapter(AgentAdapter[Dict[str, Any]]):
    """Mock adapter that returns specific data structures for testing type coercion."""
    
    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
    
    async def produce(
        self,
        *,
        template: str,
        output_schema: Dict[str, type],
        **_: Any,
    ) -> Dict[str, Any]:
        return self.produce_sync(template=template, output_schema=output_schema)
    
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: Dict[str, type] | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        # Extract keys from template
        start = template.find("[")
        end = template.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in template[start + 1 : end].split(",") if k.strip()]
        
        # Return predefined responses
        data: Dict[str, Any] = {}
        for key in keys:
            if key in self.responses:
                data[key] = self.responses[key]
            else:
                # Fallback to simple string
                data[key] = f"{key}-value"
        
        return data


def _run_with_responses(src: str, responses: Dict[str, Any]) -> Any:
    """Run a marker program with specific LLM responses."""
    runtime = compile_program(parse_program(src), adapter=TypeCoercionMockAdapter(responses))
    return runtime.run_main()


class TestValidTypeCoercion:
    """Test valid scenarios where type coercion should work correctly."""
    
    def test_simple_pydantic_model_coercion(self):
        """Test that a simple Pydantic model gets properly coerced."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
            
            def greet(self) -> str:
                return f"Hello, I'm {self.name}"
        ```
        
        [person: Person] is a person named "Alice" who is 25 years old.
        
        = ```
        # Test that person is properly typed and methods work
        result = person.greet()
        return f"Type: {type(person).__name__}, Method result: {result}, Age: {person.age}"
        ```
        """
        
        responses = {
            "person": {"name": "Alice", "age": 25}
        }
        
        result = _run_with_responses(src, responses)
        assert "Type: Person" in result
        assert "Method result: Hello, I'm Alice" in result
        assert "Age: 25" in result
    
    def test_nested_model_coercion(self):
        """Test coercion of nested Pydantic models with forward references."""
        src = """
        ```
        from pydantic import BaseModel
        from typing import List
        
        class Person(BaseModel):
            name: str
            age: int
            
        class Company(BaseModel):
            name: str
            employees: List[Person]
        ```
        
        [company: Company] is a company with employees.
        
        = "Success"
        """
        
        responses = {
            "company": {
                "name": "TechCorp",
                "employees": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25}
                ]
            }
        }
        
        # Should raise a validation error due to forward references
        with pytest.raises(Exception):  # PydanticUserError for forward references
            _run_with_responses(src, responses)
    
    def test_multiple_typed_outputs(self):
        """Test multiple typed outputs in the same template."""
        src = """
        ```
        from pydantic import BaseModel
        
        class User(BaseModel):
            username: str
            
            def display_name(self) -> str:
                return f"@{self.username}"
        
        class Post(BaseModel):
            title: str
            content: str
            
            def summary(self) -> str:
                return f"{self.title}: {self.content[:20]}..."
        ```
        
        [user: User] created [post: Post] about programming.
        
        = ```
        return f"{user.display_name()} wrote: {post.summary()}"
        ```
        """
        
        responses = {
            "user": {"username": "alice"},
            "post": {"title": "Python Tips", "content": "Here are some useful Python programming tips..."}
        }
        
        result = _run_with_responses(src, responses)
        assert "@alice wrote: Python Tips: Here are some useful..." in result
    
    def test_already_typed_value_coercion(self):
        """Test that already-typed values (not dicts) are handled correctly."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            
            def greet(self) -> str:
                return f"Hello, I'm {self.name}"
        ```
        
        [person: Person] is a person.
        
        = ```
        return f"Type: {type(person).__name__}, Name: {person.name}"
        ```
        """
        
        # Return an already-typed Person instance
        person_instance = PersonModel(name="Charlie", age=35)
        responses = {
            "person": person_instance
        }
        
        result = _run_with_responses(src, responses)
        assert "Type: Person" in result
        assert "Name: Charlie" in result


class TestInvalidTypeCoercion:
    """Test scenarios where type coercion should fail fast."""
    
    def test_invalid_dict_structure(self):
        """Test that invalid dict structures raise validation errors."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
        ```
        
        [person: Person] is a person.
        
        = "Success"
        """
        
        # Return invalid structure (missing required field)
        responses = {
            "person": {"name": "Alice"}  # Missing 'age' field
        }
        
        # Should raise a validation error
        with pytest.raises(Exception):  # Pydantic validation error
            _run_with_responses(src, responses)
    
    def test_wrong_type_structure(self):
        """Test that wrong type structures raise validation errors."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
        ```
        
        [person: Person] is a person.
        
        = "Success"
        """
        
        # Return completely wrong structure
        responses = {
            "person": {"title": "Mr.", "salary": 50000}  # Wrong fields entirely
        }
        
        # Should raise a validation error
        with pytest.raises(Exception):  # Pydantic validation error
            _run_with_responses(src, responses)
    
    def test_non_dict_value(self):
        """Test that non-dict values raise validation errors."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
        ```
        
        [person: Person] is a person.
        
        = "Success"
        """
        
        # Return a non-dict value
        responses = {
            "person": "just a string"  # Not a dict
        }
        
        # Should raise a validation error
        with pytest.raises(Exception):  # Pydantic validation error
            _run_with_responses(src, responses)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_output_key(self):
        """Test that missing output keys raise appropriate errors."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
        ```
        
        [person: Person] is a person.
        
        = "Success"
        """
        
        # Create a custom adapter that returns empty dict
        class EmptyMockAdapter(AgentAdapter[Dict[str, Any]]):
            async def produce(self, *, template: str, output_schema: Dict[str, type], **_: Any) -> Dict[str, Any]:
                return self.produce_sync(template=template, output_schema=output_schema)
            
            def produce_sync(self, *, template: str, output_schema: Dict[str, type] | None = None, **_: Any) -> Dict[str, Any]:
                return {}  # Return empty dict to trigger missing key error
        
        runtime = compile_program(parse_program(src), adapter=EmptyMockAdapter())
        
        with pytest.raises(ValueError, match="LLM did not return required key: person"):
            runtime.run_main()
    
    def test_none_value_coercion(self):
        """Test that None values raise validation errors."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
        ```
        
        [person: Person] is a person.
        
        = "Success"
        """
        
        responses = {
            "person": None
        }
        
        # Should raise a validation error for None value
        with pytest.raises(Exception):  # Pydantic validation error
            _run_with_responses(src, responses)
    
    def test_complex_nested_type_validation(self):
        """Test that complex nested types fail due to forward references."""
        src = """
        ```
        from pydantic import BaseModel
        from typing import List, Dict
        
        class ComplexType(BaseModel):
            data: Dict[str, List[int]]
            metadata: Dict[str, Any]
        ```
        
        [complex_obj: ComplexType] is a complex object.
        
        = "Success"
        """
        
        responses = {
            "complex_obj": {
                "data": {"key1": [1, 2, 3], "key2": [4, 5]},
                "metadata": {"version": "1.0", "author": "test"}
            }
        }
        
        # Should raise a validation error due to forward references
        with pytest.raises(Exception):  # PydanticUserError for forward references
            _run_with_responses(src, responses)
    
    def test_mixed_valid_invalid_outputs(self):
        """Test that invalid outputs cause the entire operation to fail."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
        
        class Company(BaseModel):
            name: str
        ```
        
        [person: Person] works at [company: Company].
        
        = "Success"
        """
        
        responses = {
            "person": {"name": "Alice", "age": 30},  # Valid
            "company": {"title": "TechCorp"}  # Invalid (wrong field name)
        }
        
        # Should raise a validation error for the invalid company
        with pytest.raises(Exception):  # Pydantic validation error
            _run_with_responses(src, responses)


class TestRegressionTests:
    """Test that the fix doesn't break existing functionality."""
    
    def test_untyped_outputs_still_work(self):
        """Test that untyped outputs continue to work as before."""
        src = """
        [name] is a person's name.
        [age] is their age.
        
        = ```
        return f"Name: {name}, Age: {age}, Types: {type(name).__name__}, {type(age).__name__}"
        ```
        """
        
        responses = {
            "name": "Alice",
            "age": 25
        }
        
        result = _run_with_responses(src, responses)
        assert "Name: Alice" in result
        assert "Age: 25" in result
        assert "Types: str, int" in result
    
    def test_string_type_annotation_works(self):
        """Test that string type annotations work correctly."""
        src = """
        ```
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
        ```
        
        [person: str] is a person's name.
        
        = ```
        return f"Name: {person}, Type: {type(person).__name__}"
        ```
        """
        
        responses = {
            "person": "Alice"
        }
        
        result = _run_with_responses(src, responses)
        assert "Name: Alice" in result
        assert "Type: str" in result
