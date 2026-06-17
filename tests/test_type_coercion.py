"""Tests for type coercion in LLM outputs."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from kedi.agent_adapter import AgentAdapter
from kedi.lang import compile_program, parse_program
from kedi.resolved_types import ResolvedKediType


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


class TypeCoercionMockAdapter(AgentAdapter[Any]):
    """Mock adapter that returns specific data structures for testing type coercion."""

    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **_: Any,
    ) -> Any:
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **_: Any,
    ) -> Any:
        # Extract keys from template
        start = template.find("[")
        end = template.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in template[start + 1 : end].split(",") if k.strip()]

        # Return predefined responses as an object with attributes
        class MockResult:
            def __init__(self, data: Dict[str, Any]):
                for key, value in data.items():
                    setattr(self, key, value)

        data: Dict[str, Any] = {}
        for key in keys:
            if key in self.responses:
                data[key] = self.responses[key]
            else:
                # Fallback to simple string
                data[key] = f"{key}-value"

        return MockResult(data)


class EmptyMockAdapter(AgentAdapter[Any]):
    """Mock adapter that returns empty object to trigger missing key errors."""

    async def produce(
        self, *, template: str, output_schema: dict[str, ResolvedKediType], **_: Any
    ) -> Any:
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **_: Any,
    ) -> Any:
        # Return empty object to trigger missing key error
        class EmptyResult:
            pass

        return EmptyResult()


def _run_with_responses(src: str, responses: Dict[str, Any]) -> Any:
    """Run a kedi program with specific LLM responses."""
    runtime = compile_program(parse_program(src), adapter=TypeCoercionMockAdapter(responses))
    return runtime.run_main()


class TestValidAdapterResponses:
    """Test scenarios where adapters return properly typed objects."""

    def test_simple_pydantic_model_response(self):
        """Test that adapter returns properly typed Pydantic model."""
        src = '\n        ```\n        from pydantic import BaseModel\n        \n        class Person(BaseModel):\n            name: str\n            age: int\n            \n            def greet(self) -> str:\n                return f"Hello, I\'m {self.name}"\n        ```\n        \n        >> [person: Person] is a person named "Alice" who is 25 years old.\n        \n        = ```\n        # Test that person is properly typed and methods work\n        result = person.greet()\n        return f"Type: {type(person).__name__}, Method result: {result}, Age: {person.age}"\n        ```\n        '

        # Adapter should return properly typed Person instance
        from pydantic import BaseModel

        class PersonModel(BaseModel):
            name: str
            age: int

            def greet(self) -> str:
                return f"Hello, I'm {self.name}"

        responses = {"person": PersonModel(name="Alice", age=25)}

        result = _run_with_responses(src, responses)
        assert "Type: PersonModel" in result
        assert "Method result: Hello, I'm Alice" in result
        assert "Age: 25" in result

    def test_nested_model_response(self):
        """Test that adapter returns properly typed nested Pydantic models."""
        src = '\n        ```\n        from pydantic import BaseModel\n        from typing import List\n        \n        class Person(BaseModel):\n            name: str\n            age: int\n            \n        class Company(BaseModel):\n            name: str\n            employees: List[Person]\n            \n            def total_employees(self) -> int:\n                return len(self.employees)\n        ```\n        \n        >> [company: Company] is a company with employees.\n        \n        = ```\n        # Test that company is properly typed and methods work\n        result = f"Company: {company.name}, Employees: {company.total_employees()}"\n        for emp in company.employees:\n            result += f", {emp.name}({emp.age})"\n        return result\n        ```\n        '

        # Adapter should return properly typed Company instance

        from pydantic import BaseModel

        class LocalPersonModel(BaseModel):
            name: str
            age: int

        class CompanyModel(BaseModel):
            name: str
            employees: List[LocalPersonModel]

            def total_employees(self) -> int:
                return len(self.employees)

        responses = {
            "company": CompanyModel(
                name="TechCorp",
                employees=[
                    LocalPersonModel(name="Alice", age=30),
                    LocalPersonModel(name="Bob", age=25),
                ],
            )
        }

        result = _run_with_responses(src, responses)
        assert "Company: TechCorp, Employees: 2" in result
        assert "Alice(30)" in result
        assert "Bob(25)" in result

    def test_multiple_typed_outputs(self):
        """Test multiple typed outputs in the same template."""
        src = '\n        ```\n        from pydantic import BaseModel\n        \n        class User(BaseModel):\n            username: str\n            \n            def display_name(self) -> str:\n                return f"@{self.username}"\n        \n        class Post(BaseModel):\n            title: str\n            content: str\n            \n            def summary(self) -> str:\n                return f"{self.title}: {self.content[:20]}..."\n        ```\n        \n        >> [user: User] created [post: Post] about programming.\n        \n        = ```\n        return f"{user.display_name()} wrote: {post.summary()}"\n        ```\n        '

        # Adapter should return properly typed objects
        from pydantic import BaseModel

        class UserModel(BaseModel):
            username: str

            def display_name(self) -> str:
                return f"@{self.username}"

        class PostModel(BaseModel):
            title: str
            content: str

            def summary(self) -> str:
                return f"{self.title}: {self.content[:20]}..."

        responses = {
            "user": UserModel(username="alice"),
            "post": PostModel(
                title="Python Tips",
                content="Here are some useful Python programming tips...",
            ),
        }

        result = _run_with_responses(src, responses)
        assert "@alice wrote: Python Tips: Here are some useful..." in result

    def test_already_typed_value_coercion(self):
        """Test that already-typed values (not dicts) are handled correctly."""
        src = '\n        ```\n        from pydantic import BaseModel\n        \n        class Person(BaseModel):\n            name: str\n            \n            def greet(self) -> str:\n                return f"Hello, I\'m {self.name}"\n        ```\n        \n        >> [person: Person] is a person.\n        \n        = ```\n        return f"Type: {type(person).__name__}, Name: {person.name}"\n        ```\n        '

        # Return an already-typed Person instance
        from pydantic import BaseModel

        class PersonModel(BaseModel):
            name: str

            def greet(self) -> str:
                return f"Hello, I'm {self.name}"

        person_instance = PersonModel(name="Charlie")
        responses = {"person": person_instance}

        result = _run_with_responses(src, responses)
        assert "Type: PersonModel" in result
        assert "Name: Charlie" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_output_key(self):
        """Test that missing output keys raise appropriate errors."""
        src = '\n        ```\n        from pydantic import BaseModel\n        \n        class Person(BaseModel):\n            name: str\n        ```\n        \n        >> [person: Person] is a person.\n        \n        = "Success"\n        '

        # Empty responses - should cause missing key error
        runtime = compile_program(parse_program(src), adapter=EmptyMockAdapter())

        with pytest.raises(ValueError, match="LLM did not return required key: person"):
            runtime.run_main()


class TestRegressionTests:
    """Test that the fix doesn't break existing functionality."""

    def test_untyped_outputs_still_work(self):
        """Test that untyped outputs continue to work as before."""
        src = '\n        >> [name] is a person\'s name.\n        [age] is their age.\n        \n        = ```\n        return f"Name: {name}, Age: {age}, Types: {type(name).__name__}, {type(age).__name__}"\n        ```\n        '

        responses = {"name": "Alice", "age": 25}

        result = _run_with_responses(src, responses)
        assert "Name: Alice" in result
        assert "Age: 25" in result
        assert "Types: str, int" in result

    def test_string_type_annotation_works(self):
        """Test that string type annotations work correctly."""
        src = '\n        ```\n        from pydantic import BaseModel\n        \n        class Person(BaseModel):\n            name: str\n        ```\n        \n        >> [person: str] is a person\'s name.\n        \n        = ```\n        return f"Name: {person}, Type: {type(person).__name__}"\n        ```\n        '

        responses = {"person": "Alice"}

        result = _run_with_responses(src, responses)
        assert "Name: Alice" in result
        assert "Type: str" in result
