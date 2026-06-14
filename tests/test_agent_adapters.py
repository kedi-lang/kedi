from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import DSPyAdapter, PydanticAdapter, make_repr
from kedi.agent_adapter.tool_override import ToolScope
from kedi.agent_adapter.utils import (
    dspy_sync_tool_function,
    kedi_procedure_tool_spec,
    python_callable_tool_spec,
)
from kedi.lang import compile_program, parse_program


class DirectOutput(BaseModel):
    value: str


def test_make_repr_uses_class_name_for_standard_class_repr() -> None:
    class Custom:
        pass

    assert make_repr(Custom) == "Custom"


def test_pydantic_adapter_builds_types_and_produces_with_schema_or_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized_models: list[Any] = []

    def fake_agent_init(self: object, model: object = None, **_: Any) -> None:
        initialized_models.append(model)

    monkeypatch.setattr(adapters_module.Agent, "__init__", fake_agent_init)

    adapter = cast(Any, PydanticAdapter(model="openai/gpt-4o-mini"))
    PydanticAdapter(model=TestModel())

    async def fake_run(template: str, *, output_type: type[Any], **kwargs: Any) -> Any:
        assert template == "template"
        assert kwargs == {"metadata": "ok"}
        if output_type is DirectOutput:
            return SimpleNamespace(output=DirectOutput(value="direct"))
        return SimpleNamespace(output=output_type(answer=3))

    monkeypatch.setattr(adapter, "run", fake_run)

    assert initialized_models[0] == "openai:gpt-4o-mini"
    assert isinstance(initialized_models[1], TestModel)

    built = PydanticAdapter.type_builder({"answer": int})
    assert issubclass(built, BaseModel)
    assert built(answer=1).answer == 1

    schema_result = asyncio.run(
        adapter.produce("template", output_schema={"answer": int}, metadata="ok")
    )
    direct_result = asyncio.run(
        adapter.produce("template", output_type=DirectOutput, metadata="ok")
    )

    assert schema_result.answer == 3
    assert direct_result == DirectOutput(value="direct")
    assert adapter.produce_sync(
        "template", output_type=DirectOutput, metadata="ok"
    ) == DirectOutput(value="direct")

    with pytest.raises(AssertionError, match="Output schema cannot be empty"):
        PydanticAdapter.type_builder({})
    with pytest.raises(AssertionError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("template"))


def test_dspy_adapter_configures_lm_and_builds_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm_calls: list[dict[str, Any]] = []
    configure_calls: list[dict[str, Any]] = []

    class FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            lm_calls.append(kwargs)

    class FakeSettings:
        def configure(self, **kwargs: Any) -> None:
            configure_calls.append(kwargs)

    monkeypatch.setattr(adapters_module, "LM", FakeLM)
    monkeypatch.setattr(adapters_module.dspy, "settings", FakeSettings())

    DSPyAdapter(model="openai:gpt-4o-mini", max_tokens=12, temperature=0.25, provider=None)

    assert lm_calls == [
        {
            "model": "openai/gpt-4o-mini",
            "provider": None,
            "max_tokens": 12,
            "temperature": 0.25,
            "cache": False,
        }
    ]
    assert configure_calls and isinstance(configure_calls[0]["lm"], FakeLM)

    signature = DSPyAdapter.type_builder({"answer": int})
    assert issubclass(signature, adapters_module.dspy.Signature)

    with pytest.raises(AssertionError, match="Output schema must be provided"):
        DSPyAdapter.type_builder({})


def test_dspy_adapter_produce_and_sync_use_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSignature:
        def __init__(self) -> None:
            self.instructions: str | None = None

        def with_instructions(self, instructions: str | None) -> "FakeSignature":
            self.instructions = instructions
            return self

    class FakePredict:
        def __init__(self, signature: FakeSignature) -> None:
            self.signature = signature

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"instructions": self.signature.instructions, **kwargs}

    def fake_type_builder(output_schema: dict[str, type], **_: Any) -> FakeSignature:
        assert output_schema == {"answer": str}
        return FakeSignature()

    monkeypatch.setattr(adapters_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(fake_type_builder))

    adapter = object.__new__(DSPyAdapter)
    result = asyncio.run(
        adapter.produce(
            "prompt",
            {"answer": str},
            user_prompt="ignored",
            instructions="use context",
            extra=1,
        )
    )

    assert result == {
        "instructions": "use context",
        "extra": 1,
        "user_prompt": "prompt",
    }
    assert adapter.produce_sync("prompt", {"answer": str}, instructions=None) == {
        "instructions": None,
        "user_prompt": "prompt",
    }

    with pytest.raises(AssertionError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("prompt", {}))


def test_dspy_adapter_register_tool_wraps_async_callable() -> None:
    async def lookup(query: str, limit: int = 3) -> str:
        """Lookup docs."""
        return f"{query}:{limit}"

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("lookup", lookup))

    tool = adapter.scoped_tools[0]
    assert getattr(tool, "name") == "lookup"
    assert getattr(tool, "desc") == "Lookup docs."
    assert getattr(tool, "args") == {
        "query": {"type": "string"},
        "limit": {"default": 3, "type": "integer"},
    }
    assert inspect.signature(tool.func) == inspect.signature(lookup)
    assert tool(query="kedi", limit=2) == "kedi:2"


def test_dspy_adapter_register_tool_wraps_async_callable_instance() -> None:
    class WebFetchLocalTool:
        """Callable object doc should not replace __call__ metadata."""

        async def __call__(self, url: str, timeout: int = 10) -> str:
            """Fetch a URL."""
            return f"{url}:{timeout}"

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("web_fetch", WebFetchLocalTool()))

    tool = adapter.scoped_tools[0]
    assert getattr(tool, "name") == "web_fetch"
    assert getattr(tool, "desc") == "Fetch a URL."
    assert getattr(tool, "args") == {
        "url": {"type": "string"},
        "timeout": {"default": 10, "type": "integer"},
    }
    assert inspect.signature(tool.func) == inspect.signature(WebFetchLocalTool().__call__)
    assert tool(url="https://example.com", timeout=2) == "https://example.com:2"


def test_dspy_adapter_register_tool_wraps_sync_callable_idempotently() -> None:
    def lookup(query: str) -> str:
        """Lookup docs."""
        return query.upper()

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("lookup", lookup))
    tool = adapter.scoped_tools[0]

    assert tool(query="kedi") == "KEDI"
    assert dspy_sync_tool_function(tool.func) is tool.func


def test_pydantic_adapter_registers_kedi_and_python_tools() -> None:
    def lookup(query: str) -> str:
        return query

    src = "@search(query: str) -> str:\n  = <query>\n"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(kedi_procedure_tool_spec(rt, prog.procedures[0]))
    adapter.register_tool(python_callable_tool_spec("lookup", lookup))

    tools = adapter.scoped_tools
    assert [tool.name for tool in tools] == ["search", "lookup"]
    assert tools[0].metadata is not None
    assert tools[1].function is lookup
