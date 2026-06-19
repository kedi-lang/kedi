from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import kedi
import kedi.api as api_module
import pytest
from pydantic import BaseModel

from kedi import configure, context, query, reset_config, tool
from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter import AgentAdapter
from kedi.agent_profile import AgentProfile, current_block_agent_profile
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

T = TypeVar("T")


class Person(BaseModel):
    name: str
    age: int


class ApiAdapter(AgentAdapter[AttrDict[str, Any]]):
    def __init__(self) -> None:
        self.templates: list[str] = []
        self.schemas: list[dict[str, ResolvedKediType]] = []
        self.models: list[str | None] = []
        self.instructions: list[str | None] = []
        self.settings: list[dict[str, Any]] = []
        self.tools: list[tuple[str, ...]] = []
        self.tool_descriptions: list[tuple[str | None, ...]] = []

    def default_model(self) -> str | None:
        return None

    def profile_override(self, profile: AgentProfile) -> Any:
        adapter = self

        class ProfileCtx:
            def __enter__(self) -> None:
                adapter.models.append(profile.model)
                adapter.settings.append(dict(profile.settings))

            def __exit__(self, *args: object) -> None:
                return None

        return ProfileCtx()

    def tool_scope(self, tools: object = ()) -> Any:
        adapter = self

        class ToolCtx:
            def __enter__(self) -> None:
                adapter._scoped_tools = []

            def __exit__(self, *args: object) -> None:
                adapter._scoped_tools = []

        return ToolCtx()

    def register_tool(self, tool: object) -> None:
        scoped = getattr(self, "_scoped_tools", [])
        scoped.append(tool)
        self._scoped_tools = scoped

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        return self.produce_sync(template=template, output_schema=output_schema, **kwargs)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        schema = dict(output_schema or {})
        self.templates.append(template)
        self.schemas.append(schema)
        self.instructions.append(kwargs.get("instructions"))
        self.tools.append(
            tuple(getattr(tool, "name", "") for tool in getattr(self, "_scoped_tools", []))
        )
        self.tool_descriptions.append(
            tuple(getattr(tool, "description", None) for tool in getattr(self, "_scoped_tools", []))
        )
        if schema.get("output") is Person:
            return AttrDict({"output": {"name": "John", "age": 30}})
        if "summary" in schema:
            return AttrDict({"summary": "short summary"})
        return AttrDict({"answer": "ok"})


def lookup() -> str:
    return "lookup result"


@pytest.fixture
def adapter() -> ApiAdapter:
    configured = ApiAdapter()
    configure(model="configured:model", adapter=configured)
    return configured


def test_query_binds_arguments_and_defaults(adapter: ApiAdapter) -> None:
    @query
    def summarize(topic: str, tone: str = "clear") -> str:
        """kedi
        >> Summarize <topic> in a <tone> tone.
        Return [summary].
        = <summary>
        """

    assert summarize("Kedi") == "short summary"
    assert "Kedi" in adapter.templates[0]
    assert "clear" in adapter.templates[0]


def test_query_supports_dynamic_runtime_output_type(adapter: ApiAdapter) -> None:
    @query
    def extract_output(*, template: str, output_type: type[T]) -> T:
        """kedi
        >> Extract [output: `output_type`] from <template>.
        = `output`
        """

    result = extract_output(
        template="John is 30 years old and works as an engineer.",
        output_type=Person,
    )

    assert result == Person(name="John", age=30)
    assert adapter.schemas[0] == {"output": Person}


def test_query_namespace_object_matches_decorator_api(adapter: ApiAdapter) -> None:
    @kedi.query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"


def test_async_query_decorates_async_functions(adapter: ApiAdapter) -> None:
    @query
    async def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert asyncio.run(answer()) == "ok"


def test_configure_is_available_on_namespace() -> None:
    adapter = ApiAdapter()
    kedi.configure(adapter=adapter)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"


def test_context_temporarily_overrides_config() -> None:
    outer = ApiAdapter()
    inner = ApiAdapter()
    configure(adapter=outer)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert len(outer.templates) == 1

    with context(adapter=inner):
        assert answer() == "ok"

    assert len(inner.templates) == 1
    assert answer() == "ok"
    assert len(outer.templates) == 2


def test_context_supports_async_manager() -> None:
    outer = ApiAdapter()
    inner = ApiAdapter()
    configure(adapter=outer)

    @query
    async def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    async def run() -> None:
        async with context(adapter=inner):
            assert await answer() == "ok"

    asyncio.run(run())

    assert len(inner.templates) == 1
    assert len(outer.templates) == 0


def test_context_exit_without_enter_is_noop() -> None:
    ctx = context(adapter=ApiAdapter())

    assert ctx.__exit__() is None


def test_reset_config_restores_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ApiAdapter()
    seen_models: list[str | None] = []

    def fake_adapter(model: str | None) -> ApiAdapter:
        seen_models.append(model)
        return adapter

    configure(adapter=ApiAdapter())
    reset_config()
    monkeypatch.setattr(adapters_module, "PydanticAdapter", fake_adapter)
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_ADAPTER", raising=False)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert seen_models == [None]


def test_query_profile_arguments_apply_for_one_call(adapter: ApiAdapter) -> None:
    @query(system="Use terse prose.", settings={"temperature": 0.2}, tools=[lookup])
    def answer() -> str:
        """kedi
        >> Call the available tool if useful and return [answer].
        = <answer>
        """

    assert current_block_agent_profile() == AgentProfile.empty()
    assert answer() == "ok"
    assert adapter.instructions == ["Use terse prose."]
    assert adapter.settings == [{"temperature": 0.2}]
    assert adapter.tools == [("lookup",)]
    assert current_block_agent_profile() == AgentProfile.empty()


def test_tool_decorator_sets_name_description_and_retries(adapter: ApiAdapter) -> None:
    calls = 0

    @tool(name="search_docs", description="Search indexed docs.", retries=1)
    def flaky_lookup() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return "lookup result"

    assert flaky_lookup() == "lookup result"
    assert calls == 2

    @query(use=[flaky_lookup])
    def answer() -> str:
        """kedi
        > use: search_docs
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert adapter.tools == [("search_docs",)]
    assert adapter.tool_descriptions == [("Search indexed docs.",)]


def test_tool_direct_decorator_without_docstring() -> None:
    def bare() -> str:
        return "ok"

    bare.__doc__ = None
    wrapped = tool(bare)

    assert wrapped() == "ok"
    assert not hasattr(wrapped, api_module._TOOL_DESCRIPTION_ATTR)


def test_tool_sync_retry_exhaustion_raises_last_error() -> None:
    @tool(retries=1)
    def always_fails() -> str:
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="still broken"):
        always_fails()


def test_tool_async_retries_success_and_exhaustion() -> None:
    calls = 0

    @tool(retries=1)
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return "ok"

    @tool(retries=1)
    async def always_fails() -> str:
        raise RuntimeError("async broken")

    assert asyncio.run(flaky()) == "ok"
    assert calls == 2
    with pytest.raises(RuntimeError, match="async broken"):
        asyncio.run(always_fails())


def test_tool_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="retries"):
        tool(retries=-1)


def test_query_use_alias_registers_callable_tools(adapter: ApiAdapter) -> None:
    @query(use=[lookup])
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert adapter.tools == [("lookup",)]


def test_configure_default_tools_are_available_to_use_directive(adapter: ApiAdapter) -> None:
    configure(adapter=adapter, tools=[lookup])

    @query
    def answer() -> str:
        """kedi
        > use: lookup
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert adapter.tools == [("lookup",)]


def test_query_requires_kedi_docstring_header() -> None:
    with pytest.raises(RuntimeError, match="standalone `kedi` header"):

        @query
        def missing_header() -> str:
            """not kedi
            = ok
            """


def test_query_requires_docstring() -> None:
    class CallableWithoutDoc:
        def __call__(self) -> str:
            return "unused"

    with pytest.raises(RuntimeError, match="CallableWithoutDoc must contain"):
        query(CallableWithoutDoc())


def test_query_rejects_empty_kedi_program() -> None:
    with pytest.raises(RuntimeError, match="empty Kedi program"):

        @query
        def empty() -> str:
            """kedi"""


def test_configure_builds_default_adapter_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ApiAdapter()
    seen_models: list[str | None] = []

    def fake_adapter(model: str | None) -> ApiAdapter:
        seen_models.append(model)
        return adapter

    monkeypatch.setattr(adapters_module, "PydanticAdapter", fake_adapter)
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_ADAPTER", raising=False)
    configure()

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert seen_models == []
    assert answer() == "ok"
    assert seen_models == [None]


def test_configure_uses_adapter_env_when_not_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ApiAdapter()
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_adapter(model: str, **kwargs: Any) -> ApiAdapter:
        seen.append((model, kwargs))
        return adapter

    monkeypatch.setenv("KEDI_ADAPTER", "dspy")
    monkeypatch.setenv("KEDI_ADAPTER_MODEL", "openrouter:env-model")
    monkeypatch.setattr(adapters_module, "DSPyAdapter", fake_adapter)
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    configure(temperature=0.1)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert seen == [("openrouter:env-model", {"temperature": 0.1})]


def test_configure_explicit_model_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ApiAdapter()
    seen_models: list[str | None] = []

    def fake_adapter(model: str | None) -> ApiAdapter:
        seen_models.append(model)
        return adapter

    monkeypatch.setenv("KEDI_ADAPTER_MODEL", "openrouter:env-model")
    monkeypatch.setattr(adapters_module, "PydanticAdapter", fake_adapter)
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    configure(model=None)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert seen_models == [None]


def test_configure_builds_dspy_adapter_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ApiAdapter()
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_adapter(model: str, **kwargs: Any) -> ApiAdapter:
        seen.append((model, kwargs))
        return adapter

    monkeypatch.setattr(adapters_module, "DSPyAdapter", fake_adapter)
    configure(model="openrouter:test-model", adapter="dspy", temperature=0.3)

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert seen == []
    assert answer() == "ok"
    assert seen == [("openrouter:test-model", {"temperature": 0.3})]


def test_configure_dspy_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_ADAPTER", raising=False)
    configure(adapter="dspy")

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    try:
        with pytest.raises(ValueError, match="DSPy adapter requires"):
            answer()
    finally:
        configure(adapter=ApiAdapter())


def test_configure_rejects_unknown_adapter_name() -> None:
    with pytest.raises(ValueError, match="adapter must be"):
        configure(adapter="unknown")  # type: ignore[arg-type]
