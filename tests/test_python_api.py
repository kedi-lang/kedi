from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydantic_dataclass

import kedi
import kedi.agent_adapter.registry as adapter_registry
import kedi.api as api_module
from kedi import (
    bind,
    cache_info,
    clear_cache,
    configure,
    context,
    query,
    reset_config,
    tool,
)
from kedi import type as kedi_type
from kedi.agent_adapter import AgentAdapter
from kedi.agent_adapter.adapters import dspy as dspy_adapter_module
from kedi.agent_adapter.adapters import langchain as langchain_adapter_module
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_profile import AgentProfile, current_block_agent_profile
from kedi.errors import KediExecutionError
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

T = TypeVar("T")


class Person(BaseModel):
    name: str
    age: int


class ApiAdapter(AgentAdapter[AttrDict[str, Any]]):
    kind = "agent-framework"
    shortname = "test"

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
        if "person" in schema:
            return AttrDict({"person": {"name": "Ada", "age": 42}})
        if "summary" in schema:
            return AttrDict({"summary": "short summary"})
        return AttrDict({"answer": "ok"})

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        return self.invoke_sync(prompt=prompt, **kwargs)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        self.templates.append(prompt)
        self.instructions.append(kwargs.get("instructions"))
        return f"response:{prompt}"


class HarnessApiAdapter(ApiAdapter):
    kind = "agent-harness"
    shortname = "test-harness"


def lookup() -> str:
    return "lookup result"


class AddressReprValue:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"<AddressReprValue label={self.label!r} at {hex(id(self))}>"


@pytest.fixture
def adapter() -> ApiAdapter:
    api_module._PARSE_CACHE.clear()
    api_module._RESPONSE_CACHE.clear()
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


def test_query_parse_cache_reuses_identical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_module._PARSE_CACHE.clear()
    calls = 0
    real_parse = api_module.parse_program

    def counting_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(api_module, "parse_program", counting_parse)

    @query
    def first() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    @query
    def second() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert first is not second
    assert calls == 1


def test_cache_info_and_clear_cache(adapter: ApiAdapter) -> None:
    @query(cache=True)
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    info = cache_info()
    assert info.parse_entries >= 1
    assert info.response_entries >= 1

    clear_cache()

    assert cache_info().parse_entries == 0
    assert cache_info().response_entries == 0


def test_query_response_cache_reuses_same_call(adapter: ApiAdapter) -> None:
    @query(cache=True)
    def summarize(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize("Kedi") == "short summary"
    assert summarize("Kedi") == "short summary"
    assert len(adapter.templates) == 1


def test_query_response_cache_splits_by_args_and_env(adapter: ApiAdapter) -> None:
    @query(cache=True)
    def summarize(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize("Kedi") == "short summary"
    assert summarize("Python") == "short summary"
    assert len(adapter.templates) == 2

    @query(cache=True, env={"topic": "env topic"})
    def summarize_from_env(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize_from_env("Kedi") == "short summary"
    assert len(adapter.templates) == 3


def test_response_cache_key_normalizes_memory_addresses() -> None:
    def sample(value: AddressReprValue) -> None:
        return None

    kwargs_a = {"value": AddressReprValue("same")}
    kwargs_b = {"value": AddressReprValue("same")}

    identity: dict[str, Any] = {
        "adapter_identity": ("agent-framework", "test", "tests", "ApiAdapter", 1),
        "adapter_kwargs": {},
        "profile": AgentProfile.empty(),
        "tool_names": (),
    }
    assert repr(kwargs_a["value"]) != repr(kwargs_b["value"])
    assert api_module._response_cache_key(
        kind="query",
        inner=sample,
        source_id="sample",
        source_hash="hash",
        args=(),
        kwargs=kwargs_a,
        config_env={},
        env={},
        **identity,
    ) == api_module._response_cache_key(
        kind="query",
        inner=sample,
        source_id="sample",
        source_hash="hash",
        args=(),
        kwargs=kwargs_b,
        config_env={},
        env={},
        **identity,
    )


def test_query_response_cache_splits_by_model(adapter: ApiAdapter) -> None:
    # B1: a cached query must not return a stale answer when the resolved model changes.
    @query(cache=True)
    def summarize(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize("Kedi") == "short summary"
    with context(model="other:model"):
        assert summarize("Kedi") == "short summary"
    # Different resolved model → a fresh model call, not a stale cache hit.
    assert len(adapter.templates) == 2
    assert adapter.models == ["configured:model", "other:model"]


def test_query_response_cache_ignores_repr_memory_address(adapter: ApiAdapter) -> None:
    @query(cache=True)
    def summarize(topic: AddressReprValue) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize(AddressReprValue("Kedi")) == "short summary"
    assert summarize(AddressReprValue("Kedi")) == "short summary"
    assert len(adapter.templates) == 1


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


def test_query_env_overrides_bound_runtime_globals(adapter: ApiAdapter) -> None:
    @query(env={"output_type": Person})
    def extract_output(*, template: str, output_type: type[Any]) -> Person:
        """kedi
        >> Extract [output: `output_type`] from <template>.
        = `output`
        """

    result = extract_output(template="John is 30 years old.", output_type=str)

    assert result == Person(name="John", age=30)
    assert adapter.schemas[0] == {"output": Person}


def test_query_cache_returns_isolated_copy(adapter: ApiAdapter) -> None:
    # B2: cached results are mutable (e.g. a pydantic model); a caller mutating a returned object
    # must not corrupt the shared cache entry for later identical calls.
    @query(cache=True)
    def extract(*, template: str, output_type: type[T]) -> T:
        """kedi
        >> Extract [output: `output_type`] from <template>.
        = `output`
        """

    first = extract(template="John is 30.", output_type=Person)
    assert first == Person(name="John", age=30)
    first.name = "MUTATED"  # corrupt the returned object in place

    second = extract(template="John is 30.", output_type=Person)  # identical args → cache hit
    assert len(adapter.templates) == 1  # genuinely a cache hit, not a re-run
    assert second == Person(name="John", age=30)  # not corrupted by the earlier mutation
    assert second is not first


def test_isolate_cached_falls_back_for_non_copyable() -> None:
    # Review #1: deep-copying a non-copyable cached result must degrade to by-reference, not crash a
    # previously-working cache=True call. Copyable results are still isolated (a distinct copy).
    import threading

    lock = threading.Lock()  # not deep-copyable
    assert api_module._isolate_cached(lock) is lock

    data = {"a": [1, 2]}
    copied = api_module._isolate_cached(data)
    assert copied == data and copied is not data


def test_query_argument_name_shadowing_tool_is_rejected(adapter: ApiAdapter) -> None:
    # B5: a query parameter whose name collides with a registered tool would overwrite the tool
    # callable in runtime globals, making the tool uncallable. Reject the collision loudly.
    @query(tools=[lookup])
    def ask(*, lookup: str) -> str:
        """kedi
        >> Answer the question [answer].
        = <answer>
        """

    with pytest.raises(KediExecutionError, match="shadow registered tool"):
        ask(lookup="value")


def test_kedi_type_transforms_bare_class_and_auto_injects(adapter: ApiAdapter) -> None:
    @kedi_type
    class AutoPerson:
        name: str
        age: int

    assert dataclasses.is_dataclass(AutoPerson)

    @query
    def extract_person(text: str) -> AutoPerson:
        """kedi
        >> Extract [person: AutoPerson] from <text>.
        = `person`
        """

    assert extract_person("Ada is 42.") == AutoPerson(name="Ada", age=42)
    assert adapter.schemas[0] == {"person": AutoPerson}


def test_kedi_type_registers_existing_type_surfaces_without_transform() -> None:
    class ApiPerson(BaseModel):
        name: str

    @dataclasses.dataclass
    class StdPerson:
        name: str

    @pydantic_dataclass
    class PydanticPerson:
        name: str

    assert kedi_type(ApiPerson) is ApiPerson
    assert kedi_type(StdPerson) is StdPerson
    assert kedi_type(PydanticPerson) is PydanticPerson


def test_kedi_type_inject_false_skips_auto_env(adapter: ApiAdapter) -> None:
    @kedi_type(inject=False)
    class HiddenPerson:
        name: str
        age: int

    assert dataclasses.is_dataclass(HiddenPerson)
    assert "HiddenPerson" not in api_module._auto_type_env(
        test_kedi_type_inject_false_skips_auto_env
    )

    @query(env={"HiddenPerson": HiddenPerson})
    def extract_person(text: str) -> HiddenPerson:
        """kedi
        >> Extract [person: HiddenPerson] from <text>.
        = `person`
        """

    assert extract_person("Ada is 42.") == HiddenPerson(name="Ada", age=42)
    assert adapter.schemas[0] == {"person": HiddenPerson}


def test_kedi_type_rejects_non_classes() -> None:
    with pytest.raises(TypeError, match="classes"):
        kedi_type(lookup)  # type: ignore[arg-type]


def test_auto_type_env_ignores_callables_without_module_name() -> None:
    def local() -> None:
        return None

    local.__module__ = None  # type: ignore[assignment]

    assert api_module._auto_type_env(local) == {}


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


def test_async_query_yields_event_loop_while_blocked() -> None:
    # B4: the async query must offload blocking model I/O to a thread so the event loop keeps
    # serving other coroutines instead of freezing for the duration of the model call.
    import threading

    api_module._PARSE_CACHE.clear()
    api_module._RESPONSE_CACHE.clear()
    in_produce = threading.Event()
    release = threading.Event()

    class BlockingAdapter(ApiAdapter):
        def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
            in_produce.set()
            release.wait(2)
            return super().produce_sync(template=template, output_schema=output_schema, **kwargs)

    configure(model="m", adapter=BlockingAdapter())

    @query
    async def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    async def scenario() -> str:
        task = asyncio.create_task(answer())
        for _ in range(2000):
            if in_produce.is_set():
                break
            await asyncio.sleep(0.001)
        # The worker is parked inside produce_sync; if the loop had been blocked by run(), the
        # task would already be finished by the time control returns here.
        assert in_produce.is_set()
        assert not task.done()
        release.set()
        return await task

    assert asyncio.run(scenario()) == "ok"


def test_query_runs_under_parallel_context(adapter: ApiAdapter) -> None:
    # kedi.parallel() opts the query into the threaded engine; a single-template query still
    # returns its concrete result (drained + forced at the API boundary).
    @kedi.query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    with kedi.parallel(max_workers=2):
        assert answer() == "ok"


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


def test_configure_env_overrides_call_arguments() -> None:
    adapter = ApiAdapter()
    configure(adapter=adapter, env={"topic": "configured topic"})

    @query
    def summarize(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    assert summarize("call topic") == "short summary"
    assert "configured topic" in adapter.templates[0]
    assert "call topic" not in adapter.templates[0]


def test_context_env_overrides_config_env() -> None:
    adapter = ApiAdapter()
    configure(adapter=adapter, env={"topic": "configured topic"})

    @query
    def summarize(topic: str) -> str:
        """kedi
        >> Summarize <topic>. Return [summary].
        = <summary>
        """

    with context(env={"topic": "context topic"}):
        assert summarize("call topic") == "short summary"

    assert "context topic" in adapter.templates[0]
    assert "configured topic" not in adapter.templates[0]
    assert "call topic" not in adapter.templates[0]


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
    monkeypatch.setattr(pydantic_adapter_module, "PydanticAdapter", fake_adapter)
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

    @query(tools=[flaky_lookup])
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


def test_query_registers_callable_tools(adapter: ApiAdapter) -> None:
    @query(tools=[lookup])
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert adapter.tools == [("lookup",)]


def test_bind_file_exposes_kedi_source_as_python_callable(
    adapter: ApiAdapter,
    tmp_path: Path,
) -> None:
    source = tmp_path / "summarize.kedi"
    source.write_text(
        ">> Summarize <topic>. Return [summary].\n= <summary>\n",
        encoding="utf-8",
    )

    @kedi.bind(file=source)
    def summarize(topic: str) -> str:
        raise AssertionError("bound function body should not run")

    assert summarize("Kedi") == "short summary"
    assert adapter.templates == [
        "The template string must maintain grammatical and semantic integrity once the output "
        "fields are substituted into their corresponding locations in the template after the "
        "final construction.\n"
        "Fields to be substituted into the template string for final construction: (summary).\n\n"
        "The template string:\n"
        "Summarize Kedi. Return [summary]."
    ]


def test_bind_parse_cache_reuses_identical_file_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_module._PARSE_CACHE.clear()
    calls = 0
    real_parse = api_module.parse_program
    source = tmp_path / "answer.kedi"
    source.write_text(">> Return [answer].\n= <answer>\n", encoding="utf-8")

    def counting_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(api_module, "parse_program", counting_parse)

    @bind(file=source)
    def first() -> str: ...

    @bind(file=source)
    def second() -> str: ...

    assert first is not second
    assert calls == 1


def test_bind_response_cache_reuses_same_call(adapter: ApiAdapter, tmp_path: Path) -> None:
    source = tmp_path / "summarize.kedi"
    source.write_text(">> Summarize <topic>. Return [summary].\n= <summary>\n", encoding="utf-8")

    @bind(file=source, cache=True)
    def summarize(topic: str) -> str: ...

    assert summarize("Kedi") == "short summary"
    assert summarize("Kedi") == "short summary"
    assert len(adapter.templates) == 1


def test_bind_reload_rereads_file_and_uses_new_hash(adapter: ApiAdapter, tmp_path: Path) -> None:
    source = tmp_path / "summarize.kedi"
    source.write_text(">> First <topic>. Return [summary].\n= <summary>\n", encoding="utf-8")

    @bind(file=source, reload=True, cache=True)
    def summarize(topic: str) -> str: ...

    assert summarize("Kedi") == "short summary"
    assert "First Kedi" in adapter.templates[-1]
    source.write_text(">> Second <topic>. Return [summary].\n= <summary>\n", encoding="utf-8")

    assert summarize("Kedi") == "short summary"
    assert "Second Kedi" in adapter.templates[-1]
    assert len(adapter.templates) == 2


def test_bind_without_reload_keeps_decorated_file_source(
    adapter: ApiAdapter,
    tmp_path: Path,
) -> None:
    source = tmp_path / "summarize.kedi"
    source.write_text(">> First <topic>. Return [summary].\n= <summary>\n", encoding="utf-8")

    @bind(file=source)
    def summarize(topic: str) -> str: ...

    source.write_text(">> Second <topic>. Return [summary].\n= <summary>\n", encoding="utf-8")

    assert summarize("Kedi") == "short summary"
    assert "First Kedi" in adapter.templates[-1]


def test_bind_file_profile_arguments_apply_for_one_call(
    adapter: ApiAdapter,
    tmp_path: Path,
) -> None:
    source = tmp_path / "answer.kedi"
    source.write_text(
        "> use: lookup\n>> Call the available tool if useful and return [answer].\n= <answer>\n",
        encoding="utf-8",
    )

    @bind(
        file=source,
        system="Use terse prose.",
        settings={"temperature": 0.2},
        tools=[lookup],
    )
    def answer() -> str: ...

    assert answer() == "ok"
    assert adapter.instructions == ["Use terse prose."]
    assert adapter.settings == [{"temperature": 0.2}]
    assert adapter.tools == [("lookup",)]


def test_bind_file_env_overrides_call_arguments(adapter: ApiAdapter, tmp_path: Path) -> None:
    source = tmp_path / "extract.kedi"
    source.write_text(
        ">> Extract [output: `output_type`] from <template>.\n= `output`\n",
        encoding="utf-8",
    )

    @bind(file=source, env={"output_type": Person})
    def extract_output(*, template: str, output_type: type[Any]) -> Person: ...

    result = extract_output(template="John is 30 years old.", output_type=str)

    assert result == Person(name="John", age=30)
    assert adapter.schemas[0] == {"output": Person}


def test_bind_file_resolves_relative_to_python_source_file(
    adapter: ApiAdapter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner.py"
    owner.write_text("# owner module\n", encoding="utf-8")
    (tmp_path / "relative.kedi").write_text(
        ">> Return [answer] about <topic>.\n= <answer>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module.inspect, "getsourcefile", lambda fn: str(owner))

    @bind(file="relative.kedi")
    def answer(topic: str) -> str: ...

    assert answer("Kedi") == "ok"
    assert "Kedi" in adapter.templates[0]


def test_bind_file_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.kedi"

    with pytest.raises(FileNotFoundError, match="Kedi bind file not found"):

        @bind(file=missing)
        def answer() -> str: ...


def test_configure_default_tools_are_available_to_use_directive(
    adapter: ApiAdapter,
) -> None:
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


def test_configure_builds_default_adapter_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ApiAdapter()
    seen_models: list[str | None] = []

    def fake_adapter(model: str | None) -> ApiAdapter:
        seen_models.append(model)
        return adapter

    monkeypatch.setattr(pydantic_adapter_module, "PydanticAdapter", fake_adapter)
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


def test_configure_uses_adapter_env_when_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ApiAdapter()
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_adapter(model: str, **kwargs: Any) -> ApiAdapter:
        seen.append((model, kwargs))
        return adapter

    monkeypatch.setenv("KEDI_ADAPTER", "dspy")
    monkeypatch.setenv("KEDI_ADAPTER_MODEL", "openrouter:env-model")
    monkeypatch.setattr(dspy_adapter_module, "DSPyAdapter", fake_adapter)
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


def test_configure_explicit_model_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ApiAdapter()
    seen_models: list[str | None] = []

    def fake_adapter(model: str | None) -> ApiAdapter:
        seen_models.append(model)
        return adapter

    monkeypatch.setenv("KEDI_ADAPTER_MODEL", "openrouter:env-model")
    monkeypatch.setattr(pydantic_adapter_module, "PydanticAdapter", fake_adapter)
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

    monkeypatch.setattr(dspy_adapter_module, "DSPyAdapter", fake_adapter)
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


def test_configure_builds_langchain_adapter_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ApiAdapter()
    seen: list[tuple[str | None, dict[str, Any]]] = []

    def fake_adapter(model: str | None, **kwargs: Any) -> ApiAdapter:
        seen.append((model, kwargs))
        return adapter

    monkeypatch.setattr(langchain_adapter_module, "LangChainAdapter", fake_adapter)
    configure(model="openrouter:test-model", adapter="langchain", temperature=0.3)

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
    with pytest.raises(ValueError, match="unknown adapter"):
        configure(adapter="unknown")  # type: ignore[arg-type]


def test_framework_and_harness_selections_are_distinct() -> None:
    with pytest.raises(ValueError, match="pass it with agent="):
        configure(adapter="claude")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pass it with adapter="):
        configure(agent="pydantic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure(adapter="pydantic", agent="claude")

    with pytest.raises(ValueError, match="pass it with agent="):
        configure(adapter=HarnessApiAdapter())
    with pytest.raises(ValueError, match="pass it with adapter="):
        configure(agent=ApiAdapter())


def test_configure_accepts_harness_agent_instance() -> None:
    harness = HarnessApiAdapter()
    configure(agent=harness, model="harness:model")

    @query
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert harness.models == ["harness:model"]


def test_query_locally_overrides_framework_and_model() -> None:
    outer = ApiAdapter()
    local = ApiAdapter()
    local.shortname = "local-framework"
    configure(adapter=outer, model="outer:model")

    @query(adapter=local, model="local:model")
    def local_answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    @query
    def outer_answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert local_answer() == "ok"
    assert outer_answer() == "ok"
    assert local.models == ["local:model"]
    assert outer.models == ["outer:model"]


def test_query_locally_overrides_agent_harness() -> None:
    outer = ApiAdapter()
    harness = HarnessApiAdapter()
    configure(adapter=outer, model="outer:model")

    @query(agent=harness, model="harness:model")
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert harness.models == ["harness:model"]
    assert outer.templates == []


def test_query_selects_harness_by_shortname(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = HarnessApiAdapter()
    configure(adapter=ApiAdapter())
    monkeypatch.setattr(adapter_registry, "build_adapter", lambda *args, **kwargs: harness)

    @query(agent="codex", model="gpt-test")
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    assert harness.models == ["gpt-test"]


def test_bind_locally_overrides_agent_harness(tmp_path: Path) -> None:
    source = tmp_path / "bound.kedi"
    source.write_text(">> Return [answer].\n= <answer>\n", encoding="utf-8")
    harness = HarnessApiAdapter()
    configure(adapter=ApiAdapter())

    @bind(file=source, agent=harness, model="harness:model")
    def answer() -> str: ...

    assert answer() == "ok"
    assert harness.models == ["harness:model"]


def test_response_cache_splits_by_adapter_instance() -> None:
    first = ApiAdapter()
    second = ApiAdapter()
    configure(adapter=first)

    @query(cache=True)
    def answer() -> str:
        """kedi
        >> Return [answer].
        = <answer>
        """

    assert answer() == "ok"
    with context(adapter=second):
        assert answer() == "ok"
    assert len(first.templates) == 1
    assert len(second.templates) == 1


def test_configure_reads_agent_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    monkeypatch.delenv("KEDI_ADAPTER", raising=False)
    monkeypatch.setenv("KEDI_AGENT", "claude")

    configure()

    config = api_module._config_var.get()
    assert config.adapter_kind == "agent-harness"
    assert config.adapter_shortname == "claude"
    assert config.adapter is not None
    assert config.adapter.kind == "agent-harness"
    assert config.adapter.shortname == "claude"


def test_configure_rejects_conflicting_environment_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module.dotenv, "load_dotenv", lambda: False)
    monkeypatch.setenv("KEDI_ADAPTER", "pydantic")
    monkeypatch.setenv("KEDI_AGENT", "claude")

    with pytest.raises(ValueError, match="KEDI_ADAPTER and KEDI_AGENT"):
        configure()


def test_configure_rejects_unknown_agent_and_incomplete_adapter() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        configure(agent="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="define kind and shortname"):
        configure(adapter=object())  # type: ignore[arg-type]


def test_profile_cache_identity_includes_adapter_selection_and_command() -> None:
    base = AgentProfile(model="same")
    selected = AgentProfile(
        model="same",
        adapter_kind="agent-harness",
        adapter_shortname="acp",
        agent_command=("uv", "run", "agent"),
    )

    assert api_module._profile_identity(base) != api_module._profile_identity(selected)


def test_public_python_api_exports_adapter_types_and_mcp_spec() -> None:
    assert kedi.AdapterLike is api_module.AdapterLike
    assert kedi.McpServerSpec.__name__ == "McpServerSpec"
