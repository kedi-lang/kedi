from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from kedi.agent_adapter import registry
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_profile import AgentProfile
from kedi.app import _build_lazy_runtime_adapter, run_program_cli
from kedi.cli import main as cli_main
from kedi.engine import SequentialEngine, ThreadEngine
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


@pytest.fixture
def backends(monkeypatch):
    built = []

    def register(name, kind):
        def build(model, kwargs):
            adapter = MockAdapter(default_model=model)
            adapter.shortname = name
            adapter.kind = kind
            built.append(adapter)
            return adapter

        monkeypatch.setitem(registry._ADAPTERS, name, registry.AdapterFactory(name, kind, build))

    register("pydantic", "agent-framework")
    register("langchain", "agent-framework")
    register("codex", "agent-harness")
    return built


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_framework_switch_preserves_model_identifier(backends, lazy, explicit):
    base = MockAdapter(default_model="openai:original")
    base.shortname = "pydantic"
    adapter = (
        LazyAdapter(lambda: base, shortname="pydantic", model="openai:original") if lazy else base
    )
    source = "> adapter: langchain\n"
    if explicit:
        source += "> model: openai:override\n"
    source += ">> Give a short answer.\n"
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
        assert len(backends) == 1
        assert backends[0].model_calls == ["openai:override" if explicit else "openai:original"]
        assert not base.model_calls
        if lazy:
            assert not adapter.is_initialized
    finally:
        asyncio.run(runtime.aclose())


def test_framework_without_profile_support_does_not_invent_model(backends, monkeypatch):
    from types import SimpleNamespace

    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime, "adapter", SimpleNamespace(shortname="minimal", kind="agent-framework")
            )
            selected = runtime._resolve_adapter(
                AgentProfile(adapter_kind="agent-framework", adapter_shortname="langchain")
            )
        assert selected is backends[0]
        assert selected.default_model() is None
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("scope", ["top", "procedure", "branch", "loop"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_sequential_framework_and_harness_switches_are_lexical(backends, scope, reverse, parallel):
    selections = [("adapter", "langchain"), ("agent", "codex")]
    if reverse:
        selections.reverse()
    body = (
        f"> {selections[0][0]}: {selections[0][1]}\n>> First request.\n"
        f"> {selections[1][0]}: {selections[1][1]}\n>> Second request.\n"
    )
    indented = "".join("    " + line + "\n" for line in body.splitlines())
    if scope == "procedure":
        source = "@work():\n" + indented + "    = done\n> show: <work()>\n"
    elif scope == "branch":
        source = "> if: `True`:\n" + indented
    elif scope == "loop":
        source = "> loop [item]: `[1]`:\n" + indented
    else:
        source = body
    source += ">> Final request.\n"
    base = MockAdapter(default_model="openai:original")
    base.shortname = "pydantic"
    engine = ThreadEngine(max_workers=3) if parallel else SequentialEngine()
    runtime = compile_program(
        parse_program(source),
        adapter=base,
        engine=engine,
        default_agent_profile=AgentProfile(
            adapter_kind="agent-framework", adapter_shortname="pydantic"
        ),
    )
    try:
        runtime.run_main()
        assert [a.shortname for a in backends] == [name for _, name in selections]
        assert backends[0].request_calls == ["First request."]
        expected = ["Second request.", "Final request."] if scope == "top" else ["Second request."]
        if parallel:
            # Independent calls keep their lexical bindings, not dispatch order.
            assert sorted(backends[1].request_calls) == sorted(expected)
        else:
            assert backends[1].request_calls == expected
        assert base.request_calls == ([] if scope == "top" else ["Final request."])
        harness = next(a for a in backends if a.shortname == "codex")
        assert harness.default_model() is None
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


def test_native_langchain_receives_model_without_initializing_default_framework():
    from kedi.agent_adapter.adapters.langchain import LangChainAdapter

    adapter = _build_lazy_runtime_adapter("pydantic", "openai:gpt-5.6-luna")
    runtime = compile_program(parse_program(""), adapter=adapter)
    try:
        selected = runtime._resolve_adapter(
            AgentProfile(adapter_kind="agent-framework", adapter_shortname="langchain")
        )
        assert isinstance(selected, LangChainAdapter)
        assert selected.default_model() == "openai:gpt-5.6-luna"
        assert selected._model_for_run() is not None
        assert not adapter.is_initialized
    finally:
        asyncio.run(runtime.aclose())


def test_procedures_capture_selection_at_definition_site(backends):
    source = (
        "> adapter: langchain\n@first():\n    >> First request.\n    = done\n"
        "> agent: codex\n@second():\n    >> Second request.\n    = done\n"
        "> show: <first()>\n> show: <second()>\n"
    )
    runtime = compile_program(parse_program(source))
    try:
        runtime.run_main()
        assert [a.shortname for a in backends] == ["langchain", "codex"]
        assert [a.request_calls for a in backends] == [["First request."], ["Second request."]]
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("selection", ["langchain", "codex"])
def test_cli_framework_default_can_be_overridden_without_initializing_it(
    backends, tmp_path, selection
):
    program = tmp_path / "switch.kedi"
    directive = "agent" if selection == "codex" else "adapter"
    program.write_text(f"> {directive}: {selection}\n>> Give a short answer.\n", encoding="utf-8")
    code, output = run_program_cli(str(program), adapter="pydantic", model="openai:original")
    assert code == 0, output
    assert [a.shortname for a in backends] == [selection]
    assert backends[0].model_calls == ([None] if selection == "codex" else ["openai:original"])


def test_lazy_model_hint_does_not_initialize_backend():
    def fail():
        raise AssertionError("Unused adapter must not be initialized")

    adapter = LazyAdapter(fail, model="openai:original")
    assert adapter.configured_model == "openai:original"
    assert not adapter.is_initialized
    assert _build_lazy_runtime_adapter("pydantic", None).configured_model == "groq:qwen/qwen3-32b"


def test_lazy_model_without_hint_is_only_read_after_initialization():
    base = MockAdapter(default_model="openai:original")
    adapter = LazyAdapter(lambda: base)
    assert adapter.configured_model is None
    assert not adapter.is_initialized
    adapter.resolve()
    assert adapter.configured_model == "openai:original"


@pytest.mark.parametrize("selection", ["langchain", "codex"])
def test_environment_selected_framework_can_be_overridden(backends, tmp_path, selection):
    program = tmp_path / "environment-switch.kedi"
    directive = "agent" if selection == "codex" else "adapter"
    program.write_text(f"> {directive}: {selection}\n>> Give a short answer.\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli_main,
        [str(program)],
        env={
            "KEDI_ADAPTER": "pydantic",
            "KEDI_ADAPTER_MODEL": "openai:environment-model",
            "PYTHON_DOTENV_DISABLED": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert [a.shortname for a in backends] == [selection]
    assert backends[0].model_calls == (
        [None] if selection == "codex" else ["openai:environment-model"]
    )


def test_python_configuration_default_does_not_lock_source_agent_selection(backends, monkeypatch):
    from kedi.api.configuration import configure
    from kedi.api.state import _config_var

    monkeypatch.setenv("KEDI_ADAPTER", "pydantic")
    monkeypatch.delenv("KEDI_AGENT", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    token = _config_var.set(_config_var.get())
    try:
        configure(model=None)
        config = _config_var.get()
    finally:
        _config_var.reset(token)
    runtime = compile_program(
        parse_program("> agent: codex\n>> Give a short answer.\n"),
        adapter=config.adapter,
        default_agent_profile=config.profile,
    )
    try:
        runtime.run_main()
        assert [a.shortname for a in backends] == ["codex"]
        assert not config.adapter.is_initialized
    finally:
        asyncio.run(runtime.aclose())
