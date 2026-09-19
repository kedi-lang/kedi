from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_profile import AgentProfile, AgentProfileOverrideUnsupported
from kedi.agent_profile.settings import (
    acp_agent_settings,
    claude_agent_settings,
    codex_agent_settings,
    dspy_lm_settings,
    langchain_model_settings,
    pydantic_model_settings,
)
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize("adapter_class", [PydanticAdapter, LangChainAdapter])
@pytest.mark.parametrize("default", [True, False])
def test_profile_overrides_constructor_without_mutating_it(adapter_class, default):
    adapter = adapter_class(tool_reason=default)
    assert adapter.active_tool_reason is default
    with adapter.profile_override(AgentProfile(settings={"tool_reason": not default})):
        assert adapter.active_tool_reason is not default
        assert adapter.tool_reason is default
        with adapter.profile_override(AgentProfile(settings={"tool_reason": default})):
            assert adapter.active_tool_reason is default
        assert adapter.active_tool_reason is not default
    assert adapter.active_tool_reason is default


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, [], {}])
def test_profile_requires_real_boolean(value):
    with pytest.raises(TypeError, match="tool_reason must be a boolean"):
        AgentProfile(settings={"tool_reason": value})


@pytest.mark.parametrize("literal", ["`1`", "`None`", '`"enabled"`', '`"true"`'])
def test_dsl_rejects_non_boolean_values(literal):
    with pytest.raises(Exception, match="tool_reason must be a boolean"):
        runtime = compile_program(
            parse_program(f"> settings:\n    tool_reason: {literal}\n= done\n")
        )
        runtime.run_main()


@pytest.mark.parametrize("literal", ["true", "false", "1", "null", "on", "off"])
def test_dsl_rejects_invalid_plain_values_at_parse_time(literal):
    from kedi.lang.parser.diagnostics import KediParseError

    with pytest.raises(KediParseError, match="tool_reason must be enabled, disabled"):
        parse_program(f"> settings:\n    tool_reason: {literal}\n")


def test_direct_ast_setting_cannot_bypass_plain_value_validation():
    from kedi.agent_profile.directives import eval_settings_field_value
    from kedi.core import KediRuntime
    from kedi.lang.ast import SettingsField

    with pytest.raises(ValueError, match="tool_reason must be enabled, disabled"):
        eval_settings_field_value(
            KediRuntime(), SettingsField(name="tool_reason", plain_value="true"), {}
        )


@pytest.mark.parametrize("env_value", [None, "", "0", "yes"])
def test_dsl_evaluates_environment_expression(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("TOOL_REASON_FOO_BAR", raising=False)
    else:
        monkeypatch.setenv("TOOL_REASON_FOO_BAR", env_value)
    observed = []

    def respond(messages, info):
        observed.append("reason" in info.function_tools[0].parameters_json_schema["properties"])
        assert "tool_reason" not in (info.model_settings or {})
        return ModelResponse(parts=[TextPart("done")])

    source = """
```
import os
```

> settings:
    tool_reason: `os.getenv("TOOL_REASON_FOO_BAR") is not None`

@write_note(content: str) -> str:
    = <content>

> use: write_note
[result] << Inspect the tool.
= <result>
"""
    adapter = PydanticAdapter(FunctionModel(respond))
    assert compile_program(parse_program(source), adapter=adapter).run_main() == "done"
    assert observed == [env_value is not None]


@pytest.mark.parametrize(
    "filter_settings",
    [
        pydantic_model_settings,
        langchain_model_settings,
        dspy_lm_settings,
        claude_agent_settings,
        codex_agent_settings,
        acp_agent_settings,
    ],
)
def test_runtime_setting_never_enters_provider_kwargs(filter_settings):
    assert "tool_reason" not in filter_settings({"tool_reason": True, "temperature": 0.2})


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("use_profile", [False, True])
def test_kedi_program_settings_are_lexical(backend, use_profile):
    observations = []
    if backend == "pydantic":

        def respond(messages, info):
            assert "tool_reason" not in (info.model_settings or {})
            prompt = next(
                p.content
                for m in messages
                for p in m.parts
                if getattr(p, "part_kind", None) == "user-prompt"
            )
            observations.append(
                (prompt, "reason" in info.function_tools[0].parameters_json_schema["properties"])
            )
            return ModelResponse(parts=[TextPart("done")])

        adapter = PydanticAdapter(FunctionModel(respond))
    else:
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        class Model(FakeMessagesListChatModel):
            reason_present: bool = False

            def bind_tools(self, tools, **kwargs):
                schema = tools[0].args_schema
                if not isinstance(schema, dict):
                    schema = schema.model_json_schema()
                return self.model_copy(update={"reason_present": "reason" in schema["properties"]})

            def _generate(self, messages, *args, **kwargs):
                assert "tool_reason" not in kwargs
                prompt = next(m.content for m in messages if m.type == "human")
                observations.append((prompt, self.reason_present))
                return super()._generate(messages, *args, **kwargs)

        adapter = LangChainAdapter(Model(responses=[AIMessage(content="done")]))
    setting = "> settings:\n    tool_reason: enabled\n"
    if use_profile:
        setting = (
            "> profile: audited:\n    > settings:\n        tool_reason: enabled\n> use: audited\n"
        )
    source = (
        """
@write_note(content: str) -> str:
    = <content>

> use: write_note
"""
        + setting
        + """
@quiet():
    > settings:
        tool_reason: disabled
    [result] << Inner.
    = <result>

[first] << Outer before.
[inner] = `quiet()`
[last] << Outer after.
= `" ".join((first, inner, last))`
"""
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    assert runtime.run_main() == "done done done"
    assert sorted(observations) == [
        ("Inner.", False),
        ("Outer after.", True),
        ("Outer before.", True),
    ]
    assert adapter.tool_reason is False
    assert adapter.active_tool_reason is False


def test_unsupported_adapter_does_not_silently_ignore_enabled_reason():
    from kedi.core import KediRuntime
    from tests.mock_adapter import MockAdapter

    with pytest.raises(AgentProfileOverrideUnsupported, match="tool_reason is not supported"):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(settings={"tool_reason": True}), MockAdapter()
        )


def test_setting_has_lsp_help_and_semantic_highlighting():
    from kedi.lsp.features.semantic_tokens import SEMANTIC_TOKEN_TYPES, compute_semantic_tokens_full
    from tests.lsp_helpers import decode_semantic_tokens, hover_text_at

    source = "> settings:\n    tool_reason: enabled\n"
    parse_program(source)
    help_text = hover_text_at(source, "reason.kedi", "tool_reason")
    assert "Pydantic AI and LangChain" in help_text
    assert "not sent to the model provider" in help_text
    tokens = decode_semantic_tokens(compute_semantic_tokens_full(source, "/tmp/reason.kedi").data)
    assert any(
        line == 1 and kind == SEMANTIC_TOKEN_TYPES.index("property")
        for line, _, _, kind, _ in tokens
    )
