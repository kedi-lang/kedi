from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, cast

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse, TextPart
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext, ConversationState
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.prefix_cache import (
    ProviderCachePolicy,
    langchain_cache_settings,
    langchain_prefix_cache_settings,
    langchain_provider_cache_policy,
    pydantic_cache_settings,
    pydantic_prefix_cache_settings,
    pydantic_provider_cache_policy,
)
from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import HistoryDirectiveStmt, ProfileDefStmt
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features import compute_diagnostics, compute_semantic_tokens_full
from kedi.artifacts import ArtifactHandle, ArtifactManager, ArtifactPolicy
from tests.lsp_helpers import decode_semantic_tokens, hover_text_at
from tests.mock_adapter import MockAdapter


class StatefulHistoryAdapter(MockAdapter):
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_artifacts=True,
        supports_stateful_history=True,
        supports_history_replay=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self._conversation: ContextVar[AdapterConversationContext | None] = ContextVar(
            "history_test_conversation",
            default=None,
        )
        self.resume_states: list[dict[str, Any] | None] = []
        self.fail = False

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Iterator[None]:
        token = self._conversation.set(conversation)
        try:
            yield
        finally:
            self._conversation.reset(token)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        conversation = self._conversation.get()
        self.resume_states.append(None if conversation is None else dict(conversation.resume_state))
        if self.fail:
            raise RuntimeError("history test failure")
        if conversation is None:
            return "stateless"
        turn = int(conversation.resume_state.get("turn", 0)) + 1
        conversation.result_state["turn"] = turn
        return f"response-{turn}"


def _openai_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        "gpt-5.4",
        provider=OpenAIProvider(api_key="test"),
    )


def _openai_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        "gpt-5.4",
        provider=OpenAIProvider(api_key="test"),
    )


def _anthropic_model() -> AnthropicModel:
    return AnthropicModel(
        "claude-opus-4-6",
        provider=AnthropicProvider(api_key="test"),
    )


def _openrouter_model(*, settings: OpenRouterModelSettings | None = None) -> OpenRouterModel:
    return OpenRouterModel(
        "openai/gpt-5.6-luna",
        provider=OpenRouterProvider(api_key="test"),
        settings=settings,
    )


def _google_model() -> GoogleModel:
    return GoogleModel(
        "gemini-3.1-flash-lite-preview",
        provider=GoogleProvider(api_key="test"),
    )


def test_history_directive_parses_at_top_level_procedure_and_profile() -> None:
    source = """
> history: enabled

> profile: stateless:
    > history: disabled

@remember():
    > history: enabled
    = ok
"""
    program = parse_program(source)

    directive = program.ordered_toplevel[0]
    assert isinstance(directive, HistoryDirectiveStmt)
    assert directive.enabled is True
    profile = program.ordered_toplevel[1]
    assert isinstance(profile, ProfileDefStmt)
    assert profile.history_members[0].enabled is False


@pytest.mark.parametrize("value", ["true", "on", "yes", "automatic", "`enabled`"])
def test_history_directive_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(KediParseError, match="history directive|unexpected source"):
        parse_program(f"> history: {value}\n")


def test_history_defaults_to_stateless_and_enabled_reuses_checkpoint() -> None:
    stateless_adapter = StatefulHistoryAdapter()
    stateless = compile_program(
        parse_program("[answer] << Stateless.\n= <answer>\n"),
        adapter=stateless_adapter,
    )
    assert stateless.run_main() == "stateless"
    assert stateless_adapter.resume_states == [None]

    adapter = StatefulHistoryAdapter()
    runtime = compile_program(
        parse_program("> history: enabled\n[first] << First.\n[second] << Second.\n= <second>\n"),
        adapter=adapter,
    )
    assert runtime.run_main() == "response-2"
    assert adapter.resume_states == [{}, {"turn": 1}]


def test_profile_history_policy_enables_shared_context() -> None:
    adapter = StatefulHistoryAdapter()
    source = """
> profile: memory:
    > history: enabled

> use: memory
[first] << First.
[second] << Second.
= <second>
"""
    runtime = compile_program(parse_program(source), adapter=adapter)

    assert runtime.run_main() == "response-2"
    assert adapter.resume_states == [{}, {"turn": 1}]


def test_nested_history_disabled_does_not_read_or_mutate_outer_checkpoint() -> None:
    adapter = StatefulHistoryAdapter()
    source = """
@stateless_call():
    > history: disabled
    [middle] << Middle.
    = <middle>

> history: enabled
[first] << First.
`stateless_call()`
[second] << Second.
= <second>
"""
    runtime = compile_program(parse_program(source), adapter=adapter)

    assert runtime.run_main() == "response-2"
    assert adapter.resume_states == [{}, None, {"turn": 1}]


def test_failed_history_turn_does_not_commit_implicit_state() -> None:
    adapter = StatefulHistoryAdapter()
    runtime = compile_program(
        parse_program("> history: enabled\n[answer] << Fail.\n= <answer>\n"),
        adapter=adapter,
    )
    adapter.fail = True
    with pytest.raises(Exception, match="history test failure"):
        runtime.run_main()

    adapter.fail = False
    assert runtime.run_main() == "response-1"
    assert adapter.resume_states == [{}, {}]


def test_history_enabled_rejects_adapter_without_stateful_capability() -> None:
    runtime = compile_program(
        parse_program("> history: enabled\n[answer] << Hello.\n= <answer>\n"),
        adapter=MockAdapter(),
    )

    with pytest.raises(Exception, match="does not support stateful history"):
        runtime.run_main()


def test_conversation_cache_keys_are_stable_opaque_adapter_scoped_and_epoch_bound() -> None:
    state = ConversationState(session_id="private-session-id")
    with state.adapter_turn("pydantic") as first:
        first.result_state["turn"] = 1
    with state.adapter_turn("pydantic") as second:
        pydantic_key = second.cache_key
        assert second.resume_state == {"turn": 1}
    with state.adapter_turn("langchain") as other:
        langchain_key = other.cache_key

    assert pydantic_key == first.cache_key
    assert pydantic_key != langchain_key
    assert "private-session-id" not in pydantic_key

    state.advance_cache_epoch()
    with state.adapter_turn("pydantic") as after_epoch:
        assert after_epoch.cache_key != pydantic_key
        assert after_epoch.resume_state == {}
    state.close()


def test_compaction_rotates_cache_epoch_and_seeds_it_with_compacted_state() -> None:
    state = ConversationState(session_id="compaction-session")
    with state.adapter_turn("langchain") as other:
        other.result_state["continuation"] = "stale"
    with state.adapter_turn("pydantic") as first:
        first.result_state["pydantic_messages"] = ["old-prefix"]

    previous_key = first.cache_key
    with state.adapter_turn("pydantic") as compacted:
        compacted.result_state["pydantic_messages"] = ["compacted-prefix"]
        compacted.cache_prefix_changed = True

    assert state.cache_epoch == 1
    with state.adapter_turn("pydantic") as after_compaction:
        assert after_compaction.cache_key != previous_key
        assert after_compaction.resume_state == {"pydantic_messages": ["compacted-prefix"]}
    with state.adapter_turn("langchain") as cleared_lane:
        assert cleared_lane.resume_state == {}
    state.close()


def test_conversation_can_borrow_artifacts_without_owning_their_lifecycle() -> None:
    artifacts = ArtifactManager(session_id="shared-session")
    state = ConversationState(artifacts=artifacts)

    assert state.cache_epoch == 0
    with pytest.raises(ValueError, match="session IDs must match"):
        ConversationState(session_id="other-session", artifacts=artifacts)
    state.close()

    stored = artifacts.artifactize(
        "still open",
        policy=ArtifactPolicy.patch({"enabled": True, "threshold": 1}),
    )
    assert isinstance(stored, ArtifactHandle)
    artifacts.close()


def test_pydantic_prefix_cache_defaults_preserve_explicit_settings() -> None:
    conversation = AdapterConversationContext(cache_key="cache-key")
    defaults = pydantic_prefix_cache_settings(
        {},
        model=_openai_model(),
        conversation=conversation,
    )
    assert defaults == {
        "openai_prompt_cache_key": "cache-key",
        "openai_previous_response_id": "auto",
    }
    explicit = pydantic_prefix_cache_settings(
        {
            "openai_prompt_cache_key": "caller-key",
            "openai_previous_response_id": "response-id",
        },
        model=_openai_model(),
        conversation=conversation,
    )
    assert explicit["openai_prompt_cache_key"] == "caller-key"
    assert explicit["openai_previous_response_id"] == "response-id"
    assert pydantic_prefix_cache_settings(
        {},
        model=_openai_chat_model(),
        conversation=conversation,
    ) == {"openai_prompt_cache_key": "cache-key"}
    assert pydantic_prefix_cache_settings(
        {},
        model=_openrouter_model(),
        conversation=conversation,
    ) == {
        "openai_prompt_cache_key": "cache-key",
        "extra_body": {
            "session_id": "cache-key",
            "cache_control": {"type": "ephemeral"},
        },
    }
    openrouter_explicit = pydantic_prefix_cache_settings(
        {"extra_body": {"run_setting": True}},
        model=_openrouter_model(
            settings=cast(
                OpenRouterModelSettings,
                {
                    "openai_prompt_cache_key": "model-key",
                    "extra_body": {
                        "session_id": "model-session",
                        "model_setting": True,
                    },
                },
            )
        ),
        conversation=conversation,
    )
    assert "openai_prompt_cache_key" not in openrouter_explicit
    assert openrouter_explicit["extra_body"] == {
        "session_id": "model-session",
        "model_setting": True,
        "run_setting": True,
        "cache_control": {"type": "ephemeral"},
    }
    invalid_extra_body = pydantic_prefix_cache_settings(
        {"extra_body": "provider-validates-this"},
        model=_openrouter_model(),
        conversation=conversation,
    )
    assert invalid_extra_body == {
        "extra_body": "provider-validates-this",
        "openai_prompt_cache_key": "cache-key",
    }

    anthropic = pydantic_prefix_cache_settings(
        {"anthropic_cache_messages": False},
        model=_anthropic_model(),
        conversation=conversation,
    )
    assert anthropic == {"anthropic_cache_messages": False}
    assert pydantic_prefix_cache_settings(
        {},
        model=_anthropic_model(),
        conversation=conversation,
    ) == {"anthropic_cache": True}
    assert pydantic_prefix_cache_settings(
        {"temperature": 0.1},
        model=_google_model(),
        conversation=conversation,
    ) == {"temperature": 0.1}


def test_provider_cache_policy_matrix_covers_supported_backends() -> None:
    assert pydantic_provider_cache_policy(_openai_model()) == ProviderCachePolicy(
        provider="openai",
        backend="openai_responses",
        prefix_cache_mode="automatic",
        stable_cache_key=True,
        continuation=True,
    )
    assert pydantic_provider_cache_policy(_openai_chat_model()).backend == "openai_chat"
    assert pydantic_provider_cache_policy(_anthropic_model()).prefix_cache_mode == (
        "explicit_breakpoints"
    )
    assert pydantic_provider_cache_policy(_openrouter_model()).exact_response_cache is True
    google = pydantic_provider_cache_policy(_google_model())
    assert google == ProviderCachePolicy(
        provider="google",
        backend="google",
        prefix_cache_mode="automatic",
        explicit_resource_cache=True,
    )
    assert pydantic_provider_cache_policy(object()).prefix_cache_mode == "unsupported"

    assert langchain_provider_cache_policy("google_genai:gemini-3.5-flash") == google
    assert langchain_provider_cache_policy("google-vertexai:gemini-3.5-flash") == google
    assert langchain_provider_cache_policy("openai:gpt-5.4").continuation is True
    assert langchain_provider_cache_policy("gpt-5.4").backend == "unknown"
    assert langchain_provider_cache_policy(object()).backend == "unknown"


def test_langchain_prefix_cache_defaults_are_provider_aware_and_non_destructive() -> None:
    conversation = AdapterConversationContext(cache_key="cache-key")
    defaults = langchain_prefix_cache_settings(
        {},
        model="openai:gpt-5.4",
        conversation=conversation,
    )
    assert defaults == {
        "model_kwargs": {"prompt_cache_key": "cache-key"},
        "use_previous_response_id": True,
    }
    explicit = langchain_prefix_cache_settings(
        {
            "model_kwargs": {"prompt_cache_key": "caller-key", "store": False},
            "use_previous_response_id": False,
        },
        model="openai:gpt-5.4",
        conversation=conversation,
    )
    assert explicit == {
        "model_kwargs": {"prompt_cache_key": "caller-key", "store": False},
        "use_previous_response_id": False,
    }
    assert langchain_prefix_cache_settings(
        {"temperature": 0.2},
        model="anthropic:claude-opus-4-6",
        conversation=conversation,
    ) == {"temperature": 0.2}
    assert langchain_prefix_cache_settings(
        {},
        model="openrouter:openai/gpt-5.6-luna",
        conversation=conversation,
    ) == {
        "session_id": "cache-key",
        "model_kwargs": {"cache_control": {"type": "ephemeral"}},
    }

    chat_openai = type("ChatOpenAI", (), {"__module__": "langchain_openai.chat_models"})()
    chat_anthropic = type(
        "ChatAnthropic",
        (),
        {"__module__": "langchain_anthropic.chat_models"},
    )()
    chat_openrouter = type(
        "ChatOpenRouter",
        (),
        {"__module__": "langchain_openrouter.chat_models", "session_id": "caller-session"},
    )()
    chat_google = type(
        "ChatGoogleGenerativeAI",
        (),
        {"__module__": "langchain_google_genai.chat_models"},
    )()
    assert (
        langchain_prefix_cache_settings(
            {},
            model=chat_openai,
            conversation=conversation,
        )["use_previous_response_id"]
        is True
    )
    assert (
        langchain_prefix_cache_settings(
            {},
            model=chat_anthropic,
            conversation=conversation,
        )
        == {}
    )
    assert langchain_prefix_cache_settings(
        {},
        model=chat_openrouter,
        conversation=conversation,
    ) == {
        "session_id": "caller-session",
        "model_kwargs": {"cache_control": {"type": "ephemeral"}},
    }
    assert langchain_prefix_cache_settings(
        {"temperature": 0.1},
        model=chat_google,
        conversation=conversation,
    ) == {"temperature": 0.1}
    assert (
        langchain_prefix_cache_settings(
            {},
            model=object(),
            conversation=conversation,
        )
        == {}
    )


def test_openrouter_response_cache_is_explicit_validated_and_provider_specific() -> None:
    pydantic_settings = pydantic_cache_settings(
        {
            "response_cache": True,
            "response_cache_ttl": 3600,
            "extra_headers": {"X-Caller": "preserved"},
        },
        model=_openrouter_model(),
    )
    assert pydantic_settings == {
        "extra_headers": {
            "X-Caller": "preserved",
            "X-OpenRouter-Cache": "true",
            "X-OpenRouter-Cache-TTL": "3600",
        }
    }
    langchain_settings = langchain_cache_settings(
        {
            "response_cache": True,
            "response_cache_ttl": 60,
            "model_kwargs": {"temperature": 0},
        },
        model="openrouter:google/gemini-3.1-flash-lite-preview",
    )
    assert langchain_settings == {
        "model_kwargs": {
            "temperature": 0,
            "http_headers": {
                "X-OpenRouter-Cache": "true",
                "X-OpenRouter-Cache-TTL": "60",
            },
        }
    }

    assert (
        pydantic_cache_settings(
            {"response_cache": False},
            model=_openai_model(),
        )
        == {}
    )
    with pytest.raises(ValueError, match="not supported by the selected model provider"):
        pydantic_cache_settings(
            {"response_cache": True},
            model=_openai_model(),
        )
    with pytest.raises(ValueError, match="requires response_cache"):
        langchain_cache_settings(
            {"response_cache_ttl": 300},
            model="openrouter:openai/gpt-5.6-luna",
        )
    with pytest.raises(ValueError, match="between 1 and 86400"):
        pydantic_cache_settings(
            {
                "response_cache": True,
                "response_cache_ttl": 86_401,
            },
            model=_openrouter_model(),
        )
    with pytest.raises(ValueError, match="response_cache must be true or false"):
        pydantic_cache_settings(
            {"response_cache": 1},
            model=_openrouter_model(),
        )
    for invalid_ttl in (True, "300"):
        with pytest.raises(ValueError, match="response_cache_ttl must be an integer"):
            pydantic_cache_settings(
                {"response_cache": True, "response_cache_ttl": invalid_ttl},
                model=_openrouter_model(),
            )

    no_ttl = pydantic_cache_settings(
        {"response_cache": True},
        model=_openrouter_model(),
    )
    assert no_ttl == {"extra_headers": {"X-OpenRouter-Cache": "true"}}
    with pytest.raises(ValueError, match="model_kwargs must be a mapping"):
        langchain_cache_settings(
            {"response_cache": True, "model_kwargs": "invalid"},
            model="openrouter:openai/gpt-5.6-luna",
        )


def test_openrouter_response_cache_rejects_conflicting_headers() -> None:
    with pytest.raises(ValueError, match="X-OpenRouter-Cache conflicts"):
        pydantic_cache_settings(
            {
                "response_cache": True,
                "extra_headers": {"x-openrouter-cache": "false"},
            },
            model=_openrouter_model(),
        )
    with pytest.raises(ValueError, match="X-OpenRouter-Cache-TTL conflicts"):
        langchain_cache_settings(
            {
                "response_cache": True,
                "response_cache_ttl": 300,
                "model_kwargs": {"http_headers": {"x-openrouter-cache-ttl": "600"}},
            },
            model="openrouter:openai/gpt-5.6-luna",
        )


def test_pydantic_run_kwargs_receive_conversation_cache_defaults() -> None:
    adapter = PydanticAdapter(_openai_model())
    conversation = AdapterConversationContext(cache_key="runtime-cache-key")

    with adapter.conversation_scope(conversation):
        run_kwargs = adapter._run_kwargs({}, tool_specs=())

    assert run_kwargs["model_settings"] == {
        "openai_prompt_cache_key": "runtime-cache-key",
        "openai_previous_response_id": "auto",
    }

    with adapter.conversation_scope(conversation):
        explicit = adapter._run_kwargs(
            {"model_settings": {"openai_prompt_cache_key": "caller-key"}},
            tool_specs=(),
        )
    explicit_settings = cast(dict[str, Any], explicit["model_settings"])
    assert explicit_settings["openai_prompt_cache_key"] == "caller-key"

    anthropic = PydanticAdapter(_anthropic_model())
    with anthropic.conversation_scope(conversation):
        dynamic = anthropic._run_kwargs(
            {
                "model_settings": lambda _ctx: {
                    "anthropic_cache_messages": False,
                }
            },
            tool_specs=(),
        )
    settings_factory = dynamic["model_settings"]
    assert callable(settings_factory)
    assert cast(Callable[[Any], dict[str, Any]], settings_factory)(object()) == {
        "anthropic_cache_messages": False
    }

    async def unused_model(_messages: Any, _info: Any) -> Any:
        raise AssertionError("model must not run while settings are inspected")

    provider_neutral = PydanticAdapter(FunctionModel(unused_model))
    with provider_neutral.conversation_scope(conversation):
        neutral_kwargs = provider_neutral._run_kwargs({}, tool_specs=())
    assert "model_settings" not in neutral_kwargs


def test_adapter_run_settings_translate_openrouter_response_cache() -> None:
    profile = AgentProfile(
        settings={
            "response_cache": True,
            "response_cache_ttl": 900,
        }
    )
    pydantic = PydanticAdapter(_openrouter_model())
    with pydantic.profile_override(profile):
        run_kwargs = pydantic._run_kwargs({}, tool_specs=())

    assert run_kwargs["model_settings"] == {
        "extra_headers": {
            "X-OpenRouter-Cache": "true",
            "X-OpenRouter-Cache-TTL": "900",
        }
    }

    langchain = LangChainAdapter("openrouter:openai/gpt-5.6-luna")
    with langchain.profile_override(profile):
        settings = langchain._settings_for_run("openrouter:openai/gpt-5.6-luna")

    assert settings == {
        "model_kwargs": {
            "http_headers": {
                "X-OpenRouter-Cache": "true",
                "X-OpenRouter-Cache-TTL": "900",
            }
        }
    }

    unsupported = PydanticAdapter(_openai_model())
    with unsupported.profile_override(profile):
        with pytest.raises(
            ValueError,
            match="not supported by the selected model provider",
        ):
            unsupported._run_kwargs({}, tool_specs=())


def test_response_cache_settings_are_known_provider_neutral_dsl_settings() -> None:
    source = """> adapter: pydantic
> model: openrouter:google/gemini-3.1-flash-lite-preview
> settings:
    response_cache: true
    response_cache_ttl: 900
"""

    assert compute_diagnostics(source, "/tmp/response-cache.kedi") == []
    assert "response_cache" in hover_text_at(
        source,
        "/tmp/response-cache.kedi",
        "settings",
    )


def test_pydantic_native_run_captures_and_replays_complete_history() -> None:
    observed: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], _info: Any) -> ModelResponse:
        observed.append(list(messages))
        return ModelResponse(parts=[TextPart(f"response-{len(observed)}")])

    adapter = PydanticAdapter(FunctionModel(respond))
    state = ConversationState()
    with state.adapter_turn(adapter.shortname) as first:
        with adapter.conversation_scope(first):
            assert adapter.run_sync("first prompt").output == "response-1"
    first_messages = first.result_state["pydantic_messages"]

    with state.adapter_turn(adapter.shortname) as second:
        with adapter.conversation_scope(second):
            assert adapter.run_sync("second prompt").output == "response-2"
    second_messages = second.result_state["pydantic_messages"]

    assert len(observed) == 2
    assert len(observed[1]) > len(observed[0])
    assert second_messages[: len(first_messages)] == first_messages
    state.close()


def test_pydantic_compaction_marks_only_a_new_prefix_rewrite() -> None:
    old_compaction = ModelResponse(parts=[CompactionPart(content="old summary")])
    ordinary_response = ModelResponse(parts=[TextPart("answer")])
    new_compaction = ModelResponse(parts=[CompactionPart(content="new summary")])

    class Result:
        def __init__(self, new_messages: list[ModelMessage]) -> None:
            self._new_messages = new_messages

        def all_messages(self) -> list[ModelMessage]:
            return [old_compaction, *self._new_messages]

        def new_messages(self) -> list[ModelMessage]:
            return list(self._new_messages)

    replayed = AdapterConversationContext()
    PydanticAdapter._capture_conversation_messages(replayed, Result([ordinary_response]))
    assert replayed.cache_prefix_changed is False

    compacted = AdapterConversationContext()
    PydanticAdapter._capture_conversation_messages(compacted, Result([new_compaction]))
    assert compacted.cache_prefix_changed is True
    assert compacted.result_state["pydantic_messages"][-1]["parts"][0]["part_kind"] == (
        "compaction"
    )


def test_langchain_adapter_applies_prefix_defaults_inside_conversation_scope() -> None:
    adapter = LangChainAdapter("openai:gpt-5.4")
    conversation = AdapterConversationContext(cache_key="adapter-cache-key")

    with adapter.conversation_scope(conversation):
        settings = adapter._settings_for_run("openai:gpt-5.4")

    assert settings["model_kwargs"]["prompt_cache_key"] == "adapter-cache-key"
    assert settings["use_previous_response_id"] is True

    openrouter = LangChainAdapter("openrouter:openai/gpt-5.6-luna")
    with openrouter.conversation_scope(conversation):
        openrouter_settings = openrouter._settings_for_run("openrouter:openai/gpt-5.6-luna")
    assert openrouter_settings["session_id"] == "adapter-cache-key"
    assert openrouter_settings["model_kwargs"]["cache_control"] == {"type": "ephemeral"}


def test_langchain_replays_complete_message_history_including_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    adapter = LangChainAdapter("openai:gpt-5.4")
    payloads: list[list[Any]] = []

    class FakeAgent:
        async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            messages = list(payload["messages"])
            payloads.append(messages)
            if len(payloads) == 1:
                return {
                    "messages": [
                        *messages,
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "lookup",
                                    "args": {"query": "Kedi"},
                                    "id": "call-1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        ToolMessage(content="tool result", tool_call_id="call-1"),
                        AIMessage(content="first answer"),
                    ]
                }
            return {"messages": [*messages, AIMessage(content="second answer")]}

    monkeypatch.setattr(adapter, "_chat_model_for_run", lambda: object())
    monkeypatch.setattr(adapter, "_tools_for_run", _empty_async_tools)
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())

    state = ConversationState()
    with state.adapter_turn(adapter.shortname) as first:
        with adapter.conversation_scope(first):
            assert adapter.invoke_sync(prompt="first prompt") == "first answer"
    with state.adapter_turn(adapter.shortname) as second:
        with adapter.conversation_scope(second):
            assert adapter.invoke_sync(prompt="second prompt") == "second answer"

    assert isinstance(payloads[0][0], HumanMessage)
    assert [type(message) for message in payloads[1]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        HumanMessage,
    ]
    assert payloads[1][2].content == "tool result"
    state.close()


def test_langchain_history_ignores_incomplete_or_invalid_message_snapshots() -> None:
    from langchain_core.messages import HumanMessage

    adapter = LangChainAdapter("openai:gpt-5.4")
    conversation = AdapterConversationContext()

    with adapter.conversation_scope(conversation):
        messages = adapter._messages_for_prompt("fresh prompt")
        adapter._capture_conversation_messages("not a mapping")
        adapter._capture_conversation_messages({"messages": "not a sequence"})
        adapter._capture_conversation_messages({"messages": [object()]})

    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert conversation.result_state == {}


async def _empty_async_tools() -> list[Any]:
    return []


def test_history_lsp_hover_highlighting_and_unsupported_adapter_error() -> None:
    source = "> agent: codex\n> history: enabled\n"
    hover = hover_text_at(source, "/tmp/history.kedi", "enabled")
    assert "conversation continuity" in hover
    assert "default is disabled" in hover

    tokens = decode_semantic_tokens(compute_semantic_tokens_full(source, "/tmp/history.kedi").data)
    # Resolve through the public legend instead of coupling to hard-coded token positions.
    from kedi.lsp import features

    keyword = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    string = features.SEMANTIC_TOKEN_TYPES.index("string")
    assert any(line == 1 and token_type == keyword for line, _, _, token_type, _ in tokens)
    assert any(line == 1 and token_type == string for line, _, _, token_type, _ in tokens)

    diagnostics = compute_diagnostics(source, "/tmp/history.kedi")
    assert any(
        diagnostic.code == "K-ADAPTER-HISTORY"
        and diagnostic.severity is not None
        and "stateful conversation history" in diagnostic.message
        for diagnostic in diagnostics
    )

    profile_source = """
> profile: memory:
    > agent: codex
    > history: enabled

> use: memory
"""
    profile_hover = hover_text_at(profile_source, "/tmp/history-profile.kedi", "enabled")
    assert "stored in this profile" in profile_hover
    profile_diagnostics = compute_diagnostics(profile_source, "/tmp/history-profile.kedi")
    assert any(diagnostic.code == "K-ADAPTER-HISTORY" for diagnostic in profile_diagnostics)
