from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAICompaction, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter.adapters._pydantic_compat import (
    PydanticAICompatibilityError,
    effective_model,
)
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.native_compaction import (
    langchain_settings_with_native_compaction,
    with_pydantic_native_compaction,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compaction import (
    CompactionSettings,
    CompactionSettingsError,
    normalize_compaction_settings,
)
from kedi.agent_profile.model import eval_settings_directive
from kedi.core import KediRuntime
from kedi.lang import parse_program
from kedi.lang.ast import ProfileDefStmt, SettingsDirectiveStmt
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features import compute_diagnostics
from tests.lsp_helpers import hover_text_at


def _openai_responses_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        "gpt-5.4",
        provider=OpenAIProvider(api_key="test"),
    )


def _anthropic_model() -> AnthropicModel:
    return AnthropicModel(
        "claude-opus-4-6",
        provider=AnthropicProvider(api_key="test"),
    )


def test_compaction_settings_validate_mode_threshold_and_disabled_policy() -> None:
    assert normalize_compaction_settings(None) is None
    assert normalize_compaction_settings({"mode": "native", "threshold": 100_000}) == (
        CompactionSettings(mode="native", threshold=100_000)
    )
    assert normalize_compaction_settings({"mode": "native"}) == CompactionSettings()
    configured = CompactionSettings(threshold=10)
    assert normalize_compaction_settings(configured) is configured
    assert normalize_compaction_settings({"mode": "disabled"}) == CompactionSettings(
        mode="disabled"
    )

    with pytest.raises(CompactionSettingsError, match="positive integer"):
        normalize_compaction_settings({"mode": "native", "threshold": True})
    with pytest.raises(CompactionSettingsError, match="positive integer"):
        normalize_compaction_settings({"mode": "native", "threshold": -1})
    with pytest.raises(CompactionSettingsError, match="must be a string"):
        normalize_compaction_settings({"mode": 1})
    with pytest.raises(CompactionSettingsError, match="must be a CompactionSettings"):
        normalize_compaction_settings("native")  # type: ignore[arg-type]
    with pytest.raises(CompactionSettingsError, match="mode must be"):
        CompactionSettings(mode="automatic")  # type: ignore[arg-type]
    with pytest.raises(CompactionSettingsError, match="cannot be set"):
        normalize_compaction_settings({"mode": "disabled", "threshold": 10})
    with pytest.raises(CompactionSettingsError, match="unknown compaction setting"):
        normalize_compaction_settings({"strategy": "native"})


def test_nested_compaction_settings_parse_and_evaluate_in_all_profile_scopes() -> None:
    source = """
> settings:
    > compaction:
        mode: native
        threshold: `50_000 * 2`

> profile: compact:
    > settings:
        > compaction:
            mode: disabled

@local():
    > settings:
        > compaction:
            mode: native
            threshold: 25_000
    = `None`
"""
    program = parse_program(source)
    settings = program.ordered_toplevel[0]
    assert isinstance(settings, SettingsDirectiveStmt)
    profile = program.ordered_toplevel[1]
    assert isinstance(profile, ProfileDefStmt)
    assert profile.settings_members[0].fields[0].fields[0].plain_value == "disabled"

    evaluated = eval_settings_directive(KediRuntime(), settings, {})
    assert evaluated.settings == {"compaction": {"mode": "native", "threshold": 100_000}}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> settings:\n    > compaction:\n        strategy: native\n",
            "compaction setting must be one of",
        ),
        (
            "> settings:\n    > compaction:\n        mode: automatic\n",
            "compaction mode must be",
        ),
        (
            "> settings:\n    > compaction:\n        mode: native\n        mode: disabled\n",
            "Duplicate compaction setting",
        ),
    ],
)
def test_nested_compaction_settings_reject_invalid_source(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_nested_compaction_settings_allow_comments_and_reject_duplicate_block() -> None:
    program = parse_program(
        "> settings:\n    > compaction:\n        # provider policy\n        mode: native\n"
    )
    settings = program.ordered_toplevel[0]
    assert isinstance(settings, SettingsDirectiveStmt)
    assert settings.fields[0].fields[0].name == "mode"

    with pytest.raises(KediParseError, match="Duplicate settings field"):
        parse_program(
            "> settings:\n"
            "    > compaction:\n"
            "        mode: native\n"
            "    > compaction:\n"
            "        mode: native\n"
        )


def test_pydantic_native_compaction_maps_openai_and_anthropic_capabilities() -> None:
    marker = object()
    openai_capabilities = with_pydantic_native_compaction(
        [marker, OpenAICompaction(token_threshold=1)],
        model=_openai_responses_model(),
        settings=CompactionSettings(threshold=100_000),
    )
    assert marker in openai_capabilities
    openai = next(item for item in openai_capabilities if isinstance(item, OpenAICompaction))
    assert openai.token_threshold == 100_000

    anthropic_capabilities = with_pydantic_native_compaction(
        openai_capabilities,
        model=_anthropic_model(),
        settings=CompactionSettings(threshold=75_000),
    )
    assert not any(isinstance(item, OpenAICompaction) for item in anthropic_capabilities)
    anthropic = next(
        item for item in anthropic_capabilities if isinstance(item, AnthropicCompaction)
    )
    assert anthropic.token_threshold == 75_000


def test_pydantic_adapter_applies_profile_compaction_and_rejects_chat_model() -> None:
    adapter = PydanticAdapter(_openai_responses_model())
    with adapter.profile_override(
        AgentProfile(settings={"compaction": {"mode": "native", "threshold": 42_000}})
    ):
        run_kwargs = adapter._run_kwargs({}, tool_specs=())
    capability = next(
        item for item in run_kwargs["capabilities"] if isinstance(item, OpenAICompaction)
    )
    assert capability.token_threshold == 42_000

    chat_model = OpenAIChatModel("gpt-5.4", provider=OpenAIProvider(api_key="test"))
    chat_adapter = PydanticAdapter(chat_model)
    with (
        chat_adapter.profile_override(AgentProfile(settings={"compaction": {"mode": "native"}})),
        pytest.raises(CompactionSettingsError, match="OpenAIResponsesModel"),
    ):
        chat_adapter._run_kwargs({}, tool_specs=())


def test_pydantic_disabled_compaction_does_not_remove_caller_capability() -> None:
    caller = OpenAICompaction(token_threshold=12_345)
    adapter = PydanticAdapter(_openai_responses_model())
    with adapter.profile_override(AgentProfile(settings={"compaction": {"mode": "disabled"}})):
        run_kwargs = adapter._run_kwargs({"capabilities": [caller]}, tool_specs=())
    assert caller in run_kwargs["capabilities"]


def test_pydantic_compaction_helper_validates_capability_container() -> None:
    with pytest.raises(TypeError, match="list or tuple"):
        with_pydantic_native_compaction(
            object(),
            model=_openai_responses_model(),
            settings=CompactionSettings(),
        )
    assert with_pydantic_native_compaction(
        ("caller",),
        model=object(),
        settings=CompactionSettings(mode="disabled"),
    ) == ["caller"]


def test_langchain_native_compaction_preserves_existing_provider_settings() -> None:
    openai = langchain_settings_with_native_compaction(
        {"context_management": [{"type": "other"}]},
        model="openai:gpt-5.4",
        settings=CompactionSettings(threshold=100_000),
    )
    assert openai["context_management"] == [
        {"type": "other"},
        {"type": "compaction", "compact_threshold": 100_000},
    ]

    anthropic = langchain_settings_with_native_compaction(
        {
            "betas": ["existing-beta"],
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        },
        model="anthropic:claude-opus-4-6",
        settings=CompactionSettings(),
    )
    assert anthropic["betas"] == ["existing-beta", "compact-2026-01-12"]
    assert anthropic["context_management"]["edits"] == [
        {"type": "clear_tool_uses_20250919"},
        {
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 150_000},
        },
    ]

    defaults = langchain_settings_with_native_compaction(
        {},
        model="anthropic:claude-opus-4-6",
        settings=CompactionSettings(threshold=50_000),
    )
    assert defaults["betas"] == ["compact-2026-01-12"]
    assert defaults["context_management"]["edits"][0]["trigger"]["value"] == 50_000


@pytest.mark.parametrize(
    ("model", "base", "message"),
    [
        (
            "openai:gpt-5.4",
            {"context_management": {}},
            "OpenAI context_management must be a list",
        ),
        (
            "anthropic:claude-opus-4-6",
            {"context_management": []},
            "Anthropic context_management must be a mapping",
        ),
        (
            "anthropic:claude-opus-4-6",
            {"context_management": {"edits": {}}},
            "edits must be a list",
        ),
        (
            "anthropic:claude-opus-4-6",
            {"betas": "compact"},
            "betas must be a sequence",
        ),
        (
            "anthropic:claude-opus-4-6",
            {"betas": [1]},
            "betas must be a sequence",
        ),
    ],
)
def test_langchain_native_compaction_rejects_invalid_provider_settings(
    model: str,
    base: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(CompactionSettingsError, match=message):
        langchain_settings_with_native_compaction(
            base,
            model=model,
            settings=CompactionSettings(),
        )


def test_langchain_native_compaction_replaces_owned_entries_without_duplicate_beta() -> None:
    settings = langchain_settings_with_native_compaction(
        {
            "betas": ["compact-2026-01-12"],
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 10},
                    }
                ]
            },
        },
        model="anthropic:claude-opus-4-6",
        settings=CompactionSettings(threshold=20),
    )
    assert settings["betas"] == ["compact-2026-01-12"]
    assert settings["context_management"]["edits"][0]["trigger"]["value"] == 20


def test_langchain_adapter_applies_profile_compaction_and_rejects_provider() -> None:
    adapter = LangChainAdapter("openai:gpt-5.4")
    with adapter.profile_override(
        AgentProfile(settings={"compaction": {"mode": "native", "threshold": 80_000}})
    ):
        settings = adapter._settings_for_run(adapter._model_for_run())
    assert settings["context_management"] == [{"type": "compaction", "compact_threshold": 80_000}]

    unsupported = LangChainAdapter("google:gemini-3-flash-preview")
    with (
        unsupported.profile_override(AgentProfile(settings={"compaction": {"mode": "native"}})),
        pytest.raises(CompactionSettingsError, match="OpenAI or Anthropic"),
    ):
        unsupported._settings_for_run(unsupported._model_for_run())


def test_langchain_adapter_copies_object_model_for_native_compaction() -> None:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-5.4", api_key="test")
    adapter = LangChainAdapter(model)
    with adapter.profile_override(
        AgentProfile(settings={"compaction": {"mode": "native", "threshold": 32_000}})
    ):
        selected = adapter._chat_model_for_run()
    assert selected is not model
    assert selected.context_management == [{"type": "compaction", "compact_threshold": 32_000}]
    assert model.context_management is None

    fake_type = type("ChatOpenAI", (), {"__module__": "langchain_openai.fake"})
    unsupported = LangChainAdapter(fake_type())
    with (
        unsupported.profile_override(AgentProfile(settings={"compaction": {"mode": "native"}})),
        pytest.raises(CompactionSettingsError, match="supports model_copy"),
    ):
        unsupported._chat_model_for_run()


def test_langchain_provider_detection_accepts_anthropic_model_object() -> None:
    fake_type = type("ChatAnthropic", (), {"__module__": "langchain_anthropic.fake"})
    settings = langchain_settings_with_native_compaction(
        {},
        model=fake_type(),
        settings=CompactionSettings(),
    )
    assert settings["betas"] == ["compact-2026-01-12"]

    with pytest.raises(CompactionSettingsError, match="OpenAI or Anthropic provider"):
        langchain_settings_with_native_compaction(
            {},
            model=object(),
            settings=CompactionSettings(),
        )


def test_compaction_settings_lsp_hover_and_diagnostics() -> None:
    source = """> settings:
    > compaction:
        mode: native
        threshold: `100_000`
"""
    assert compute_diagnostics(source, "compaction.kedi") == []
    assert "provider-native context compaction" in hover_text_at(
        source, "compaction.kedi", "compaction"
    )
    assert "input-token threshold" in hover_text_at(source, "compaction.kedi", "threshold")
    assert 'Literal["native", "disabled"]' in hover_text_at(source, "compaction.kedi", "mode")
    assert "Configured fields: `compaction`" in hover_text_at(source, "compaction.kedi", "settings")


def test_compaction_adapter_capabilities_advertise_reserved_setting() -> None:
    assert "compaction" in PydanticAdapter.capabilities.settings
    assert "compaction" in LangChainAdapter.capabilities.settings
    assert PydanticAdapter.capabilities.supports("native_compaction")
    assert LangChainAdapter.capabilities.supports("native_compaction")

    KediRuntime._assert_profile_override_supported(
        AgentProfile(settings={"compaction": {"mode": "native"}}),
        object.__new__(PydanticAdapter),
    )


def test_langchain_disabled_compaction_leaves_settings_unchanged() -> None:
    base: dict[str, Any] = {"temperature": 0.2}
    assert (
        langchain_settings_with_native_compaction(
            base,
            model="google:gemini-3-flash-preview",
            settings=CompactionSettings(mode="disabled"),
        )
        == base
    )


def test_unsupported_adapter_rejects_native_compaction_at_runtime_and_in_lsp() -> None:
    from kedi.agent_adapter.adapters.dspy import DSPyAdapter

    profile = AgentProfile(settings={"compaction": {"mode": "native"}})
    adapter = object.__new__(DSPyAdapter)
    with pytest.raises(CompactionSettingsError, match="adapter 'dspy'"):
        KediRuntime._assert_profile_override_supported(profile, adapter)

    source = """> adapter: dspy
> settings:
    > compaction:
        mode: native
"""
    diagnostics = compute_diagnostics(source, "compaction.kedi")
    assert [(item.code, item.severity) for item in diagnostics] == [("K-ADAPTER-COMPACTION", 1)]

    profile_source = """> profile: unsupported:
    > adapter: dspy
    > settings:
        > compaction:
            threshold: 1000
"""
    profile_diagnostics = compute_diagnostics(profile_source, "compaction.kedi")
    assert [item.code for item in profile_diagnostics] == ["K-ADAPTER-COMPACTION"]


def test_disabled_compaction_is_accepted_by_unsupported_adapter() -> None:
    from kedi.agent_adapter.adapters.dspy import DSPyAdapter

    profile = AgentProfile(settings={"compaction": {"mode": "disabled"}})
    adapter = object.__new__(DSPyAdapter)
    KediRuntime._assert_profile_override_supported(profile, adapter)

    source = """> adapter: dspy
> settings:
    > compaction:
        mode: disabled
"""
    assert compute_diagnostics(source, "compaction.kedi") == []

    dynamic_source = """> adapter: dspy
> settings:
    > compaction:
        mode: `selected_mode`
"""
    assert compute_diagnostics(dynamic_source, "compaction.kedi") == []


def test_native_compaction_rejects_adapter_without_capability_metadata() -> None:
    class ProfileAdapter:
        shortname = "custom"

        def profile_override(self, _profile: AgentProfile) -> Any:
            raise AssertionError("profile override must not be entered")

    with pytest.raises(CompactionSettingsError, match="adapter 'custom'"):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(settings={"compaction": {"mode": "native"}}),
            ProfileAdapter(),
        )


def test_effective_pydantic_model_reports_private_api_layout_error() -> None:
    with pytest.raises(PydanticAICompatibilityError, match="unsupported Agent tool layout"):
        effective_model(object())  # type: ignore[arg-type]
