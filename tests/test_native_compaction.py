from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAICompaction, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter.adapters.claude import ClaudeAdapter
from kedi.agent_adapter.adapters.codex import CodexAdapter
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.compat import (
    PydanticAICompatibilityError,
    effective_model,
)
from kedi.agent_adapter.compaction.native_settings import (
    codex_thread_params_with_native_compaction,
    langchain_settings_with_native_compaction,
    validate_claude_native_compaction,
    with_pydantic_native_compaction,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compaction import (
    CompactionSettings,
    CompactionSettingsError,
    normalize_compaction_settings,
)
from kedi.agent_profile.directives import eval_history_directive
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import parse_program
from kedi.lang.ast import HistoryDirectiveStmt, ProfileDefStmt
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


def test_flat_compaction_settings_parse_and_evaluate_in_all_profile_scopes() -> None:
    source = """
> history:
    enabled: true
    compaction_mode: native
    compaction_threshold: `50_000 * 2`

> profile: compact:
    > history:
        enabled: true
        compaction_mode: disabled

@local():
    > history:
        enabled: true
        compaction_mode: native
        compaction_threshold: 25_000
    = `None`
"""
    program = parse_program(source)
    history = program.ordered_toplevel[0]
    assert isinstance(history, HistoryDirectiveStmt)
    profile = program.ordered_toplevel[1]
    assert isinstance(profile, ProfileDefStmt)
    assert profile.history_members[0].fields[0].plain_value == "disabled"

    evaluated = eval_history_directive(KediRuntime(), history, {})
    assert evaluated.history is True
    assert evaluated.compaction == CompactionSettings(mode="native", threshold=100_000)

    threshold_only = parse_program(
        "> history:\n    enabled: true\n    compaction_threshold: 2_000\n"
    ).ordered_toplevel[0]
    assert isinstance(threshold_only, HistoryDirectiveStmt)
    assert eval_history_directive(KediRuntime(), threshold_only, {}).compaction == (
        CompactionSettings(mode="native", threshold=2_000)
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> history:\n    enabled: true\n    compaction_strategy: native\n",
            "history setting must be one of",
        ),
        (
            "> history:\n    enabled: true\n    compaction_mode: automatic\n",
            "history.compaction_mode: expected one of",
        ),
        (
            "> history:\n    enabled: true\n    compaction_mode: native\n    compaction_mode: disabled\n",
            "Duplicate history setting",
        ),
    ],
)
def test_flat_compaction_settings_reject_invalid_source(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_flat_compaction_settings_allow_comments_and_reject_duplicates() -> None:
    program = parse_program(
        "> history:\n    enabled: true\n    # provider policy\n    compaction_mode: native\n"
    )
    history = program.ordered_toplevel[0]
    assert isinstance(history, HistoryDirectiveStmt)
    assert history.fields[0].name == "compaction_mode"

    with pytest.raises(KediParseError, match="Duplicate history setting"):
        parse_program(
            "> history:\n"
            "    enabled: true\n"
            "    compaction_threshold: 10_000\n"
            "    compaction_threshold: 20_000\n"
        )


@pytest.mark.parametrize(
    "source",
    [
        "> settings:\n    > compaction:\n        mode: native\n",
        "> history:\n    enabled: true\n    > compaction:\n        mode: native\n",
    ],
)
def test_compaction_rejects_nested_directive_syntax(source: str) -> None:
    with pytest.raises(KediParseError):
        parse_program(source)


def test_expanded_history_requires_enabled_and_rejects_native_when_disabled() -> None:
    with pytest.raises(KediParseError, match="history block requires an 'enabled"):
        parse_program("> history:\n    compaction_mode: native\n")

    program = parse_program("> history:\n    enabled: false\n    compaction_mode: native\n")
    history = program.ordered_toplevel[0]
    assert isinstance(history, HistoryDirectiveStmt)
    with pytest.raises(KediExecutionError, match="requires history to be enabled"):
        eval_history_directive(KediRuntime(), history, {})

    invalid = parse_program(
        "> history:\n"
        "    enabled: true\n"
        "    compaction_mode: disabled\n"
        "    compaction_threshold: 10_000\n"
    ).ordered_toplevel[0]
    assert isinstance(invalid, HistoryDirectiveStmt)
    with pytest.raises(CompactionSettingsError, match="cannot be set"):
        eval_history_directive(KediRuntime(), invalid, {})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> history:\n    unknown: true\n    enabled: true\n",
            "history setting must be one of",
        ),
        (
            "> history:\n    enabled: yes\n",
            "must be 'true' or 'false'",
        ),
        (
            "> settings:\n    temperature: 0.2\n    temperature: 0.3\n",
            "Duplicate settings field",
        ),
    ],
)
def test_history_and_settings_blocks_reject_invalid_fields(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_history_compaction_profile_merge_and_field_hover() -> None:
    compaction = CompactionSettings(threshold=10_000)
    merged = AgentProfile(history=True).merge(AgentProfile(compaction=compaction))
    assert merged.compaction is compaction

    source = """> profile: compact:
    > history:
        # keep the provider prefix stable
        enabled: true
        compaction_mode: native
"""
    assert "provider-native conversation compaction" in hover_text_at(
        source,
        "compaction.kedi",
        "compaction_mode",
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


def test_pydantic_native_compaction_sees_through_budget_wrappers() -> None:
    from kedi.agent_adapter.adapters.pydantic.budget import BudgetedModel

    wrapped = BudgetedModel(BudgetedModel(_openai_responses_model()))
    capabilities = with_pydantic_native_compaction(
        None, model=wrapped, settings=CompactionSettings(threshold=18_000)
    )
    assert [type(item) for item in capabilities] == [OpenAICompaction]
    assert capabilities[0].token_threshold == 18_000


def test_pydantic_adapter_applies_profile_compaction_and_rejects_chat_model() -> None:
    adapter = PydanticAdapter(_openai_responses_model())
    with adapter.profile_override(
        AgentProfile(history=True, compaction=CompactionSettings(threshold=42_000))
    ):
        run_kwargs = adapter._run_kwargs({}, tool_specs=())
    capability = next(
        item for item in run_kwargs["capabilities"] if isinstance(item, OpenAICompaction)
    )
    assert capability.token_threshold == 42_000

    chat_model = OpenAIChatModel("gpt-5.4", provider=OpenAIProvider(api_key="test"))
    chat_adapter = PydanticAdapter(chat_model)
    with (
        chat_adapter.profile_override(AgentProfile(history=True, compaction=CompactionSettings())),
        pytest.raises(CompactionSettingsError, match="OpenAIResponsesModel"),
    ):
        chat_adapter._run_kwargs({}, tool_specs=())


def test_pydantic_disabled_compaction_does_not_remove_caller_capability() -> None:
    caller = OpenAICompaction(token_threshold=12_345)
    adapter = PydanticAdapter(_openai_responses_model())
    with adapter.profile_override(
        AgentProfile(history=True, compaction=CompactionSettings(mode="disabled"))
    ):
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


def test_codex_native_compaction_merges_thread_config_without_mutating_caller() -> None:
    base = {"config": {"features.web_search": True}, "ephemeral": False}
    configured = codex_thread_params_with_native_compaction(
        base,
        settings=CompactionSettings(threshold=64_000),
    )

    assert configured == {
        "config": {
            "features.web_search": True,
            "model_auto_compact_token_limit": 64_000,
        },
        "ephemeral": False,
    }
    assert base == {"config": {"features.web_search": True}, "ephemeral": False}
    assert (
        codex_thread_params_with_native_compaction(
            base,
            settings=CompactionSettings(),
        )
        == base
    )
    assert (
        codex_thread_params_with_native_compaction(
            base,
            settings=CompactionSettings(mode="disabled"),
        )
        == base
    )
    with pytest.raises(CompactionSettingsError, match="config must be a mapping"):
        codex_thread_params_with_native_compaction(
            {"config": "invalid"},
            settings=CompactionSettings(threshold=10),
        )


def test_claude_native_compaction_uses_sdk_policy_without_exact_threshold() -> None:
    validate_claude_native_compaction(CompactionSettings())
    validate_claude_native_compaction(CompactionSettings(mode="disabled"))
    with pytest.raises(CompactionSettingsError, match="does not expose an exact"):
        validate_claude_native_compaction(CompactionSettings(threshold=64_000))


def test_langchain_adapter_applies_profile_compaction_and_rejects_provider() -> None:
    adapter = LangChainAdapter("openai:gpt-5.4")
    with adapter.profile_override(
        AgentProfile(history=True, compaction=CompactionSettings(threshold=80_000))
    ):
        settings = adapter._settings_for_run(adapter._model_for_run())
    assert settings["context_management"] == [{"type": "compaction", "compact_threshold": 80_000}]

    unsupported = LangChainAdapter("google:gemini-3-flash-preview")
    with (
        unsupported.profile_override(AgentProfile(history=True, compaction=CompactionSettings())),
        pytest.raises(CompactionSettingsError, match="OpenAI or Anthropic"),
    ):
        unsupported._settings_for_run(unsupported._model_for_run())


def test_langchain_adapter_copies_object_model_for_native_compaction() -> None:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-5.4", api_key="test")
    adapter = LangChainAdapter(model)
    with adapter.profile_override(
        AgentProfile(history=True, compaction=CompactionSettings(threshold=32_000))
    ):
        selected = adapter._chat_model_for_run()
    assert selected is not model
    assert selected.context_management == [{"type": "compaction", "compact_threshold": 32_000}]
    assert model.context_management is None

    fake_type = type("ChatOpenAI", (), {"__module__": "langchain_openai.fake"})
    unsupported = LangChainAdapter(fake_type())
    with (
        unsupported.profile_override(AgentProfile(history=True, compaction=CompactionSettings())),
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
    source = """> history:
    enabled: true
    compaction_mode: native
    compaction_threshold: `100_000`
"""
    assert compute_diagnostics(source, "compaction.kedi") == []
    assert "provider-native conversation compaction" in hover_text_at(
        source, "compaction.kedi", "compaction_mode"
    )
    assert "input-token threshold" in hover_text_at(
        source, "compaction.kedi", "compaction_threshold"
    )
    assert 'Literal["native", "disabled"]' in hover_text_at(
        source, "compaction.kedi", "compaction_mode"
    )
    assert "conversation continuity through stateful history" in hover_text_at(
        source, "compaction.kedi", "history"
    )


def test_compaction_adapter_capabilities_advertise_native_support() -> None:
    assert "compaction" not in PydanticAdapter.capabilities.settings
    assert "compaction" not in LangChainAdapter.capabilities.settings
    assert PydanticAdapter.capabilities.supports("native_compaction")
    assert LangChainAdapter.capabilities.supports("native_compaction")
    assert CodexAdapter.capabilities.supports("native_compaction")
    assert ClaudeAdapter.capabilities.supports("native_compaction")

    KediRuntime._assert_profile_override_supported(
        AgentProfile(history=True, compaction=CompactionSettings()),
        object.__new__(PydanticAdapter),
    )

    with pytest.raises(CompactionSettingsError, match="requires history to be enabled"):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(history=False, compaction=CompactionSettings()),
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

    profile = AgentProfile(history=True, compaction=CompactionSettings())
    adapter = object.__new__(DSPyAdapter)
    with pytest.raises(CompactionSettingsError, match="adapter 'dspy'"):
        KediRuntime._assert_profile_override_supported(profile, adapter)

    source = """> adapter: dspy
> history:
    enabled: true
    compaction_mode: native
"""
    diagnostics = compute_diagnostics(source, "compaction.kedi")
    assert [item.code for item in diagnostics] == [
        "K-ADAPTER-HISTORY",
        "K-ADAPTER-COMPACTION",
    ]

    profile_source = """> profile: unsupported:
    > adapter: dspy
    > history:
        enabled: true
        compaction_threshold: 1000
"""
    profile_diagnostics = compute_diagnostics(profile_source, "compaction.kedi")
    assert [item.code for item in profile_diagnostics] == [
        "K-ADAPTER-HISTORY",
        "K-ADAPTER-COMPACTION",
    ]


def test_disabled_compaction_is_accepted_by_unsupported_adapter() -> None:
    from kedi.agent_adapter.adapters.dspy import DSPyAdapter

    profile = AgentProfile(history=True, compaction=CompactionSettings(mode="disabled"))
    adapter = object.__new__(DSPyAdapter)
    KediRuntime._assert_profile_override_supported(profile, adapter)

    source = """> adapter: dspy
> history:
    enabled: true
    compaction_mode: disabled
"""
    assert [item.code for item in compute_diagnostics(source, "compaction.kedi")] == [
        "K-ADAPTER-HISTORY"
    ]

    dynamic_source = """> adapter: dspy
> history:
    enabled: true
    compaction_mode: `selected_mode`
"""
    assert [item.code for item in compute_diagnostics(dynamic_source, "compaction.kedi")] == [
        "K-ADAPTER-HISTORY",
    ]


def test_native_compaction_rejects_adapter_without_capability_metadata() -> None:
    class ProfileAdapter:
        shortname = "custom"

        def profile_override(self, _profile: AgentProfile) -> Any:
            raise AssertionError("profile override must not be entered")

    with pytest.raises(CompactionSettingsError, match="adapter 'custom'"):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(history=True, compaction=CompactionSettings()),
            ProfileAdapter(),
        )


def test_effective_pydantic_model_reports_private_api_layout_error() -> None:
    with pytest.raises(PydanticAICompatibilityError, match="unsupported Agent tool layout"):
        effective_model(object())  # type: ignore[arg-type]
