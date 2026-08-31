from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .conversation import AdapterConversationContext

_RESPONSE_CACHE = "response_cache"
_RESPONSE_CACHE_TTL = "response_cache_ttl"
_OPENROUTER_CACHE_HEADER = "X-OpenRouter-Cache"
_OPENROUTER_CACHE_TTL_HEADER = "X-OpenRouter-Cache-TTL"

CacheBackend = Literal[
    "openai_responses",
    "openai_chat",
    "anthropic",
    "openrouter",
    "google",
    "moonshot",
    "alibaba",
    "zai",
    "bedrock",
    "azure_responses",
    "azure_chat",
    "deepseek",
    "xai",
    "unknown",
]
PrefixCacheMode = Literal["automatic", "explicit_breakpoints", "unsupported"]


@dataclass(frozen=True, slots=True)
class ProviderCachePolicy:
    """Provider cache capabilities used by adapter-specific translation."""

    provider: str
    backend: CacheBackend
    prefix_cache_mode: PrefixCacheMode
    stable_cache_key: bool = False
    continuation: bool = False
    exact_response_cache: bool = False
    explicit_resource_cache: bool = False
    provider_managed_ttl: bool = True


_UNKNOWN_POLICY = ProviderCachePolicy(
    provider="unknown",
    backend="unknown",
    prefix_cache_mode="unsupported",
)
_OPENAI_RESPONSES_POLICY = ProviderCachePolicy(
    provider="openai",
    backend="openai_responses",
    prefix_cache_mode="automatic",
    stable_cache_key=True,
    continuation=True,
)
_OPENAI_CHAT_POLICY = ProviderCachePolicy(
    provider="openai",
    backend="openai_chat",
    prefix_cache_mode="automatic",
    stable_cache_key=True,
)
_ANTHROPIC_POLICY = ProviderCachePolicy(
    provider="anthropic",
    backend="anthropic",
    prefix_cache_mode="explicit_breakpoints",
)
_OPENROUTER_POLICY = ProviderCachePolicy(
    provider="openrouter",
    backend="openrouter",
    prefix_cache_mode="explicit_breakpoints",
    stable_cache_key=True,
    exact_response_cache=True,
)
_GOOGLE_POLICY = ProviderCachePolicy(
    provider="google",
    backend="google",
    prefix_cache_mode="automatic",
    explicit_resource_cache=True,
)
_MOONSHOT_POLICY = ProviderCachePolicy(
    provider="moonshot",
    backend="moonshot",
    prefix_cache_mode="automatic",
    stable_cache_key=True,
)
_ALIBABA_POLICY = ProviderCachePolicy(
    provider="alibaba",
    backend="alibaba",
    prefix_cache_mode="automatic",
)
_ZAI_POLICY = ProviderCachePolicy(
    provider="zai",
    backend="zai",
    prefix_cache_mode="automatic",
)
_BEDROCK_POLICY = ProviderCachePolicy(
    provider="bedrock",
    backend="bedrock",
    prefix_cache_mode="explicit_breakpoints",
)
_AZURE_RESPONSES_POLICY = ProviderCachePolicy(
    provider="azure",
    backend="azure_responses",
    prefix_cache_mode="automatic",
    stable_cache_key=True,
    continuation=True,
)
_AZURE_CHAT_POLICY = ProviderCachePolicy(
    provider="azure",
    backend="azure_chat",
    prefix_cache_mode="automatic",
    stable_cache_key=True,
)
_DEEPSEEK_POLICY = ProviderCachePolicy(
    provider="deepseek",
    backend="deepseek",
    prefix_cache_mode="automatic",
)
_XAI_POLICY = ProviderCachePolicy(
    provider="xai",
    backend="xai",
    prefix_cache_mode="automatic",
)


def has_response_cache_settings(settings: Mapping[str, Any]) -> bool:
    return any(key in settings for key in (_RESPONSE_CACHE, _RESPONSE_CACHE_TTL))


def pydantic_provider_cache_policy(model: object) -> ProviderCachePolicy:
    """Return the cache behavior available through a Pydantic AI model."""

    provider = _normalize_provider(getattr(model, "system", None))
    model_kind = _pydantic_model_kind(model)
    if model_kind == "openrouter" or provider == "openrouter":
        return _OPENROUTER_POLICY
    if provider == "azure":
        if model_kind == "openai_responses":
            return _AZURE_RESPONSES_POLICY
        return _AZURE_CHAT_POLICY
    if provider == "moonshot":
        return _MOONSHOT_POLICY
    if provider == "alibaba":
        return _ALIBABA_POLICY
    if provider == "zai":
        return _ZAI_POLICY
    if provider == "bedrock":
        return _BEDROCK_POLICY
    if provider == "deepseek":
        return _DEEPSEEK_POLICY
    if provider == "xai":
        return _XAI_POLICY
    if model_kind == "openai_responses" or provider == "openai_responses":
        return _OPENAI_RESPONSES_POLICY
    if model_kind == "openai_chat" or provider == "openai":
        return _OPENAI_CHAT_POLICY
    if model_kind == "anthropic" or provider == "anthropic":
        return _ANTHROPIC_POLICY
    if model_kind == "google" or provider == "google":
        return _GOOGLE_POLICY
    return _UNKNOWN_POLICY


def _pydantic_model_kind(model: object) -> str | None:
    """Identify built-in Pydantic models without importing optional providers."""

    kinds = {
        "AnthropicModel": "anthropic",
        "GoogleModel": "google",
        "OpenAIChatModel": "openai_chat",
        "OpenAIResponsesModel": "openai_responses",
        "OpenRouterModel": "openrouter",
    }
    for model_type in type(model).__mro__:
        if model_type.__module__.startswith("pydantic_ai.models."):
            kind = kinds.get(model_type.__name__)
            if kind is not None:
                return kind
    return None


def langchain_provider_cache_policy(model: str | object) -> ProviderCachePolicy:
    """Return the cache behavior available through a LangChain chat model."""

    provider = _langchain_provider(model)
    if provider == "openrouter":
        return _OPENROUTER_POLICY
    if provider == "openai":
        return _OPENAI_RESPONSES_POLICY
    if provider == "anthropic":
        return _ANTHROPIC_POLICY
    if provider == "google":
        return _GOOGLE_POLICY
    if provider == "moonshot":
        return _MOONSHOT_POLICY
    if provider == "alibaba":
        return _ALIBABA_POLICY
    if provider == "zai":
        return _ZAI_POLICY
    if provider == "bedrock":
        return _BEDROCK_POLICY
    if provider == "azure":
        return _AZURE_CHAT_POLICY
    if provider == "deepseek":
        return _DEEPSEEK_POLICY
    if provider == "xai":
        return _XAI_POLICY
    return _UNKNOWN_POLICY


def pydantic_prefix_cache_settings(
    base: Mapping[str, Any],
    *,
    model: object,
    conversation: AdapterConversationContext,
) -> dict[str, Any]:
    """Apply provider cache defaults without overriding explicit model settings."""

    return pydantic_cache_settings(base, model=model, conversation=conversation)


def pydantic_cache_settings(
    base: Mapping[str, Any],
    *,
    model: object,
    conversation: AdapterConversationContext | None = None,
) -> dict[str, Any]:
    """Translate Kedi cache policy to Pydantic AI provider settings."""

    result = dict(base)
    model_settings = getattr(model, "settings", None)
    configured = model_settings if isinstance(model_settings, Mapping) else {}

    policy = pydantic_provider_cache_policy(model)
    response_cache, response_cache_ttl = _response_cache_policy(
        result,
        policy=policy,
    )

    if policy.backend in ("openai_responses", "azure_responses") and conversation is not None:
        _setdefault_unconfigured(
            result,
            configured,
            "openai_prompt_cache_key",
            conversation.cache_key,
        )
        if not _pydantic_response_storage_disabled(result, configured):
            _setdefault_unconfigured(result, configured, "openai_previous_response_id", "auto")
    elif policy.backend == "openrouter":
        if conversation is not None:
            for key in (
                "openrouter_cache_instructions",
                "openrouter_cache_messages",
                "openrouter_cache_tool_definitions",
            ):
                _setdefault_unconfigured(result, configured, key, True)
        if response_cache:
            _set_openrouter_response_cache_headers(
                result,
                configured,
                ttl=response_cache_ttl,
            )
    elif policy.backend in ("openai_chat", "moonshot", "azure_chat") and conversation is not None:
        _setdefault_unconfigured(
            result,
            configured,
            "openai_prompt_cache_key",
            conversation.cache_key,
        )
    elif policy.backend == "bedrock" and conversation is not None:
        for key in (
            "bedrock_cache_instructions",
            "bedrock_cache_messages",
            "bedrock_cache_tool_definitions",
        ):
            _setdefault_unconfigured(result, configured, key, True)
    elif (
        policy.backend == "anthropic"
        and conversation is not None
        and not any(
            key in result or key in configured
            for key in (
                "anthropic_cache",
                "anthropic_cache_instructions",
                "anthropic_cache_messages",
                "anthropic_cache_tool_definitions",
            )
        )
    ):
        result["anthropic_cache"] = True
    return result


def langchain_prefix_cache_settings(
    base: Mapping[str, Any],
    *,
    model: str | object,
    conversation: AdapterConversationContext,
) -> dict[str, Any]:
    """Apply provider cache defaults to LangChain model construction settings."""

    return langchain_cache_settings(base, model=model, conversation=conversation)


def langchain_cache_settings(
    base: Mapping[str, Any],
    *,
    model: str | object,
    conversation: AdapterConversationContext | None = None,
) -> dict[str, Any]:
    """Translate Kedi cache policy to LangChain provider settings."""

    result = dict(base)
    policy = langchain_provider_cache_policy(model)
    response_cache, response_cache_ttl = _response_cache_policy(
        result,
        policy=policy,
    )
    if policy.backend == "openrouter":
        if conversation is not None:
            configured_session = getattr(model, "session_id", None)
            result.setdefault("session_id", configured_session or conversation.cache_key)
        if response_cache:
            model_kwargs = _mapping_setting(result, "model_kwargs")
            headers = _mapping_value(model_kwargs.get("http_headers"))
            _set_cache_headers(headers, ttl=response_cache_ttl)
            model_kwargs["http_headers"] = headers
            result["model_kwargs"] = model_kwargs
        return result
    if not policy.stable_cache_key or conversation is None:
        return result
    configured_model_kwargs = result.get("model_kwargs")
    model_kwargs = (
        dict(configured_model_kwargs) if isinstance(configured_model_kwargs, Mapping) else {}
    )
    model_kwargs.setdefault("prompt_cache_key", conversation.cache_key)
    result["model_kwargs"] = model_kwargs
    if policy.continuation and not _langchain_response_storage_disabled(result, model):
        result.setdefault("use_previous_response_id", True)
    return result


def _langchain_provider(model: str | object) -> str:
    if isinstance(model, str):
        provider, separator, _ = model.partition(":")
        if not separator:
            return "unknown"
        return _normalize_provider(provider)
    module = type(model).__module__
    name = type(model).__name__
    if module.startswith("langchain_openrouter") and name == "ChatOpenRouter":
        return "openrouter"
    if module.startswith("langchain_openai") and name == "ChatOpenAI":
        return _openai_compatible_provider(model)
    if module.startswith("langchain_openai") and name == "AzureChatOpenAI":
        return "azure"
    if module.startswith("langchain_anthropic") and name == "ChatAnthropic":
        return "anthropic"
    if module.startswith(("langchain_google_genai", "langchain_google_vertexai")) and name in (
        "ChatGoogleGenerativeAI",
        "ChatVertexAI",
    ):
        return "google"
    if module.startswith("langchain_aws") and name in (
        "ChatBedrock",
        "ChatBedrockConverse",
        "ChatAnthropicBedrock",
    ):
        return "bedrock"
    if module.startswith("langchain_deepseek") and name == "ChatDeepSeek":
        return "deepseek"
    if module.startswith("langchain_xai") and name == "ChatXAI":
        return "xai"
    if name in ("MoonshotChat", "ChatMoonshot"):
        return "moonshot"
    if name in ("ChatTongyi", "ChatQwen"):
        return "alibaba"
    if name in ("ChatZhipuAI", "ChatZAI"):
        return "zai"
    return "unknown"


def _normalize_provider(provider: object) -> str:
    if not isinstance(provider, str):
        return "unknown"
    normalized = provider.replace("-", "_").lower()
    aliases = {
        "google_genai": "google",
        "google_vertexai": "google",
        "moonshotai": "moonshot",
        "moonshot_ai": "moonshot",
        "kimi": "moonshot",
        "dashscope": "alibaba",
        "qwen": "alibaba",
        "tongyi": "alibaba",
        "z_ai": "zai",
        "zhipu": "zai",
        "zhipuai": "zai",
        "amazon_bedrock": "bedrock",
        "bedrock_converse": "bedrock",
        "anthropic_bedrock": "bedrock",
        "azure_openai": "azure",
        "azure_ai": "azure",
        "grok": "xai",
    }
    return aliases.get(normalized, normalized)


def _openai_compatible_provider(model: object) -> str:
    base_url = getattr(model, "openai_api_base", None)
    if base_url is None:
        client = getattr(model, "root_client", None)
        base_url = getattr(client, "base_url", None)
    normalized_url = str(base_url or "").lower()
    if "moonshot" in normalized_url or "platform.kimi" in normalized_url:
        return "moonshot"
    if "dashscope" in normalized_url or "aliyuncs.com" in normalized_url:
        return "alibaba"
    if "deepseek" in normalized_url:
        return "deepseek"
    if "bigmodel.cn" in normalized_url or "api.z.ai" in normalized_url:
        return "zai"
    return "openai"


def _setdefault_unconfigured(
    result: dict[str, Any],
    configured: Mapping[str, Any],
    key: str,
    value: Any,
) -> None:
    if key not in configured:
        result.setdefault(key, value)


def _pydantic_response_storage_disabled(
    result: Mapping[str, Any],
    configured: Mapping[str, Any],
) -> bool:
    if "openai_store" in result:
        return result["openai_store"] is False
    return configured.get("openai_store") is False


def _langchain_response_storage_disabled(
    result: Mapping[str, Any],
    model: str | object,
) -> bool:
    if result.get("store") is False:
        return True
    result_model_kwargs = result.get("model_kwargs")
    if isinstance(result_model_kwargs, Mapping) and result_model_kwargs.get("store") is False:
        return True
    if isinstance(model, str):
        return False
    if getattr(model, "store", None) is False:
        return True
    configured_model_kwargs = getattr(model, "model_kwargs", None)
    return (
        isinstance(configured_model_kwargs, Mapping)
        and configured_model_kwargs.get("store") is False
    )


def _response_cache_policy(
    settings: dict[str, Any],
    *,
    policy: ProviderCachePolicy,
) -> tuple[bool, int | None]:
    enabled = settings.pop(_RESPONSE_CACHE, False)
    ttl = settings.pop(_RESPONSE_CACHE_TTL, None)
    if not isinstance(enabled, bool):
        raise ValueError("response_cache must be true or false")
    if ttl is not None and (not isinstance(ttl, int) or isinstance(ttl, bool)):
        raise ValueError("response_cache_ttl must be an integer")
    if ttl is not None and not 1 <= ttl <= 86_400:
        raise ValueError("response_cache_ttl must be between 1 and 86400")
    if ttl is not None and not enabled:
        raise ValueError("response_cache_ttl requires response_cache: true")
    if enabled and not policy.exact_response_cache:
        raise ValueError("response_cache is not supported by the selected model provider")
    return enabled, ttl


def _set_openrouter_response_cache_headers(
    result: dict[str, Any],
    configured: Mapping[str, Any],
    *,
    ttl: int | None,
) -> None:
    headers = _mapping_value(configured.get("extra_headers"))
    headers.update(_mapping_value(result.get("extra_headers")))
    _set_cache_headers(headers, ttl=ttl)
    result["extra_headers"] = headers


def _set_cache_headers(headers: dict[str, Any], *, ttl: int | None) -> None:
    cache_key = _case_insensitive_key(headers, _OPENROUTER_CACHE_HEADER)
    if cache_key is not None and str(headers[cache_key]).lower() != "true":
        raise ValueError("X-OpenRouter-Cache conflicts with response_cache: true")
    headers.setdefault(cache_key or _OPENROUTER_CACHE_HEADER, "true")
    if ttl is None:
        return
    ttl_key = _case_insensitive_key(headers, _OPENROUTER_CACHE_TTL_HEADER)
    if ttl_key is not None and str(headers[ttl_key]) != str(ttl):
        raise ValueError("X-OpenRouter-Cache-TTL conflicts with response_cache_ttl")
    headers.setdefault(ttl_key or _OPENROUTER_CACHE_TTL_HEADER, str(ttl))


def _mapping_setting(settings: dict[str, Any], key: str) -> dict[str, Any]:
    value = settings.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping when Kedi cache policy is enabled")
    return dict(value)


def _mapping_value(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return dict(cast("Mapping[str, Any]", value))


def _case_insensitive_key(values: Mapping[str, Any], expected: str) -> str | None:
    expected_lower = expected.lower()
    return next((key for key in values if key.lower() == expected_lower), None)
