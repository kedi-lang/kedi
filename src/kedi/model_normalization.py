"""Normalize model id strings between LiteLLM (DSPy) and Pydantic AI.

The same logical id is often written two ways (any vendor, not one provider):

- **LiteLLM / DSPy**: ``<provider>/<rest>`` — first slash separates the routing
  name from the rest, e.g. ``openai/gpt-6-luna``. Additional slashes in the
  model name are preserved, as in ``openrouter/openai/gpt-6-luna``.

- **Pydantic AI** (and similar): ``<provider>:<rest>`` — first colon separates the
  same two parts, e.g. ``openrouter:openai/gpt-6-luna``. OpenAI chat-completions models
  use Pydantic AI's explicit ``openai-chat:`` provider to avoid the deprecated
  ambiguous ``openai:`` prefix.

Kedi normalizes at each boundary so either spelling works in CLI and env.

Strings without a first ``provider/rest`` or ``provider:rest`` split (no slash, or
already in one canonical form) are left unchanged. URLs containing ``://`` are never
modified.
"""

from __future__ import annotations


def normalize_for_litellm(model: str) -> str:
    """Use the first slash between provider and rest (LiteLLM-style)."""
    if not model or ":" not in model or "://" in model:
        return model
    ci = model.find(":")
    si = model.find("/")
    if si != -1 and si < ci:
        return model
    provider = model[:ci]
    rest = model[ci + 1 :]
    # LiteLLM's canonical OpenAI provider is "openai"; map back the Pydantic-AI-specific
    # "openai-chat"/"openai-responses" so a pydantic→litellm round-trip stays "openai/...".
    if provider in ("openai-chat", "openai-responses"):
        provider = "openai"
    return provider + "/" + rest


def normalize_for_pydantic_ai(model: str) -> str:
    """Use the first colon between provider and rest (Pydantic-style)."""
    if not model or "://" in model:
        return model
    if model.startswith("openai-chat:") or model.startswith("openai-responses:"):
        return model
    if model.startswith("openai:"):
        return "openai-chat:" + model[len("openai:") :]
    si = model.find("/")
    ci = model.find(":")
    if si == -1:
        return model
    if ci != -1 and (si == -1 or ci < si):
        return model
    provider = model[:si]
    rest = model[si + 1 :]
    if provider == "openai":
        provider = "openai-chat"
    return provider + ":" + rest


def normalize_for_langchain(model: str) -> str:
    """Use LangChain's provider-prefixed chat model id form."""
    if not model or "://" in model:
        return model
    if model.startswith("openai-chat:"):
        return "openai:" + model[len("openai-chat:") :]
    si = model.find("/")
    ci = model.find(":")
    if si == -1 or (ci != -1 and ci < si):
        return model
    return model[:si] + ":" + model[si + 1 :]
