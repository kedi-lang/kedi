"""Normalize model id strings between LiteLLM (DSPy) and Pydantic AI.

The same logical id is often written two ways (any vendor, not one provider):

- **LiteLLM / DSPy**: ``<provider>/<rest>`` — first slash separates the routing
  name from the rest (which may contain more slashes), e.g. ``groq/qwen/qwen3-32b``.

- **Pydantic AI** (and similar): ``<provider>:<rest>`` — first colon separates the
  same two parts, e.g. ``groq:qwen/qwen3-32b``. OpenAI chat-completions models
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
    return model[:ci] + "/" + model[ci + 1 :]


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
