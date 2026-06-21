"""Tests for provider model id normalization (LiteLLM vs Pydantic-style)."""

from kedi.model_normalization import (
    normalize_for_langchain,
    normalize_for_litellm,
    normalize_for_pydantic_ai,
)


def test_colon_to_slash_for_litellm() -> None:
    assert normalize_for_litellm("acme:google/gemini-3-flash-preview") == (
        "acme/google/gemini-3-flash-preview"
    )
    assert normalize_for_litellm("groq:qwen/qwen3-32b") == "groq/qwen/qwen3-32b"


def test_slash_form_unchanged_for_litellm() -> None:
    assert normalize_for_litellm("acme/foo/bar-baz") == "acme/foo/bar-baz"
    assert normalize_for_litellm("groq/llama-3-8b") == "groq/llama-3-8b"
    assert normalize_for_litellm("acme/foo:bar-baz") == "acme/foo:bar-baz"


def test_slash_to_colon_for_pydantic() -> None:
    assert normalize_for_pydantic_ai("acme/foo/bar-baz") == "acme:foo/bar-baz"
    assert normalize_for_pydantic_ai("groq/qwen/qwen3-32b") == "groq:qwen/qwen3-32b"
    assert normalize_for_pydantic_ai("openai/gpt-4o") == "openai-chat:gpt-4o"


def test_colon_form_unchanged_for_pydantic() -> None:
    assert normalize_for_pydantic_ai("acme:foo/bar-baz") == "acme:foo/bar-baz"
    assert normalize_for_pydantic_ai("groq:qwen/qwen3-32b") == "groq:qwen/qwen3-32b"
    assert normalize_for_pydantic_ai("openai-chat:gpt-4o") == "openai-chat:gpt-4o"
    assert normalize_for_pydantic_ai("openai-responses:gpt-4o") == ("openai-responses:gpt-4o")


def test_round_trip_multi_segment_provider() -> None:
    lit = "acme/foo/bar-baz"
    pyd = normalize_for_pydantic_ai(lit)
    assert pyd == "acme:foo/bar-baz"
    assert normalize_for_litellm(pyd) == lit


def test_urls_untouched() -> None:
    u = "https://example.com/v1"
    assert normalize_for_litellm(u) == u
    assert normalize_for_pydantic_ai(u) == u


def test_no_slash_no_change_for_pydantic() -> None:
    assert normalize_for_pydantic_ai("gpt-4o") == "gpt-4o"
    assert normalize_for_pydantic_ai("openai:gpt-4o") == "openai-chat:gpt-4o"


def test_slash_to_colon_for_langchain() -> None:
    assert normalize_for_langchain("openrouter/google/gemini-3-flash-preview") == (
        "openrouter:google/gemini-3-flash-preview"
    )
    assert normalize_for_langchain("groq/qwen/qwen3-32b") == "groq:qwen/qwen3-32b"
    assert normalize_for_langchain("openai-chat:gpt-4o") == "openai:gpt-4o"
    assert normalize_for_langchain("openai:gpt-4o") == "openai:gpt-4o"
    assert normalize_for_langchain("https://example.com/v1") == "https://example.com/v1"
