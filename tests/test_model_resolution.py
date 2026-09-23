from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.model_resolution import (
    pydantic_model_resolution,
    resolve_pydantic_model,
)
from kedi.agent_profile import AgentProfile


def test_model_resolver_applies_to_constructor_and_profile_override() -> None:
    seen: list[str] = []

    def resolve(model: str) -> TestModel:
        seen.append(model)
        return TestModel()

    with pydantic_model_resolution(resolve):
        adapter = PydanticAdapter(model="openai:example")
        with adapter.profile_override(AgentProfile(model="anthropic:another")):
            pass

    assert seen == ["openai-chat:example", "anthropic:another"]
    assert resolve_pydantic_model("openai-chat:example") == "openai-chat:example"


def test_nested_model_resolver_restores_scope_after_failure() -> None:
    outer = TestModel()
    inner = TestModel()
    with pydantic_model_resolution(lambda _: outer):
        with pytest.raises(ValueError, match="failed"):
            with pydantic_model_resolution(lambda _: inner):
                assert resolve_pydantic_model("test") is inner
                raise ValueError("failed")
        assert resolve_pydantic_model("test") is outer
    assert resolve_pydantic_model("test") == "test"
