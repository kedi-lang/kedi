from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.api.configuration import _lazy_adapter
from kedi.app import run_program_cli


def respond(messages, info):
    if info.output_tools:
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"value": 37})])
    return ModelResponse([TextPart("ready")])


@pytest.mark.asyncio
async def test_unused_default_provider_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    adapter = _lazy_adapter("pydantic", "groq:qwen/qwen3-32b", {}).resolve()
    model = FunctionModel(respond)
    monkeypatch.setattr(adapter, "_resolve_model", lambda name: model)
    with adapter.profile_override(AgentProfile(model="test-override")):
        assert await adapter.invoke(prompt="Reply ready") == "ready"
    assert adapter.default_model() == "groq:qwen/qwen3-32b"
    with pytest.raises(Exception, match="GROQ_API_KEY"):
        await adapter.invoke(prompt="Use the default now")


def test_explicit_eager_model_validation_is_preserved(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception, match="GROQ_API_KEY"):
        _lazy_adapter("pydantic", "groq:qwen/qwen3-32b", {"defer_model_check": False}).resolve()


def test_direct_adapter_keeps_native_eager_validation(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception, match="GROQ_API_KEY"):
        PydanticAdapter("groq:qwen/qwen3-32b")


def test_cli_program_model_overrides_unused_default_without_credentials(monkeypatch, tmp_path):
    import kedi.agent_adapter.adapters.pydantic.adapter as models

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(models, "codex_responses_model", lambda *a, **kw: FunctionModel(respond))
    path = tmp_path / "override.kedi"
    path.write_text("> model: codex/test\n>> The usable stock is [value: int].\n= <value>\n")
    code, output = run_program_cli(str(path), adapter="pydantic", model=None)
    assert code == 0, output
    assert output == "37"
