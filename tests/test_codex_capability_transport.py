"""Endpoint-specific model profiles must stay local to the owning run."""

import re
from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolAvailabilityDeltaPart, ToolCallPart
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi import Capability
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.budget import BudgetedModel
from kedi.agent_adapter.adapters.pydantic.compat import (
    _transport_profile_model,
    transport_model_scope,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope

CodexResponsesModel = pytest.importorskip("codex_auth_helper.model").CodexResponsesModel


class Receipt(BaseModel):
    receipt: str


@pytest.mark.asyncio
async def test_transport_override_preserves_wrapper_and_shared_client():
    from pydantic_ai.models.wrapper import WrapperModel

    model = LocalCodexModel()
    wrapper = WrapperModel(model)
    wrapper.owner_label = "caller-owned"
    original_profile = model.profile.copy()
    try:
        prepared = _transport_profile_model(wrapper)
        assert prepared is not wrapper
        assert prepared.owner_label == "caller-owned"
        assert prepared.wrapped is not model
        assert prepared.wrapped.client is model.client
        assert prepared.wrapped.profile["tool_addition_mode"] is None
        assert model.profile == original_profile
        assert wrapper.wrapped is model
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_broken_optional_transport_dependency_is_not_silently_ignored(monkeypatch):
    import builtins

    model = LocalCodexModel()
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "codex_auth_helper.model":
            raise ModuleNotFoundError("Missing transport dependency", name="transport_dependency")
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", broken_import)
            with pytest.raises(ModuleNotFoundError, match="Missing transport dependency"):
                _transport_profile_model(model)
    finally:
        await model.client.close()


class LocalCodexModel(CodexResponsesModel):
    def __init__(self, connection="websocket"):
        super().__init__(
            "gpt-6-luna",
            default_instructions="Follow the active instructions.",
            provider=OpenAIProvider(api_key="test-not-a-credential"),
            connection=connection,
        )
        self.client.max_retries = 0
        self.requests = []
        self.sessions = []

    @asynccontextmanager
    async def responses_session(self, *, isolate=False):
        self.sessions.append(isolate)
        yield

    @asynccontextmanager
    async def responses_turn(self):
        yield

    async def request(self, messages, model_settings, model_request_parameters):
        _, parameters = self.prepare_request(model_settings, model_request_parameters)
        self.requests.append((self, messages, parameters))
        step = len(self.requests)
        if step == 1:
            instructions = "\n".join(part.content for part in parameters.instruction_parts or ())
            identity = re.search(r"kedi_inventory_[a-f0-9]{12}", instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        if step == 2:
            return ModelResponse([ToolCallPart("read_inventory", {}, "read")])
        if parameters.output_tools:
            return ModelResponse(
                [ToolCallPart(parameters.output_tools[0].name, {"receipt": "verified"}, "result")]
            )
        return ModelResponse([TextPart("verified")])


@pytest.mark.parametrize("binding", ["default", "explicit", "override"])
@pytest.mark.parametrize("entry", ["native", "invoke", "produce"])
@pytest.mark.asyncio
async def test_websocket_deferred_tools_use_scoped_compatible_profile(binding, entry):
    model = LocalCodexModel()
    original = model.profile.copy()
    calls = []

    def read_inventory() -> str:
        calls.append("read")
        return "verified"

    capability = Capability(
        name="inventory", identity="fixture.inventory", tools=(read_inventory,), defer=True
    )
    tool = ToolSpec(
        name="read_inventory",
        fn=read_inventory,
        risk="read_only",
        deferred_capability="fixture.inventory",
    )
    adapter = PydanticAdapter(model if binding == "default" else None)
    from contextlib import nullcontext

    kwargs = {"model": model} if binding == "explicit" else {}
    try:
        with adapter.override(model=model) if binding == "override" else nullcontext():
            with adapter.profile_override(capability.as_profile()), adapter.tool_scope((tool,)):
                if entry == "native":
                    actual = (await adapter.run("Read inventory.", **kwargs)).output
                elif entry == "invoke":
                    actual = await adapter.invoke(prompt="Read inventory.", **kwargs)
                else:
                    actual = (
                        await adapter.produce(
                            template="The receipt is [receipt: str].", output_type=Receipt, **kwargs
                        )
                    ).receipt
        assert actual == "verified"
        assert calls == ["read"]
        assert len(model.requests) == 3
        assert all(
            active is not model and active.client is model.client for active, _, _ in model.requests
        )
        assert all(active.profile["tool_addition_mode"] is None for active, _, _ in model.requests)
        assert all(active.profile["tool_deferral_mode"] is None for active, _, _ in model.requests)
        assert not any(
            isinstance(part, ToolAvailabilityDeltaPart)
            for _, messages, _ in model.requests
            for message in messages
            for part in message.parts
        )
        first, second = (model.requests[i][2] for i in (0, 1))
        assert first.visibility_of("read_inventory") == "withheld"
        assert second.visibility_of("read_inventory") == "visible"
        assert model.profile == original
        assert not adapter.active_profile.capabilities
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_transport_profile_preserves_budget_admission_and_unwinds_on_failure():
    model = LocalCodexModel()
    profile = model.profile.copy()
    adapter = PydanticAdapter(model)
    budget = RunBudget(request_limit=0)
    try:
        with run_budget_scope(budget):
            with transport_model_scope(adapter, model):
                active = adapter._get_model_outside_run()
                assert isinstance(active, BudgetedModel)
                assert active.wrapped.client is model.client
                with pytest.raises(RunBudgetExceeded):
                    await adapter.run("Do not make a model request.")
        assert adapter._get_model_outside_run() is model
        assert model.profile == profile and not model.requests
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_http_and_other_provider_profiles_are_not_changed():
    model = LocalCodexModel(connection="http")
    other = OpenAIResponsesModel(
        "gpt-6-luna", provider=OpenAIProvider(api_key="test-not-a-credential")
    )
    try:
        assert _transport_profile_model(model) is model
        assert _transport_profile_model(other) is other
        first = _transport_profile_model(LocalCodexModel())
        try:
            assert _transport_profile_model(first) is first
        finally:
            await first.client.close()
    finally:
        await model.client.close()
        await other.client.close()


def test_unrelated_websocket_model_does_not_require_the_optional_helper(monkeypatch):
    import builtins

    from pydantic_ai.models.test import TestModel

    class OtherWebSocketModel(TestModel):
        responses_connection = "websocket"

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "codex_auth_helper.model":
            raise ModuleNotFoundError("Optional package is absent", name="codex_auth_helper")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    model = OtherWebSocketModel()
    assert _transport_profile_model(model) is model


@pytest.mark.asyncio
async def test_scoped_transport_profile_charges_each_generation_once():
    model = LocalCodexModel()
    adapter = PydanticAdapter(model)
    capability = Capability(name="inventory", identity="fixture.inventory", defer=True)
    try:
        with run_budget_scope(RunBudget(request_limit=1)) as ledger:
            with adapter.profile_override(capability.as_profile()):
                with pytest.raises(RunBudgetExceeded):
                    await adapter.run("Load inventory, then report.")
            assert len(model.requests) == 1
            assert ledger.spent("request") == 1
        assert adapter._get_model_outside_run() is model
    finally:
        await model.client.close()
