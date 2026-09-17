from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import codex_responses_model
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_profile import AgentProfile


class _SessionTestModel(TestModel):
    def __init__(self) -> None:
        super().__init__(custom_output_text="ok")
        self.session_enters = 0
        self.session_exits = 0

    @asynccontextmanager
    async def responses_session(self) -> AsyncIterator[None]:
        self.session_enters += 1
        try:
            yield
        finally:
            self.session_exits += 1


class _IsolatedSessionTestModel(_SessionTestModel):
    supports_responses_session_isolation = True

    def __init__(self) -> None:
        super().__init__()
        self.isolation_flags: list[bool] = []

    @asynccontextmanager
    async def responses_session(self, *, isolate: bool = False) -> AsyncIterator[None]:
        self.isolation_flags.append(isolate)
        async with super().responses_session():
            yield


def _fake_helper(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str, str]],
    options: list[dict[str, object]] | None = None,
) -> None:
    helper = ModuleType("codex_auth_helper")

    class CodexAuthConfig:
        def __init__(self, *, auth_path: Path) -> None:
            self.auth_path = auth_path

    def pydantic_factory(
        model: str,
        *,
        instructions: str,
        config: object | None = None,
        **kwargs: object,
    ) -> object:
        calls.append(("pydantic", model, instructions))
        if options is not None:
            options.append(dict(kwargs))
        return SimpleNamespace(config=config)

    def langchain_factory(
        model: str,
        *,
        instructions: str,
        config: object | None = None,
        **kwargs: object,
    ) -> object:
        calls.append(("langchain", model, instructions))
        if options is not None:
            options.append(dict(kwargs))
        return SimpleNamespace(config=config)

    setattr(helper, "CodexAuthConfig", CodexAuthConfig)
    setattr(helper, "CodexResponsesConnection", object())
    setattr(helper, "CodexResponsesTransportEvent", object())
    setattr(helper, "create_codex_responses_model", pydantic_factory)
    setattr(helper, "create_codex_chat_openai", langchain_factory)
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)


def test_codex_responses_model_builds_supported_framework_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _fake_helper(monkeypatch, calls)

    assert codex_responses_model("codex/gpt-5.6-luna", "pydantic") is not None
    assert codex_responses_model("gpt-5.6-luna", "langchain") is not None
    assert [(adapter, model) for adapter, model, _instructions in calls] == [
        ("pydantic", "gpt-5.6-luna"),
        ("langchain", "gpt-5.6-luna"),
    ]
    assert all("active instructions" in instructions for _, _, instructions in calls)


@pytest.mark.parametrize(
    "options",
    [
        {"connection": "invalid"},
        {"fallback": "invalid"},
        {"connection": "http", "fallback": "http"},
    ],
)
def test_codex_rejects_invalid_transport_options(monkeypatch, options):
    _fake_helper(monkeypatch, [])
    with pytest.raises(ValueError):
        codex_responses_model("test", "pydantic", **options)


def test_codex_does_not_hide_missing_transitive_dependency(monkeypatch):
    import builtins

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "codex_auth_helper":
            raise ModuleNotFoundError("missing transport dependency", name="transport_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError, match="transport dependency"):
        codex_responses_model("test", "pydantic")


def test_codex_responses_model_forwards_only_explicit_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    options: list[dict[str, object]] = []
    _fake_helper(monkeypatch, calls, options)

    codex_responses_model("codex/gpt-5.6-luna", "pydantic")
    codex_responses_model(
        "codex/gpt-5.6-luna",
        "pydantic",
        connection="websocket",
        fallback="http",
    )

    assert options == [
        {},
        {"connection": "websocket", "fallback": "http"},
    ]


def test_codex_responses_model_forwards_transport_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    options: list[dict[str, object]] = []
    _fake_helper(monkeypatch, calls, options)

    def observer(_event: object) -> None:
        return None

    codex_responses_model(
        "codex/gpt-5.6-luna",
        "pydantic",
        connection="websocket",
        transport_observer=observer,
    )

    assert options == [
        {"connection": "websocket", "transport_observer": observer},
    ]


def test_codex_responses_model_forwards_langchain_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    options: list[dict[str, object]] = []
    _fake_helper(monkeypatch, calls, options)

    def observer(event: object) -> None:
        pass

    codex_responses_model(
        "codex/gpt-5.6-luna",
        "langchain",
        connection="websocket",
        fallback="http",
        transport_observer=observer,
    )
    assert options == [
        {"connection": "websocket", "fallback": "http", "transport_observer": observer}
    ]


def test_codex_responses_model_uses_explicit_auth_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _fake_helper(monkeypatch, calls)
    auth_path = tmp_path / "auth.json"
    monkeypatch.setenv("KEDI_CODEX_AUTH_FILE", str(auth_path))

    model = cast(Any, codex_responses_model("codex/gpt-5.6-luna", "pydantic"))

    assert model.config.auth_path == auth_path


def test_codex_responses_model_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "codex_auth_helper", None)

    with pytest.raises(ModuleNotFoundError, match=r"kedi\[codex-model\]"):
        codex_responses_model("codex/gpt-5.6-luna", "pydantic")


def test_codex_responses_model_reports_missing_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = ModuleType("codex_auth_helper")

    def missing_login(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("auth.json")

    setattr(helper, "create_codex_responses_model", missing_login)
    setattr(helper, "create_codex_chat_openai", missing_login)
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)

    with pytest.raises(RuntimeError, match="codex login"):
        codex_responses_model("codex/gpt-5.6-luna", "pydantic")


def test_codex_responses_model_reports_old_helper_for_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = ModuleType("codex_auth_helper")
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)

    with pytest.raises(RuntimeError, match="newer codex-auth-helper"):
        codex_responses_model(
            "codex/gpt-5.6-luna",
            "pydantic",
            connection="websocket",
        )


def test_codex_responses_model_reports_old_helper_for_transport_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = ModuleType("codex_auth_helper")
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)

    with pytest.raises(RuntimeError, match="transport observation"):
        codex_responses_model(
            "codex/gpt-5.6-luna",
            "pydantic",
            transport_observer=lambda _event: None,
        )


@pytest.mark.parametrize("model_id", ["codex/", "codex/   ", "https://example.test/model"])
def test_codex_responses_model_rejects_invalid_ids(model_id: str) -> None:
    with pytest.raises(ValueError, match="Codex model id"):
        codex_responses_model(model_id, "pydantic")


def test_pydantic_adapter_resolves_and_reuses_codex_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    models = {
        "codex/gpt-5.6-luna": TestModel(custom_output_text="default"),
        "codex/gpt-5.6-terra": TestModel(custom_output_text="override"),
    }

    def build(model_id: str, *, adapter: str) -> Any:
        assert adapter == "pydantic"
        calls.append(model_id)
        return models[model_id]

    monkeypatch.setattr(pydantic_adapter_module, "codex_responses_model", build)

    adapter = PydanticAdapter("codex/gpt-5.6-luna")
    assert adapter.default_model() == "codex/gpt-5.6-luna"
    assert calls == ["codex/gpt-5.6-luna"]

    with adapter.profile_override(AgentProfile(model="codex/gpt-5.6-terra", effort="high")):
        assert adapter.active_profile.model == "codex/gpt-5.6-terra"
    with adapter.profile_override(AgentProfile(model="codex/gpt-5.6-terra")):
        pass

    assert calls == ["codex/gpt-5.6-luna", "codex/gpt-5.6-terra"]


def test_pydantic_adapter_owns_one_responses_session_per_run() -> None:
    async def run() -> None:
        model = _SessionTestModel()
        adapter = PydanticAdapter(model)

        assert await adapter.invoke(prompt="hello") == "ok"

        assert model.session_enters == 1
        assert model.session_exits == 1

    asyncio.run(run())


def test_pydantic_native_iter_owns_one_responses_session() -> None:
    async def run() -> None:
        model = _SessionTestModel()
        adapter = PydanticAdapter(model)

        async with adapter.iter("hello") as agent_run:
            async for _ in agent_run:
                pass

        assert model.session_enters == 1
        assert model.session_exits == 1

    asyncio.run(run())


def test_pydantic_sync_runs_close_each_responses_session() -> None:
    model = _SessionTestModel()
    adapter = PydanticAdapter(model)

    assert adapter.invoke_sync(prompt="first") == "ok"
    assert adapter.invoke_sync(prompt="second") == "ok"

    assert model.session_enters == 2
    assert model.session_exits == 2


def test_concurrent_pydantic_runs_request_isolated_responses_sessions() -> None:
    async def run() -> None:
        model = _IsolatedSessionTestModel()
        adapter = PydanticAdapter(model)

        assert await asyncio.gather(
            adapter.invoke(prompt="first"),
            adapter.invoke(prompt="second"),
        ) == ["ok", "ok"]

        assert model.session_enters == 2
        assert model.session_exits == 2
        assert model.isolation_flags == [True, True]

    asyncio.run(run())


def test_codex_responses_model_rejects_unknown_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _fake_helper(monkeypatch, calls)

    with pytest.raises(ValueError, match="Unsupported Codex model adapter"):
        codex_responses_model("codex/gpt-5.6-luna", cast(Any, "dspy"))


def test_explicit_adapter_session_reuses_connection_across_native_and_kedi_runs() -> None:
    async def run() -> None:
        model = _IsolatedSessionTestModel()
        adapter = PydanticAdapter(model)
        async with adapter.responses_session():
            assert await adapter.invoke(prompt="first") == "ok"
            async with adapter.responses_session():
                assert (await adapter.run("second")).output == "ok"
            async with adapter.iter("third") as agent_run:
                async for _ in agent_run:
                    pass
            assert model.session_enters == 1
            assert model.session_exits == 0
        assert model.session_exits == 1
        assert await adapter.invoke(prompt="independent") == "ok"
        assert model.session_enters == model.session_exits == 2

    asyncio.run(run())


def test_explicit_adapter_session_does_not_share_socket_with_child_tasks() -> None:
    async def run() -> None:
        model = _IsolatedSessionTestModel()
        adapter = PydanticAdapter(model)
        async with adapter.responses_session():
            assert await adapter.invoke(prompt="parent") == "ok"
            assert await asyncio.gather(adapter.invoke(prompt="a"), adapter.invoke(prompt="b")) == [
                "ok",
                "ok",
            ]
            assert model.session_enters == 3
            assert model.session_exits == 2
        assert model.session_exits == 3
        assert model.isolation_flags == [True, True, True]

    asyncio.run(run())


def test_explicit_adapter_session_closes_on_cancellation() -> None:
    async def run() -> None:
        model = _IsolatedSessionTestModel()
        adapter = PydanticAdapter(model)
        ready = asyncio.Event()
        blocked = asyncio.Event()

        async def consume() -> None:
            async with adapter.responses_session():
                assert await adapter.invoke(prompt="first") == "ok"
                ready.set()
                await blocked.wait()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(ready.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert model.session_enters == model.session_exits == 1

    asyncio.run(run())


def test_explicit_adapter_session_uses_model_override_and_closes_on_error() -> None:
    async def run() -> None:
        original = _IsolatedSessionTestModel()
        replacement = _IsolatedSessionTestModel()
        adapter = PydanticAdapter(original)
        with adapter.override(model=replacement):
            with pytest.raises(ValueError, match="application failure"):
                async with adapter.responses_session():
                    assert await adapter.invoke(prompt="hello") == "ok"
                    raise ValueError("application failure")
        assert replacement.session_enters == replacement.session_exits == 1
        assert original.session_enters == 0

    asyncio.run(run())


def test_explicit_adapter_session_rejects_unsupported_models() -> None:
    async def run() -> None:
        adapter = PydanticAdapter(TestModel())
        with pytest.raises(TypeError, match="does not support Responses sessions"):
            async with adapter.responses_session():
                pytest.fail("unsupported session entered")

    asyncio.run(run())
