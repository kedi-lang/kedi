from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import DSPyAdapter
from kedi.agent_profile import AgentProfile


@contextmanager
def _record_dspy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Capture which LM is active via ``dspy.context`` during a produce call."""
    contexts: list[Any] = []

    class FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs.get("model")
            self.kwargs = kwargs

    @contextmanager
    def fake_context(*, lm: Any) -> Iterator[None]:
        contexts.append(lm)
        try:
            yield
        finally:
            contexts.append(("exit", lm))

    class FakeSettings:
        def configure(self, **_: Any) -> None:
            return None

    class FakeSignature:
        def with_instructions(self, _instructions: str | None) -> "FakeSignature":
            return self

    class FakePredict:
        def __init__(self, _signature: Any) -> None:
            pass

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return dict(kwargs)

    monkeypatch.setattr(adapters_module, "LM", FakeLM)
    monkeypatch.setattr(adapters_module.dspy, "settings", FakeSettings())
    monkeypatch.setattr(adapters_module.dspy, "context", fake_context)
    monkeypatch.setattr(adapters_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(lambda _schema: FakeSignature()))
    yield contexts


def test_plain_produce_runs_inside_default_lm_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the produce path is authoritative via ``dspy.context``."""
    with _record_dspy(monkeypatch) as contexts:
        adapter = DSPyAdapter(model="openai/gpt-4o-mini")
        adapter.produce_sync("prompt", {"answer": str})

    # Default LM context was entered (and exited) for the produce call.
    entered = [lm for lm in contexts if not isinstance(lm, tuple)]
    assert len(entered) == 1
    assert entered[0].model == "openai/gpt-4o-mini"


def test_active_profile_context_is_not_overridden_by_produce_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an active profile override must win; produce adds no default."""
    with _record_dspy(monkeypatch) as contexts:
        adapter = DSPyAdapter(model="openai/gpt-4o-mini")
        with adapter.profile_override(AgentProfile(model="anthropic/claude")):
            contexts.clear()
            adapter.produce_sync("prompt", {"answer": str})

    # While a profile override is on the stack, produce_sync does not enter a
    # second (default) LM context that would shadow the profile's context.
    entered = [lm for lm in contexts if not isinstance(lm, tuple)]
    assert entered == []
    assert adapter.active_profile == AgentProfile.empty()


def test_profile_override_uses_profile_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _record_dspy(monkeypatch) as contexts:
        adapter = DSPyAdapter(model="openai/gpt-4o-mini")
        with adapter.profile_override(AgentProfile(model="anthropic/claude")):
            pass

    entered = [lm for lm in contexts if not isinstance(lm, tuple)]
    assert len(entered) == 1
    assert entered[0].model == "anthropic/claude"
