from unittest.mock import Mock

import pytest

dspy = pytest.importorskip("dspy")

from kedi.optimization.config import OptimizerConfig  # noqa: E402
from kedi.optimization.gepa_reflection import build_reflection_lm  # noqa: E402
from kedi.reporter import BufferReporter  # noqa: E402


def test_default_reflection_reuses_context_lm_and_is_accepted_by_gepa():
    lm = Mock()
    reporter = BufferReporter()
    with dspy.context(lm=lm):
        reflection = build_reflection_lm(OptimizerConfig(), reporter)
        optimizer = dspy.GEPA(
            metric=lambda *args: 0.0, max_metric_calls=4, reflection_lm=reflection
        )
    assert reflection is lm
    assert optimizer.reflection_lm is lm
    assert "main DSPy LM" in reporter.getvalue()


def test_missing_main_and_reflection_model_fails_before_optimization():
    with dspy.context(lm=None), pytest.raises(ValueError, match="main DSPy LM"):
        build_reflection_lm(OptimizerConfig(), BufferReporter())


def test_explicit_reflection_model_keeps_separate_configuration(monkeypatch):
    constructor = Mock(return_value=Mock(return_value=["candidate instruction"]))
    monkeypatch.setattr("kedi.optimization.gepa_reflection.ThinkingStrippingLM", constructor)
    with dspy.context(lm=Mock()):
        reflection = build_reflection_lm(
            OptimizerConfig(reflection_model="openai:gpt-6-sol", reflection_model_max_tokens=9000),
            BufferReporter(),
        )
    constructor.assert_called_once_with(
        model="openai/gpt-6-sol", temperature=1.0, max_tokens=9000, timeout=300
    )
    assert reflection("Improve this instruction") == ["candidate instruction"]
    constructor.return_value.assert_called_once_with("Improve this instruction")


def test_explicit_reflection_lm_is_a_base_lm_and_forwards_messages(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    base = Mock(return_value=["sync"])
    base.acall = AsyncMock(return_value=["async"])
    monkeypatch.setattr(
        "kedi.optimization.gepa_reflection.ThinkingStrippingLM", Mock(return_value=base)
    )
    reporter = BufferReporter()
    reflection = build_reflection_lm(OptimizerConfig(reflection_model="openai:gpt-6-sol"), reporter)
    messages = [{"role": "user", "content": "Improve"}]

    assert isinstance(reflection, dspy.BaseLM)
    assert reflection(messages=messages) == ["sync"]
    base.assert_called_once_with(None, messages=messages)
    assert asyncio.run(reflection.acall("Improve", temperature=0.5)) == ["async"]
    base.acall.assert_awaited_once_with("Improve", temperature=0.5)
    assert "Reflection LM call #2" in reporter.getvalue()
