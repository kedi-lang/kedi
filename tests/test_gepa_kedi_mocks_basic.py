"""Unit tests for mock components used in GEPA optimizer testing.

Verifies that every piece of the RANGE[lo,hi] mock protocol behaves
deterministically and correctly in isolation, before any end-to-end tests.
"""

from __future__ import annotations

import json
from typing import Any, cast

import dspy
import pytest

from tests.dspy_mocks import (
    REFLECTION_STEP,
    LMCall,
    MockLM,
    PredictCall,
    RangeAwareAdapter,
    Recorder,
    install_mock_dspy,
    parse_first_range,
    sample_from_range,
)

# ── parse_first_range ────────────────────────────────────────────────


class TestParseFirstRange:
    def test_valid_positive(self):
        assert parse_first_range("prefix RANGE[10,20] suffix") == (10.0, 20.0)

    def test_valid_negative(self):
        assert parse_first_range("RANGE[-100,-50]") == (-100.0, -50.0)

    def test_valid_decimals(self):
        assert parse_first_range("RANGE[1.5,3.7]") == (1.5, 3.7)

    def test_returns_first_when_multiple(self):
        assert parse_first_range("RANGE[10,20] and RANGE[100,200]") == (10.0, 20.0)

    def test_missing_returns_default(self):
        assert parse_first_range("no range here") == (-100.0, -50.0)

    def test_custom_default(self):
        assert parse_first_range("nothing", default=(0.0, 1.0)) == (0.0, 1.0)

    def test_inverted_range_returns_default(self):
        assert parse_first_range("RANGE[20,10]") == (-100.0, -50.0)

    def test_equal_bounds_returns_default(self):
        assert parse_first_range("RANGE[5,5]") == (-100.0, -50.0)

    def test_non_string_returns_default(self):
        assert parse_first_range(42) == (-100.0, -50.0)  # type: ignore[arg-type]

    def test_nan_returns_default(self):
        assert parse_first_range("RANGE[NaN,NaN]") == (-100.0, -50.0)

    def test_whitespace_tolerant(self):
        assert parse_first_range("RANGE[ 10 , 20 ]") == (10.0, 20.0)


# ── sample_from_range ────────────────────────────────────────────────


class TestSampleFromRange:
    def test_midpoint_positive_range(self):
        ival, fval = sample_from_range("RANGE[10,20]", prefer_midpoint=True)
        assert ival == 15
        assert fval == 15.0

    def test_midpoint_negative_range(self):
        ival, fval = sample_from_range("RANGE[-100,-50]", prefer_midpoint=True)
        assert ival == -75
        assert fval == -75.0

    def test_seeded_is_deterministic(self):
        a = sample_from_range("RANGE[0,100]", prefer_midpoint=False)
        b = sample_from_range("RANGE[0,100]", prefer_midpoint=False)
        assert a == b

    def test_seeded_within_bounds(self):
        for text in ["RANGE[0,100]", "RANGE[-50,50]", "RANGE[1000,2000]"]:
            lo, hi = parse_first_range(text)
            ival, fval = sample_from_range(text, prefer_midpoint=False)
            assert lo <= fval <= hi
            assert lo <= ival <= hi

    def test_missing_range_uses_default(self):
        ival, fval = sample_from_range("no range", prefer_midpoint=True)
        assert ival == -75
        assert fval == -75.0


# ── MockLM optimizer response ────────────────────────────────────────


class TestMockLMOptimizer:
    def _call(self, prompt: str) -> dict[str, Any]:
        lm = MockLM(model="mock/optimizer")
        Recorder.reset()
        result = lm.forward(messages=[{"content": prompt}])
        return json.loads(result.choices[0].message.content)

    def test_produces_midpoint_for_int_field(self):
        prompt = (
            "Your output fields are:\n"
            "1. `y` (int): the answer\n\n"
            "Respond with a JSON object in the following order of fields: `y`.\n"
            "Instructions: RANGE[10,20]"
        )
        out = self._call(prompt)
        assert out["y"] == 15

    def test_produces_midpoint_for_float_field(self):
        prompt = (
            "Your output fields are:\n1. `score` (float): the score\n\nInstructions: RANGE[0,100]"
        )
        out = self._call(prompt)
        assert out["score"] == 50.0

    def test_produces_list_for_list_field(self):
        prompt = (
            "Your output fields are:\n1. `tags` (list[str]): tag list\n\nInstructions: RANGE[0,10]"
        )
        out = self._call(prompt)
        assert isinstance(out["tags"], list)
        assert len(out["tags"]) == 2

    def test_multiple_output_fields(self):
        prompt = (
            "Your output fields are:\n"
            "1. `x` (int): first\n"
            "2. `name` (str): second\n\n"
            "Instructions: RANGE[10,20]"
        )
        out = self._call(prompt)
        assert "x" in out
        assert "name" in out

    def test_value_changes_with_range(self):
        out_low = self._call("Your output fields are:\n1. `y` (int):\nInstructions: RANGE[0,10]")
        Recorder.reset()
        out_high = self._call(
            "Your output fields are:\n1. `y` (int):\nInstructions: RANGE[100,200]"
        )
        assert out_low["y"] < out_high["y"]

    def test_records_to_recorder(self):
        Recorder.reset()
        lm = MockLM(model="mock/optimizer")
        lm.forward(messages=[{"content": "RANGE[0,10] `y` (int)"}])
        assert len(Recorder.lm_calls) == 1
        assert Recorder.lm_calls[0].model == "mock/optimizer"
        assert "RANGE[0,10]" in Recorder.lm_calls[0].prompt


# ── MockLM reflection response ──────────────────────────────────────


class TestMockLMReflection:
    def _call(self, prompt: str) -> str:
        lm = MockLM(model="mock/reflection")
        Recorder.reset()
        result = lm.forward(messages=[{"content": prompt}])
        return result.choices[0].message.content

    def test_shifts_up_on_too_low(self):
        response = self._call("current RANGE[-100.000,-50.000] feedback: TOO_LOW")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(-100.0 + REFLECTION_STEP)
        assert hi == pytest.approx(-50.0 + REFLECTION_STEP)

    def test_shifts_down_on_too_high(self):
        response = self._call("current RANGE[100.000,150.000] feedback: TOO_HIGH")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(100.0 - REFLECTION_STEP)
        assert hi == pytest.approx(150.0 - REFLECTION_STEP)

    def test_no_shift_on_ok(self):
        response = self._call("current RANGE[40.000,60.000] feedback: OK")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(40.0)
        assert hi == pytest.approx(60.0)

    def test_no_shift_on_mixed_signals(self):
        response = self._call("RANGE[10.000,20.000] TOO_LOW and TOO_HIGH")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(20.0)

    def test_no_shift_when_no_feedback(self):
        response = self._call("RANGE[10.000,20.000] no specific feedback")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(20.0)

    def test_default_range_when_no_range_in_prompt(self):
        response = self._call("no range here TOO_LOW")
        lo, hi = parse_first_range(response)
        assert lo == pytest.approx(-100.0 + REFLECTION_STEP)
        assert hi == pytest.approx(-50.0 + REFLECTION_STEP)

    def test_records_to_recorder(self):
        Recorder.reset()
        self._call("RANGE[10,20] TOO_LOW")
        assert len(Recorder.lm_calls) == 1
        assert Recorder.lm_calls[0].model == "mock/reflection"


# ── Recorder helpers ─────────────────────────────────────────────────


class TestRecorder:
    def test_reset_clears_everything(self):
        Recorder.calls.append(PredictCall(instructions="x", kwargs={}))
        Recorder.lm_calls.append(LMCall(prompt="p", response="r", model="m"))
        Recorder.adapter_calls.append({"template": "t"})
        Recorder.reset()
        assert len(Recorder.calls) == 0
        assert len(Recorder.lm_calls) == 0
        assert len(Recorder.adapter_calls) == 0

    def test_optimizer_and_reflection_call_filters(self):
        Recorder.reset()
        Recorder.lm_calls.append(LMCall(prompt="a", response="b", model="mock/optimizer"))
        Recorder.lm_calls.append(LMCall(prompt="c", response="d", model="mock/reflection"))
        Recorder.lm_calls.append(LMCall(prompt="e", response="f", model="mock/optimizer"))
        assert len(Recorder.optimizer_calls()) == 2
        assert len(Recorder.reflection_calls()) == 1

    def test_predict_ranges_extraction(self):
        Recorder.reset()
        Recorder.calls.append(PredictCall(instructions="RANGE[10,20]", kwargs={}))
        Recorder.calls.append(PredictCall(instructions="RANGE[30,40]", kwargs={}))
        ranges = Recorder.predict_ranges()
        assert ranges == [(10.0, 20.0), (30.0, 40.0)]

    def test_call_index_increments(self):
        Recorder.reset()
        lm = MockLM(model="mock/optimizer")
        lm.forward(messages=[{"content": "first"}])
        lm.forward(messages=[{"content": "second"}])
        assert Recorder.lm_calls[0].index < Recorder.lm_calls[1].index


# ── RangeAwareAdapter ────────────────────────────────────────────────


class TestRangeAwareAdapter:
    def test_int_output(self):
        adapter = RangeAwareAdapter()
        result = adapter.produce_sync(
            template="RANGE[10,20]",
            output_schema={"val": int},
        )
        assert result["val"] == 15

    def test_float_output(self):
        adapter = RangeAwareAdapter()
        result = adapter.produce_sync(
            template="RANGE[10,20]",
            output_schema={"val": float},
        )
        assert result["val"] == 15.0

    def test_str_output(self):
        adapter = RangeAwareAdapter()
        result = adapter.produce_sync(
            template="RANGE[10,20]",
            output_schema={"val": str},
        )
        assert result["val"] == "15.0"

    def test_list_output(self):
        adapter = RangeAwareAdapter()
        result = adapter.produce_sync(
            template="RANGE[10,20]",
            output_schema={"items": list[str]},
        )
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2

    def test_records_call(self):
        Recorder.reset()
        adapter = RangeAwareAdapter()
        adapter.produce_sync(template="RANGE[5,15]", output_schema={"v": int})
        assert len(Recorder.adapter_calls) == 1
        assert Recorder.adapter_calls[0]["range"] == (5.0, 15.0)
        assert Recorder.adapter_calls[0]["ival"] == 10

    def test_empty_schema_returns_empty(self):
        adapter = RangeAwareAdapter()
        result = adapter.produce_sync(template="RANGE[0,10]", output_schema={})
        assert len(result) == 0


# ── install_mock_dspy integration ────────────────────────────────────


class TestInstallMockDspy:
    def test_predict_records_calls(self, monkeypatch: pytest.MonkeyPatch):
        install_mock_dspy(monkeypatch)
        Recorder.reset()

        sig = (
            cast(Any, dspy)
            .Signature("foo: str -> bar: str")
            .with_instructions("Test instruction with RANGE[0,10]")
        )
        pred = dspy.Predict(sig)
        pred(foo="hello")

        assert len(Recorder.calls) >= 1
        last = Recorder.calls[-1]
        assert "RANGE[0,10]" in last.instructions
        assert "foo" in last.kwargs

    def test_lm_factory_returns_mock(self, monkeypatch: pytest.MonkeyPatch):
        install_mock_dspy(monkeypatch)
        lm = dspy.LM("my-model/test")
        assert isinstance(lm, MockLM)
        assert lm.model == "my-model/test"

    def test_reflection_model_routes_correctly(self, monkeypatch: pytest.MonkeyPatch):
        install_mock_dspy(monkeypatch)
        Recorder.reset()
        lm = dspy.LM("mock/reflection")
        lm.forward(messages=[{"content": "RANGE[0,10] TOO_LOW"}])
        assert len(Recorder.reflection_calls()) == 1
        assert len(Recorder.optimizer_calls()) == 0
