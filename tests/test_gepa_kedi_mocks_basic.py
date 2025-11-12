from __future__ import annotations

from typing import Any, Dict

import pytest

from kedi.lang import compile_program, parse_program
from tests.dspy_mocks import (
    Recorder,
    RangeAwareAdapter,
    install_mock_dspy,
    parse_first_range,
    sample_from_range,
)


def test_range_parsing_and_midpoint():
    text = "foo RANGE[10,20] bar RANGE[100,200]"
    lo, hi = parse_first_range(text)
    assert lo == 10.0 and hi == 20.0
    ival, fval = sample_from_range(text, prefer_midpoint=True)
    assert ival == 15
    assert fval == 15.0


def test_range_parsing_default_when_missing():
    text = "no range token here"
    lo, hi = parse_first_range(text)
    assert lo == -100.0 and hi == -50.0
    ival, fval = sample_from_range(text, prefer_midpoint=True)
    assert ival == -75
    assert fval == -75.0


def test_range_adapter_generates_within_range():
    src = """
@once(seed: str) -> int:
  The answer is [y: int]
  = `y`
"""
    rt = compile_program(parse_program(src), adapter=RangeAwareAdapter())
    Recorder.reset()
    # Call the procedure via _invoke_procedure to set up environment
    once_proc = rt._procedures["once"]
    res = rt._invoke_procedure(once_proc, ["x"])
    assert isinstance(res, int)
    # Validate the adapter call recorded a range and output within that range defaults
    assert Recorder.adapter_calls
    last = Recorder.adapter_calls[-1]
    lo, hi = last["range"]
    assert lo < hi
    assert lo <= last["ival"] <= hi


def test_predict_records_calls(monkeypatch: pytest.MonkeyPatch):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    # Use real dspy.Predict with our mocked LM
    import dspy
    
    # Create a simple signature
    sig = dspy.Signature("foo -> bar").with_instructions("Generate using RANGE[0,10]")
    p = dspy.Predict(sig)
    
    # Call it
    _ = p(foo="test_value")
    
    # Verify recording
    assert Recorder.calls
    call = Recorder.calls[-1]
    assert "RANGE[0,10]" in call.instructions
    assert "foo" in call.kwargs


