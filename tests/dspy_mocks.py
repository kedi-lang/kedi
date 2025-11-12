from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytest


# =========
# Utilities
# =========

_RANGE_RE = re.compile(r"RANGE\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_first_range(text: str, default: Tuple[float, float] = (-100.0, -50.0)) -> Tuple[float, float]:
    """Parse the first RANGE[min,max] token from text. Returns default if absent or invalid."""
    if not isinstance(text, str):
        return default
    m = _RANGE_RE.search(text)
    if not m:
        return default
    try:
        lo, hi = float(m.group(1)), float(m.group(2))
    except Exception:
        return default
    if not (lo < hi):
        return default
    return (lo, hi)


def deterministic_uniform_0_1(key: str) -> float:
    """Deterministic pseudo-random in [0,1) derived from a stable key."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # Use first 8 bytes as integer, normalize
    val = int.from_bytes(h[:8], "big")
    return (val % 10_000_000) / 10_000_000.0


def sample_from_range(template: str, *, prefer_midpoint: bool = True) -> Tuple[int, float]:
    """Produce an (int_value, float_value) determined by RANGE[...] in template."""
    lo, hi = parse_first_range(template)
    if prefer_midpoint:
        mid = (lo + hi) / 2.0
        return (int(round(mid)), float(mid))
    # seeded by template to be deterministic
    frac = deterministic_uniform_0_1(template)
    value = lo + (hi - lo) * frac
    return (int(round(value)), float(value))


# =========
# Recorder
# =========

@dataclass
class PredictCall:
    instructions: str
    kwargs: Dict[str, Any]


@dataclass
class LMCall:
    prompt: str
    response: str
    model: str


class Recorder:
    calls: list[PredictCall] = []
    adapter_calls: list[dict[str, Any]] = []
    lm_calls: list[LMCall] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()
        cls.adapter_calls.clear()
        cls.lm_calls.clear()


# =======================
# Range-aware Test Adapter
# =======================

from kedi.agent_adapter import AgentAdapter
from kedi.utils import AttrDict


class RangeAwareAdapter(AgentAdapter[AttrDict[str, Any]]):
    """Adapter that generates outputs guided by RANGE[min,max] tokens present in the prompt."""

    def __init__(self, *, prefer_midpoint: bool = True):
        self.prefer_midpoint = prefer_midpoint

    async def produce(
        self,
        *,
        template: str,
        output_schema: Dict[str, type],
        **_: Any,
    ) -> AttrDict[str, Any]:
        # Simple async wrapper
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: Dict[str, type] | None = None,
        **_: Any,
    ) -> AttrDict[str, Any]:
        ival, fval = sample_from_range(template, prefer_midpoint=self.prefer_midpoint)
        lo, hi = parse_first_range(template)
        
        # Record the call for test assertions
        Recorder.adapter_calls.append({
            "template": template,
            "range": (lo, hi),
            "ival": ival,
            "fval": fval,
        })
        
        # Generate outputs based on schema
        data: Dict[str, Any] = {}
        if output_schema:
            for key, typ in output_schema.items():
                if typ == int or typ is int:
                    data[key] = ival
                elif typ == float or typ is float:
                    data[key] = fval
                elif hasattr(typ, "__origin__"):
                    # Handle list[str], etc
                    origin = getattr(typ, "__origin__", None)
                    if origin in (list, List):
                        data[key] = [f"{key}_item1", f"{key}_item2"]
                    else:
                        data[key] = str(fval)
                else:
                    data[key] = str(fval)
        
        return AttrDict(data)


# ======================
# Mocked LM Layer
# ======================

_REFLECTION_FAIL_NEXT = False


def reflection_fail_once() -> None:
    """Configure the mock reflection LM to emit invalid content on its next call."""
    global _REFLECTION_FAIL_NEXT
    _REFLECTION_FAIL_NEXT = True


import dspy
from dspy import BaseLM


class MockLM(BaseLM):
    """Mock LM that intercepts calls and returns controlled responses.
    
    This replaces the actual LLM calls while keeping real DSPy Predict/Module structure.
    """
    
    def __init__(self, model: str = "mock/optimizer", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.history = []
    
    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any):
        """Main LM forward method - returns object with 'choices' containing completions."""
        # Extract the actual prompt text - combine all messages to capture system message with instructions
        if messages:
            # Chat format - combine all messages to get full context including system instructions
            if isinstance(messages, list) and messages:
                # Combine all message contents
                all_content = []
                for msg in messages:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    all_content.append(content)
                prompt_text = "\n".join(all_content)
            else:
                prompt_text = str(messages)
        elif prompt:
            prompt_text = str(prompt)
        else:
            prompt_text = ""
        
        
        
        # Determine response based on model type
        if "reflection" in self.model:
            response = self._reflection_response(prompt_text)
        else:
            response = self._optimizer_response(prompt_text)
        
        # Record the call
        Recorder.lm_calls.append(LMCall(
            prompt=prompt_text,
            response=response,
            model=self.model
        ))
        
        # Return in DSPy format (object with choices attribute)
        from types import SimpleNamespace
        choice = SimpleNamespace(message=SimpleNamespace(content=response))
        # Usage needs to be dict-like for dict() conversion
        class UsageDict(dict):
            def __init__(self):
                super().__init__(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0
        
        return SimpleNamespace(
            choices=[choice],
            usage=UsageDict(),
            model=self.model
        )
    
    def _reflection_response(self, prompt: str) -> str:
        """Generate reflection LM response that adjusts RANGE based on feedback."""
        global _REFLECTION_FAIL_NEXT
        
        if _REFLECTION_FAIL_NEXT:
            _REFLECTION_FAIL_NEXT = False
            return "RANGE[NaN,NaN]"
        
        # Read current range from the prompt
        lo, hi = parse_first_range(prompt, default=(-100.0, -50.0))
        
        # Adjust based on feedback - use VERY LARGE steps to reach target in 1-2 iterations
        txt = prompt or ""
        # For testing, we want to converge quickly, so use a large fixed step
        # that will reach the target range [40,60] in just 1-2 iterations
        step = 100.0  # Large enough to jump from [-100,-50] to [0,50] to [40,60]
        
        if "TOO_LOW" in txt and "TOO_HIGH" not in txt:
            lo += step
            hi += step
        elif "TOO_HIGH" in txt and "TOO_LOW" not in txt:
            lo -= step
            hi -= step
        # else keep as is (OK or mixed)
        
        # Return new instruction
        return f"Generate numeric outputs using RANGE[{lo:.3f},{hi:.3f}]."
    
    def _optimizer_response(self, prompt: str) -> str:
        """Generate optimizer LM response - produces JSON with output fields."""
        import json
        import re
        
        # Extract RANGE if present
        lo, hi = parse_first_range(prompt, default=(-100.0, -50.0))
        ival, fval = sample_from_range(prompt, prefer_midpoint=True)
        
        # Parse output field names from the prompt
        # DSPy prompts typically list output fields like:
        # "Your output fields are:\n1. `field_name` (type):"
        # or "Respond with a JSON object in the following order of fields: `field1`, `field2`"
        outputs = {}
        
        # Try to find field names in backticks
        field_pattern = r'`([a-zA-Z_][a-zA-Z0-9_]*)`'
        fields_found = re.findall(field_pattern, prompt)
        
        # Filter out common non-field words
        exclude = {'foo', 'bar', 'completed', 'input', 'output', 'fields', 'order'}
        output_fields = [f for f in fields_found if f not in exclude]
        
        # If we found "output fields are" section, extract from there
        output_section_match = re.search(r'Your output fields are:(.*?)(?:All interactions|Respond|$)', prompt, re.DOTALL)
        if output_section_match:
            output_section = output_section_match.group(1)
            output_fields = re.findall(field_pattern, output_section)
        
        # If we found "Respond with a JSON object" section
        respond_match = re.search(r'Respond with a JSON object.*?fields:\s*(.+?)\.', prompt, re.DOTALL)
        if respond_match:
            respond_section = respond_match.group(1)
            output_fields = re.findall(field_pattern, respond_section)
        
        # Generate outputs for each field
        for field in output_fields:
            # Determine type based on context around the field
            field_context = prompt[max(0, prompt.find(field)-50):prompt.find(field)+50].lower()
            
            if "int" in field_context:
                outputs[field] = ival
            elif "float" in field_context:
                outputs[field] = fval
            elif "list" in field_context or "tags" in field:
                outputs[field] = [f"{field}_item1", f"{field}_item2"]
            elif "str" in field_context:
                outputs[field] = str(ival)
            else:
                # Default: use int for numeric-looking fields, str otherwise
                if any(c in field for c in ['y', 'v', 'w', 'num', 'val', 'count']):
                    outputs[field] = ival
                else:
                    outputs[field] = str(ival)
        
        # If no outputs detected, return empty (will cause error, which is correct)
        if not outputs:
            outputs = {"unknown": ival}
        
        return json.dumps(outputs)


def install_mock_dspy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch dspy to use real Predict/Module but with mocked LM layer."""
    import dspy
    
    # Create mock LM instances
    optimizer_lm = MockLM(model="mock/optimizer")
    
    # Patch the LM factory to return our mocks
    original_lm_class = dspy.LM
    
    def mock_lm_factory(model: str, **kwargs: Any):
        """Factory that returns MockLM instances."""
        return MockLM(model=model, **kwargs)
    
    # Patch dspy.LM to return our mock
    monkeypatch.setattr(dspy, "LM", mock_lm_factory, raising=True)
    
    # Configure dspy.settings with mock LM
    dspy.settings.configure(lm=optimizer_lm, cache=False)
    
    # Disable caching
    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass
    
    # Wrap dspy.Predict to record calls
    original_predict_class = dspy.Predict
    
    class RecordingPredict(original_predict_class):
        """Wrapper around real dspy.Predict that records calls."""
        
        def forward(self, **kwargs):
            # Record the call
            instructions = ""
            if hasattr(self, 'signature'):
                sig = self.signature
                if hasattr(sig, 'instructions'):
                    instructions = sig.instructions or ""
                elif hasattr(sig, '_instructions'):
                    instructions = sig._instructions or ""
            
            Recorder.calls.append(PredictCall(
                instructions=instructions,
                kwargs=dict(kwargs)
            ))
            
            # Call the real Predict
            return super().forward(**kwargs)
    
    monkeypatch.setattr(dspy, "Predict", RecordingPredict, raising=True)


# Import List for type hints
from typing import List
