from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dspy
import pytest
from dspy import BaseLM

from kedi.agent_adapter import AgentAdapter
from kedi.agent_adapter.adapters import dspy as dspy_adapter_module
from kedi.optimization import gepa_reflection as gepa_reflection_module
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

# =========
# Utilities
# =========

_RANGE_RE = re.compile(r"RANGE\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")

REFLECTION_STEP = 100.0


def parse_first_range(
    text: str, default: Tuple[float, float] = (-100.0, -50.0)
) -> Tuple[float, float]:
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
    val = int.from_bytes(h[:8], "big")
    return (val % 10_000_000) / 10_000_000.0


def sample_from_range(template: str, *, prefer_midpoint: bool = True) -> Tuple[int, float]:
    """Produce an (int_value, float_value) determined by RANGE[...] in template."""
    lo, hi = parse_first_range(template)
    if prefer_midpoint:
        mid = (lo + hi) / 2.0
        return (int(round(mid)), float(mid))
    frac = deterministic_uniform_0_1(template)
    value = lo + (hi - lo) * frac
    return (int(round(value)), float(value))


def dist_to_target(lo: float, hi: float, target_lo: float = 40.0, target_hi: float = 60.0) -> float:
    """Distance from interval [lo,hi] to target [target_lo,target_hi]."""
    if hi < target_lo:
        return target_lo - hi
    if lo > target_hi:
        return lo - target_hi
    return 0.0


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
    index: int = 0


class Recorder:
    calls: list[PredictCall] = []
    adapter_calls: list[dict[str, Any]] = []
    lm_calls: list[LMCall] = []
    _call_counter: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()
        cls.adapter_calls.clear()
        cls.lm_calls.clear()
        cls._call_counter = 0

    @classmethod
    def optimizer_calls(cls) -> list[LMCall]:
        """Return LM calls that went to the optimizer model (non-reflection)."""
        return [c for c in cls.lm_calls if "reflection" not in c.model]

    @classmethod
    def reflection_calls(cls) -> list[LMCall]:
        """Return LM calls that went to the reflection model."""
        return [c for c in cls.lm_calls if "reflection" in c.model]

    @classmethod
    def predict_ranges(cls) -> List[Tuple[float, float]]:
        """Extract RANGE[lo,hi] from each Predict call's instructions."""
        ranges: List[Tuple[float, float]] = []
        for call in cls.calls:
            lo, hi = parse_first_range(call.instructions)
            ranges.append((lo, hi))
        return ranges


# =======================
# Range-aware Test Adapter
# =======================


class RangeAwareAdapter(AgentAdapter[AttrDict[str, Any]]):
    """Adapter that generates outputs guided by RANGE[min,max] tokens present in the prompt."""

    def __init__(self, *, prefer_midpoint: bool = True):
        self.prefer_midpoint = prefer_midpoint

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **_: Any,
    ) -> AttrDict[str, Any]:
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **_: Any,
    ) -> AttrDict[str, Any]:
        ival, fval = sample_from_range(template, prefer_midpoint=self.prefer_midpoint)
        lo, hi = parse_first_range(template)

        Recorder.adapter_calls.append(
            {
                "template": template,
                "range": (lo, hi),
                "ival": ival,
                "fval": fval,
            }
        )

        data: Dict[str, Any] = {}
        if output_schema:
            for key, typ in output_schema.items():
                if typ is int:
                    data[key] = ival
                elif typ is float:
                    data[key] = fval
                elif hasattr(typ, "__origin__"):
                    origin = getattr(typ, "__origin__", None)
                    if origin in (list, List):
                        data[key] = [f"{key}_item1", f"{key}_item2"]
                    else:
                        data[key] = str(fval)
                else:
                    data[key] = str(fval)

        return AttrDict(data)

    async def invoke(self, *, prompt: str, **_: Any) -> str:
        return self.invoke_sync(prompt=prompt)

    def invoke_sync(self, *, prompt: str, **_: Any) -> str:
        return f"response:{prompt}"


# ======================
# Mocked LM Layer
# ======================

_REFLECTION_FAIL_NEXT = False


def reflection_fail_once() -> None:
    """Configure the mock reflection LM to emit invalid content on its next call."""
    global _REFLECTION_FAIL_NEXT
    _REFLECTION_FAIL_NEXT = True


class MockLM(BaseLM):
    """Mock LM that intercepts calls and returns controlled responses.

    Routing:
      - model name contains "reflection" -> _reflection_response
      - otherwise -> _optimizer_response

    Optimizer response: parses RANGE[lo,hi] from prompt, returns JSON with
    midpoint values for every detected output field.

    Reflection response: parses RANGE[lo,hi] AND feedback signals (TOO_LOW /
    TOO_HIGH / OK) from prompt. Shifts the range by REFLECTION_STEP in the
    direction indicated by the feedback and returns a new instruction string.
    """

    def __init__(self, model: str = "mock/optimizer", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.history: list[dict[str, Any]] = []

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any):
        if messages:
            if isinstance(messages, list) and messages:
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

        if "reflection" in self.model:
            response = self._reflection_response(prompt_text)
            if "`new_instruction`" in prompt_text:
                # DSPy 3.4.1 asks for the instruction through a ProposeInstruction signature,
                # formatted by the JSONAdapter Kedi configures.
                response = json.dumps({"new_instruction": response})
        else:
            response = self._optimizer_response(prompt_text)

        Recorder._call_counter += 1
        Recorder.lm_calls.append(
            LMCall(
                prompt=prompt_text,
                response=response,
                model=self.model,
                index=Recorder._call_counter,
            )
        )

        from types import SimpleNamespace

        choice = SimpleNamespace(message=SimpleNamespace(content=response))

        class UsageDict(dict):
            def __init__(self):
                super().__init__(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0

        return SimpleNamespace(
            choices=[choice],
            usage=UsageDict(),
            model=self.model,
        )

    def _reflection_response(self, prompt: str) -> str:
        """Shift RANGE based on TOO_LOW / TOO_HIGH feedback in the prompt."""
        global _REFLECTION_FAIL_NEXT

        if _REFLECTION_FAIL_NEXT:
            _REFLECTION_FAIL_NEXT = False
            return "RANGE[NaN,NaN]"

        lo, hi = parse_first_range(prompt, default=(-100.0, -50.0))
        txt = prompt or ""
        step = REFLECTION_STEP

        has_too_low = "TOO_LOW" in txt and "TOO_HIGH" not in txt
        has_too_high = "TOO_HIGH" in txt and "TOO_LOW" not in txt

        if has_too_low:
            lo += step
            hi += step
        elif has_too_high:
            lo -= step
            hi -= step

        return f"Generate numeric outputs using RANGE[{lo:.3f},{hi:.3f}]."

    def _optimizer_response(self, prompt: str) -> str:
        """Produce JSON outputs driven by RANGE[lo,hi] in the prompt."""
        import json

        lo, hi = parse_first_range(prompt, default=(-100.0, -50.0))
        ival, fval = sample_from_range(prompt, prefer_midpoint=True)

        outputs: dict[str, Any] = {}

        field_pattern = r"`([a-zA-Z_][a-zA-Z0-9_]*)`"

        # Prefer the "output fields" section, then "Respond with JSON" section,
        # finally fall back to all backtick-quoted identifiers.
        output_section_match = re.search(
            r"Your output fields are:(.*?)(?:All interactions|Respond|In adhering|Notes|$)",
            prompt,
            re.DOTALL,
        )
        respond_match = re.search(
            r"Respond with a JSON object.*?fields:\s*(.+?)\.",
            prompt,
            re.DOTALL,
        )

        if output_section_match:
            output_fields = re.findall(field_pattern, output_section_match.group(1))
        elif respond_match:
            output_fields = re.findall(field_pattern, respond_match.group(1))
        else:
            all_found = re.findall(field_pattern, prompt)
            exclude = {
                "foo",
                "bar",
                "completed",
                "input",
                "output",
                "fields",
                "order",
                "type",
                "json",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "true",
                "false",
                "null",
                "none",
            }
            output_fields = [f for f in all_found if f.lower() not in exclude]

        for fld in output_fields:
            field_context = prompt[max(0, prompt.find(fld) - 80) : prompt.find(fld) + 80].lower()

            if "int" in field_context and "list" not in field_context:
                outputs[fld] = ival
            elif "float" in field_context and "list" not in field_context:
                outputs[fld] = fval
            elif "list" in field_context or "tags" in fld:
                outputs[fld] = [f"{fld}_item1", f"{fld}_item2"]
            elif "str" in field_context:
                outputs[fld] = str(ival)
            else:
                if any(
                    tok in fld
                    for tok in [
                        "y",
                        "v",
                        "w",
                        "num",
                        "val",
                        "count",
                        "answer",
                        "result",
                        "score",
                    ]
                ):
                    outputs[fld] = ival
                else:
                    outputs[fld] = str(ival)

        if not outputs:
            outputs = {"unknown": ival}

        return json.dumps(outputs)


def install_mock_dspy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch dspy to use real Predict/Module but with mocked LM layer."""

    optimizer_lm = MockLM(model="mock/optimizer")

    def mock_lm_factory(model: str, **kwargs: Any):
        return MockLM(model=model, **kwargs)

    monkeypatch.setattr(dspy_adapter_module, "ThinkingStrippingLM", mock_lm_factory)
    monkeypatch.setattr(gepa_reflection_module, "ThinkingStrippingLM", mock_lm_factory)
    monkeypatch.setattr(dspy, "LM", mock_lm_factory, raising=True)

    dspy.settings.configure(lm=optimizer_lm, cache=False)

    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass

    original_predict_class = dspy.Predict

    class RecordingPredict(original_predict_class):
        """Wrapper around real dspy.Predict that records calls."""

        def forward(self, **kwargs):
            if getattr(self.signature, "__module__", "").startswith("dspy.teleprompt."):
                # GEPA's own reflection predictors (DSPy 3.4.1) are not program spans.
                return super().forward(**kwargs)
            instructions = ""
            if hasattr(self, "signature"):
                sig = self.signature
                instructions = (
                    getattr(sig, "instructions", None) or getattr(sig, "_instructions", None) or ""
                )

            Recorder.calls.append(
                PredictCall(
                    instructions=instructions,
                    kwargs=dict(kwargs),
                )
            )

            return super().forward(**kwargs)

    monkeypatch.setattr(dspy, "Predict", RecordingPredict, raising=True)
