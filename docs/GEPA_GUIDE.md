# GEPA Optimizer Debugging Guide

> Status: debugging/reference note. This document explains GEPA integration
> behavior and diagnostic patterns; it is not the canonical language guide.
> For current Kedi syntax, start with `docs/KEDI_DSL.md`.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Key Components](#key-components)
4. [Debugging Strategies](#debugging-strategies)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Testing Infrastructure](#testing-infrastructure)

---

## Overview

This document covers debugging the Kedi GEPA (Generative Prompt Optimization with Adaptive Prompting) optimizer integration with DSPy.

### Multi-Line Template Spans

Optimize spans may contain multiple template lines (for example, a header line and a line with output declarations). The two mechanisms treat those lines differently, and the difference matters when debugging:

- The **optimized prompt prefix is applied once per span, to the first template line only** (`span.template_lines[0].optimized_prompt_prefix`). Continuation lines keep their seed text; they do not each receive a copy of the optimized prompt. `span_to_instruction` prepends the first line's prefix when the span's effective instruction is built.
- The **output override is applied to every template line in the span** (`_apply_output_override` sets `optimized_output_override` on each line), so every line short-circuits to the DSPy-predicted outputs during metric evaluation.

---

## Architecture Overview

### Data Flow During GEPA Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. GEPA Initialization                                          │
│    - Build DSPy module from Kedi span                           │
│    - Create metric wrapper                                      │
│    - Configure reflection LM                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. GEPA Optimization Loop (DSPy Internal)                       │
│    - Evaluate current program on training data                  │
│    - Generate predictions with current prompt                   │
│    - Call metric wrapper with (Example, Prediction)             │
│    - Collect feedback from metric                               │
│    - Reflection LM proposes new prompt based on feedback        │
│    - Update module's signature with new prompt                  │
│    - Repeat until budget exhausted                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Metric Evaluation (Critical Path)                            │
│    a. Extract inputs from DSPy Example                          │
│    b. Extract outputs from DSPy Prediction                      │
│    c. Inject outputs into ALL span template lines               │
│    d. Extract current instructions from module                  │
│    e. Apply instructions to the span's FIRST template line only │
│    f. Call Kedi metric function                                 │
│    g. Kedi metric calls procedure (uses overridden outputs)     │
│    h. Return (score, feedback) to GEPA                          │
│    i. Restore original state                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Post-Optimization                                            │
│    - Extract optimized prompt from best module                  │
│    - Inject prompt into the span's FIRST template line only     │
│    - Calculate final training score                             │
│    - Save optimized prompts                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. GEPAOptimizer (`src/kedi/optimization/gepa.py`)

**Responsibilities**:
- Orchestrates the entire GEPA optimization process
- Configures DSPy settings and LMs
- Builds DSPy modules from Kedi spans
- Wraps Kedi metrics for DSPy compatibility
- Extracts and saves optimized prompts

**Key Methods**:
- `optimize_procedure()`: Main entry point for optimization
- `_save_prompts_immediately()`: Saves optimized prompts to disk

Per-span optimization runs in `GEPASession`
(`src/kedi/optimization/gepa_session.py`), which extracts the optimized
prompt after `optimizer.compile(...)` and stores it once per span:

```python
extracted = self._prompt_extractor.extract_prompts(optimized_module, [span])
raw_prompt = extracted.get(span.name)
# ...
optimized_prompt = normalize_optimized_prompt_prefix(span, raw_prompt)
optimized_prompts[span.name] = optimized_prompt
if span.template_lines:
    span.template_lines[0].optimized_prompt_prefix = optimized_prompt or None
```

Only the first template line receives the prefix. `GEPAOptimizer` also
snapshots that first line's prefix before optimization and restores it
afterwards, so a failed run leaves the span unchanged. The output override,
by contrast, is set on every template line (see section 2).

---

### 2. GEPAMetricWrapper (`src/kedi/optimization/services/gepa_metric_wrapper.py`)

**Responsibilities**:
- Bridges DSPy's metric interface with Kedi's metric interface
- Extracts inputs from DSPy Examples
- Extracts outputs from DSPy Predictions
- Injects outputs into span template lines (output override)
- Applies optimized prompts to span template lines
- Converts Kedi metric results to DSPy's ScoreWithFeedback format

**Key Methods**:
- `wrap_metric()`: Creates a DSPy-compatible metric from a Kedi metric
- `_extract_prediction_outputs()`: Extracts output values from DSPy predictions
- `_apply_output_override()`: Injects outputs into ALL template lines
- `_apply_prompt_prefix()`: Injects instructions into the FIRST template line only
- `_restore_output_override()`: Restores original state after metric evaluation

The metric callback (`gepa_metric` inside `wrap_metric`) applies the two
mechanisms asymmetrically, under a lock so concurrent DSPy evaluator threads
cannot leak one callback's mutations into another:

```python
if span and prediction_obj:
    output_override = self._extract_prediction_outputs(prediction_obj, span)
    if output_override:
        output_override_handle = self._apply_output_override(span, output_override)

if module_ref:
    optimized_prompt = self._extract_current_instructions(module_ref)
    prompt_handle = self._apply_prompt_prefix(span, optimized_prompt)

score, feedback = kedi_metric(input_val, expected_output)
# ... build ScoreWithFeedback, cache it ...
# finally:
#     self._restore_prompt_prefix(prompt_handle)
#     self._restore_output_override(output_override_handle)
```

`_apply_output_override` loops over every line in `span.template_lines`;
`_apply_prompt_prefix` sets `optimized_prompt_prefix` on
`span.template_lines[0]` only. Do not expect the optimized prompt on
continuation lines when inspecting span state.

---

### 3. StatementCompiler (`src/kedi/lang/compiler/statement_compiler.py`)

**Responsibilities**:
- Compiles Kedi template lines into executable Python functions
- Checks for output overrides before invoking LLM
- Applies optimized prompt prefixes to template segments

**Key Method**: `compile_template_line()`

The compiled function (`_fn`) checks the line's own fields at call time:

```python
def _fn(**kwargs: Any) -> Any:
    # If an override is provided, short-circuit LLM invocation and inject values.
    if stmt.optimized_output_override is not None:
        # Validate each overridden value against its output contract,
        # declare it in the runtime environment, and return the override
        # without calling the model.
        ...

    # If there's an optimized prompt prefix, prepend it to the segments.
    # This allows GEPA to output plain text while preserving type annotations.
    segments_to_use = stmt.segments
    if stmt.optimized_prompt_prefix:
        from ..ast import TextSegment

        optimized_text = TextSegment(text=stmt.optimized_prompt_prefix + "\n\n")
        segments_to_use = [optimized_text] + stmt.segments

    expr, _ = prepare_segments(self.rt, segments_to_use, kwargs)
    return self.rt.m(expr)
```

Because the prefix lives only on the span's first template line, only that
line's compiled function prepends it. The override check runs on every line,
which is why the override must be set on all of them. A template block
(`compile_template_block`) merges its lines into one line before compiling,
so block-level execution is a single model call.

---

## Debugging Strategies

### 1. Enable the Built-in Optimized-Prompt Debug Flag

There is no `KEDI_DEBUG_GEPA` flag in the shipped code. The built-in flag
is `KEDI_DEBUG_OPTIMIZED`, read by the CLI entrypoint
(`src/kedi/app.py`, `run_program_cli`) and passed to
`load_program_with_optimized_prompts`
(`src/kedi/optimization/services/program_loader.py`). It reports optimized
prompt loading and injection, not per-line override tracing.

**Usage**:
```bash
export KEDI_DEBUG_OPTIMIZED=1
kedi examples/your_program.kedi --optimize --optimizer gepa
```

**Expected Output** (stderr, when a `.optimized.json` file exists):
```
[DEBUG] Checking for optimized prompts: True
[DEBUG] Optimized path: <path to the .optimized.json file>
[DEBUG] BEFORE injection - <proc>::<span>: <first 80 chars of the seed instruction>...
[DEBUG] AFTER injection - <proc>::<span>: <first 80 chars of the injected instruction>...
```

During optimization itself, `GEPASession` reports extraction through the
normal reporter, for example `🔍 Extracting optimized prompt...`,
`Previous seed length: N chars`, `Optimized prefix length: N chars`, and
`Effective prompt length: N chars`. If no prompt is found it reports
`❌ No optimized prompt found`. Do not leave the flag enabled in normal
production logs: the output may reveal prompt content.

To trace the output override itself, add temporary prints to
`_apply_output_override` in
`src/kedi/optimization/services/gepa_metric_wrapper.py` (which sets
`optimized_output_override` on every line in `span.template_lines`) and
remove them afterwards. There is no built-in flag for that trace; a line
whose override stays `None` while others are set means the override was not
applied to all template lines.

---

### 2. Trace Template Line Execution

**Location**: `src/kedi/lang/compiler/statement_compiler.py`

Temporarily add a print at the start of `_fn` inside
`compile_template_line` (and remove it afterwards — there is no built-in
flag for this trace):

```python
def _fn(**kwargs: Any) -> Any:
    # DEBUG (temporary): log the override check for this line.
    print(f"    [TEMPLATE EXEC] Line {id(stmt)}, override={stmt.optimized_output_override}")
    # ... rest of the function
```

**Expected Output** (with a two-line span whose outputs were overridden):
```
[TEMPLATE EXEC] Line 4727861136, override={'v': 25, 'w': 25.0, 'tags': '25'}
[TEMPLATE EXEC] Line 4729265424, override={'v': 25, 'w': 25.0, 'tags': '25'}
```

**Red Flag**: If you see:
```
[TEMPLATE EXEC] Line 4729265424, override=None
```
This means the output override was not applied to all template lines!

---

### 3. Monitor GEPA Prompt Evolution

**Location**: `tests/dspy_mocks.py` (for testing) or create a custom LM wrapper

Add logging to track prompt changes:

```python
class DebugLM(dspy.BaseLM):
    def __init__(self, base_lm):
        super().__init__(model=base_lm.model)
        self.base_lm = base_lm
        self.call_count = 0
        
    def forward(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1
        
        # Extract prompt text
        if messages:
            if isinstance(messages, list):
                prompt_text = "\n".join([
                    msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    for msg in messages
                ])
            else:
                prompt_text = str(messages)
        else:
            prompt_text = str(prompt) if prompt else ""
        
        # Log RANGE tokens (for mock testing)
        import re
        ranges = re.findall(r'RANGE\[([^\]]+)\]', prompt_text)
        if ranges:
            print(f"[LM Call #{self.call_count}] RANGE found: {ranges}")
        else:
            print(f"[LM Call #{self.call_count}] No RANGE in prompt")
        
        return self.base_lm.forward(prompt=prompt, messages=messages, **kwargs)
```

**Usage**:
```python
# In your optimization code
base_lm = dspy.LM(model="openai/gpt-6-luna")
debug_lm = DebugLM(base_lm)
dspy.settings.configure(lm=debug_lm)
```

---

### 4. Verify Metric Evaluation Flow

**Location**: `src/kedi/optimization/services/gepa_metric_wrapper.py`

The metric callback is `gepa_metric`, created by `GEPAMetricWrapper.wrap_metric`.
Its current flow is:

1. It takes the module inputs/outputs from the callback `context`
   (`module_inputs` / `module_outputs`) when present, falling back to the
   `gold` example and `pred` prediction.
2. It returns a cached `ScoreWithFeedback` when the same prediction was
   already scored.
3. Otherwise, under the wrapper's span lock, it extracts the prediction's
   outputs and applies the output override to **every** template line, then
   applies the current optimized instructions as a prompt prefix to the
   **first** template line only.
4. It calls the Kedi metric, converts the result to `ScoreWithFeedback`,
   caches it, and restores the prefix and override in a `finally` block.
   Any exception becomes a `ScoreWithFeedback` of `0.0` with the traceback
   as feedback, so one failing metric call does not abort the run.

The reporter emits one line per callback, for example
`[metric callback 3] starting...` followed by
`[metric callback 3] score=0.800  avg=0.650  best=0.800  (12s)`.
A callback that never reaches the metric usually means the override or the
prefix step raised before the call; check the restored span state first.

### 5. Inspect Span Structure

Add this helper function to inspect span structure during debugging:

```python
def debug_span_structure(span):
    """Print detailed information about a span's structure."""
    print(f"\n=== SPAN STRUCTURE: {span.name} ===")
    print(f"Template lines: {len(span.template_lines)}")
    for i, line in enumerate(span.template_lines):
        print(f"\n  Line {i} (id={id(line)}):")
        print(f"    Segments: {len(line.segments)}")
        for j, seg in enumerate(line.segments):
            seg_type = type(seg).__name__
            if hasattr(seg, 'text'):
                print(f"      Segment {j}: {seg_type} = '{seg.text}'")
            elif hasattr(seg, 'name'):
                print(f"      Segment {j}: {seg_type} = <{seg.name}>")
        print(f"    optimized_prompt_prefix: {line.optimized_prompt_prefix}")
        print(f"    optimized_output_override: {line.optimized_output_override}")
```

**Usage**:
```python
# In gepa.py, after building the module
debug_span_structure(span)
```

**Expected Output**:
```
=== SPAN STRUCTURE: typed_span ===
Template lines: 2

  Line 0 (id=4727861136):
    Segments: 1
      Segment 0: TextSegment = 'Provide values:'
    optimized_prompt_prefix: Generate numeric outputs using RANGE[0.000,50.000].
    optimized_output_override: {'v': 25, 'w': 25.0, 'tags': '25'}

  Line 1 (id=4729265424):
    Segments: 7
      Segment 0: TextSegment = 'v='
      Segment 1: OutputSegment = <v>
      Segment 2: TextSegment = ', w='
      Segment 3: OutputSegment = <w>
      Segment 4: TextSegment = ', labels='
      Segment 5: OutputSegment = <tags>
    optimized_prompt_prefix: None
    optimized_output_override: {'v': 25, 'w': 25.0, 'tags': '25'}
```

Note the asymmetry: the output override is set on both lines, but the
optimized prompt prefix lives on the first line only. That is the intended
once-per-span behavior, not a missing application.

---

### 6. Track GEPA Iteration Progress

**Location**: `src/kedi/optimization/gepa_session.py`

`GEPASession` drives one span at a time: it calls `optimizer.compile(...)`
for the span's DSPy module, extracts the optimized prompt, and reports
progress through the reporter. There is no separate completion hook to add;
watch the reporter's per-span output instead, for example the
`🔍 Extracting optimized prompt...` line followed by the seed, optimized
prefix, and effective prompt lengths, and the per-callback
`[metric callback N] score=... avg=... best=...` lines from the metric
wrapper. A span that reports `❌ No optimized prompt found` keeps its seed
prompt and scores `0.0` for that span.

---

## Common Issues and Solutions

### Issue 1: Score Not Improving Despite Prompt Changes

**Symptoms**:
- GEPA logs show "Proposed new text for predictor: ..."
- But "New subsample score is not better, skipping"
- Final training score matches initial score

**Diagnosis**:
1. Enable the built-in debug flag: `export KEDI_DEBUG_OPTIMIZED=1`
2. Check if output override is applied to ALL template lines
3. Verify that template line execution uses the override

**Solution**: Verify the output override is applied to every template line in the span, and that the optimized prompt prefix is set on the span's first template line (once per span, not once per line).

---

### Issue 2: Template Lines Not Found

**Symptoms**:
- `AttributeError: 'OptimizeSpan' object has no attribute 'template_lines'`
- Or `template_lines` is empty

**Diagnosis**:
```python
# Add this check in gepa.py
if not span.template_lines:
    print(f"WARNING: Span '{span.name}' has no template lines!")
    print(f"Span object: {span}")
```

**Solution**: Verify that the Kedi parser correctly populates `template_lines` for optimize spans. The span is built in `src/kedi/lang/parser/validation_conversion.py` (`_walk_optimize_span`); the old `program_parser.py` module no longer exists.

---

### Issue 3: Only the First Line Carries the Optimized Prompt

**Symptoms**:
- Only the first template line has an `optimized_prompt_prefix`
- Later lines still carry only their original seed text

**Diagnosis**:
```python
# Check span structure
print(f"Span '{span.name}' has {len(span.template_lines)} template lines")
for i, line in enumerate(span.template_lines):
    print(f"  Line {i}: prefix={line.optimized_prompt_prefix!r}")
```

**Explanation**: This is the intended once-per-span behavior, not a missing
application. The optimized prompt is stored on `span.template_lines[0]` only
(see `GEPASession` in `src/kedi/optimization/gepa_session.py`), and
`span_to_instruction` (`src/kedi/optimization/utils.py`) prepends that one
prefix to the newline-joined text of all template lines when the span's
effective instruction is built. Continuation lines are still part of the
optimized instruction; they simply do not each carry a copy of the prefix.
The output override, by contrast, must be present on every line — if a later
line's `optimized_output_override` is `None`, that is a real defect.

---

### Issue 4: DSPy Prediction Missing Expected Fields

**Symptoms**:
- `AttributeError` when accessing prediction fields
- Output override extraction returns empty dict

**Diagnosis**:
```python
# In _extract_prediction_outputs (span, prediction) -> dict | None
from kedi.optimization.utils import extract_output_fields
print(f"Prediction type: {type(prediction)}")
print(f"Prediction attributes: {dir(prediction)}")
print(f"Expected fields: {[name for name, _ in extract_output_fields(span)]}")
for field_name, _ in extract_output_fields(span):
    has_field = hasattr(prediction, field_name)
    value = getattr(prediction, field_name, None) if has_field else None
    print(f"  Field '{field_name}': has={has_field}, value={value}")
```

**Solution**: Ensure DSPy module signature matches span output fields. The signature is built by `DSPyModuleBuilder.build_module` in `src/kedi/optimization/services/dspy_module_builder.py`, from the span's input and output fields — not from the procedure parameters.

---

### Issue 5: Metric Called with Wrong Inputs

**Symptoms**:
- Metric receives procedure-level parameters instead of span-level inputs
- `KeyError` or unexpected input values

**Diagnosis**:
```python
# In wrap_metric
print(f"Procedure params: {proc_params}")
print(f"Span inputs: {extract_input_fields(span)}")
print(f"Example keys: {list(gold.items())}")
```

**Solution**: Verify that `DSPyExampleFormatter` correctly extracts span-level inputs. See `src/kedi/optimization/services/dspy_example_formatter.py`.

---

## Testing Infrastructure

### Mock GEPA Testing

The test suite uses a sophisticated mocking infrastructure to test GEPA without calling real LLMs:

**Key Components**:
1. **MockLM** (`tests/dspy_mocks.py`): Simulates LLM responses based on `RANGE[min,max]` tokens
2. **RangeAwareAdapter**: Parses RANGE tokens and generates deterministic outputs
3. **RecordingPredict**: Wraps `dspy.Predict` to record all calls
4. **Gradual Metrics**: Return continuous scores based on distance to target

**Example Test Structure**:
```python
def test_gepa_optimization(monkeypatch):
    # Install mocks
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    
    # Define Kedi program with optimize span
    src = """
    @proc(seed: str) -> float:
      > optimize: span_name:
        >> Generate value: [y: int]
      = `float(y)`
    
    @eval: proc:
      > data: ds:
        = ```
        return [("test", None)]
        ```
      > metric: score(ds):
        = ```
        val = proc(ds)
        # Gradual metric
        if 5.5 <= val <= 6.5:
            return (1.0, "OK")
        elif val < 5.5:
            distance = 5.5 - val
            score = max(0.0, 1.0 - distance / 100.0)
            return (score, "TOO_LOW")
        else:
            distance = val - 6.5
            score = max(0.0, 1.0 - distance / 100.0)
            return (score, "TOO_HIGH")
        ```
    """
    
    # Parse and optimize
    prog, rt, proc, param_types, _, metric = _prepare(src)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=100,
            reflection_model="mock/reflection",
        ),
        runtime=rt,
    )
    
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=proc.optimize_spans,
        train_data=[("test", None)],
        test_data=None,
        metric=metric,
    )
    
    # Verify optimization succeeded
    assert score >= 0.8  # Should converge to target range
    assert "RANGE" in optimized["span_name"]  # Should contain optimized RANGE
```

---

### Test Categories

1. **Basic Tests** (`test_gepa_kedi_mocks_basic.py`):
   - Range parsing and sampling
   - Predict call recording
   - Basic optimization flow

2. **Integration Tests** (`test_gepa_kedi_mocks_integration.py`):
   - Reflection improves scores
   - Feedback directionality
   - End-to-end optimization

3. **Advanced Tests** (`test_gepa_kedi_mocks_advanced.py`):
   - Typed outputs (int, float, list)
   - Complex input forwarding
   - Multi-line template spans

4. **Failure Tests** (`test_gepa_kedi_mocks_failures.py`):
   - Invalid reflection output
   - Missing output fields
   - Empty training data

5. **Span Isolation Tests** (`test_gepa_kedi_mocks_span_isolation.py`):
   - Verify only span inputs forwarded
   - Multi-span independent optimization

6. **Multi-Span Tests** (`test_gepa_kedi_mocks_multi_span.py`):
   - Multiple spans optimized in one procedure
   - Per-span prompt and score isolation

---

## Performance Considerations

### 1. Output Override Cost

**Impact**: Minimal - only dictionary assignment per template line

**Optimization**: The override mechanism short-circuits LLM invocation, providing significant speedup during metric evaluation.

### 2. Prompt Application Cost

**Impact**: Minimal - only string concatenation per template line

**Optimization**: Prompt prefix is prepended to segments, avoiding full template re-parsing.

### 3. Memory Usage

**Impact**: Each template line stores:
- `optimized_prompt_prefix`: ~100-1000 bytes (typical prompt)
- `optimized_output_override`: ~100 bytes (typical output dict)

**For 100 spans with 2 lines each**: ~40-200 KB total overhead

---

## Future Improvements

### 1. Automatic Multi-Line Detection

Add validation to warn when multi-line spans are detected:

```python
def validate_span_structure(span):
    if len(span.template_lines) > 1:
        print(f"⚠️  Span '{span.name}' has {len(span.template_lines)} template lines")
        print(f"   Ensure all optimization mechanisms handle multiple lines")
```

### 2. Span-Level Caching

Cache optimized prompts and outputs per span to avoid redundant computation:

```python
class SpanCache:
    def __init__(self):
        self._cache = {}
    
    def get_optimized_prompt(self, span_name):
        return self._cache.get(f"{span_name}_prompt")
    
    def set_optimized_prompt(self, span_name, prompt):
        self._cache[f"{span_name}_prompt"] = prompt
```

### 3. Differential Prompt Updates

Skip the update when the prompt is unchanged; otherwise replace the span's
single stored prefix (the first template line only):

```python
def apply_prompt_diff(span, old_prompt, new_prompt):
    if old_prompt == new_prompt:
        return  # No change
    if span.template_lines:
        span.template_lines[0].optimized_prompt_prefix = new_prompt
```

### 4. Structured Logging

Replace print statements with structured logging:

```python
import logging

logger = logging.getLogger("kedi.optimization.gepa")

# In code
logger.debug(
    "Applied output override",
    extra={
        "span_name": span.name,
        "num_lines": len(span.template_lines),
        "outputs": outputs,
    }
)
```

---

## Appendix: Complete Debugging Checklist

When debugging GEPA optimization issues, check these in order:

- [ ] **Span Structure**
  - [ ] Span has template_lines attribute
  - [ ] template_lines is not empty
  - [ ] Count of template lines matches expectation
  - [ ] Each line has segments

- [ ] **Output Override**
  - [ ] Override applied to ALL template lines (not just first)
  - [ ] Override contains expected output fields
  - [ ] Override values match DSPy prediction
  - [ ] Override is restored after metric evaluation

- [ ] **Prompt Application**
  - [ ] Optimized prompt extracted from module
  - [ ] Prompt applied to the span's FIRST template line only (once per span)
  - [ ] Prompt is not empty or unchanged
  - [ ] Prompt contains expected optimization markers (e.g., RANGE tokens)

- [ ] **Metric Evaluation**
  - [ ] Metric receives correct inputs (span-level, not procedure-level)
  - [ ] Metric uses overridden outputs (not re-invoking LLM)
  - [ ] Metric returns (score, feedback) tuple
  - [ ] Score is numeric and in valid range

- [ ] **GEPA Integration**
  - [ ] DSPy module built correctly from span
  - [ ] Module signature matches span inputs/outputs
  - [ ] GEPA receives wrapped metric
  - [ ] Reflection LM configured if specified
  - [ ] Training/validation data formatted correctly

- [ ] **Final Verification**
  - [ ] Final training score > initial score
  - [ ] Optimized prompt saved to disk
  - [ ] Optimized prompt differs from original
  - [ ] No warnings or errors in logs

---

## Contact and Support

For issues or questions about GEPA optimization:
1. Check this documentation first
2. Enable debug logging (`KEDI_DEBUG_OPTIMIZED=1`)
3. Review test cases in `tests/test_gepa_kedi_mocks_*.py`
4. Examine DSPy GEPA logs for iteration details

**Key Files to Review**:
- `src/kedi/optimization/gepa.py` - Main optimizer
- `src/kedi/optimization/gepa_session.py` - Per-span optimization and prompt extraction
- `src/kedi/optimization/services/gepa_metric_wrapper.py` - Metric bridge
- `src/kedi/lang/compiler/statement_compiler.py` - Template execution
- `tests/dspy_mocks.py` - Mock infrastructure for testing
