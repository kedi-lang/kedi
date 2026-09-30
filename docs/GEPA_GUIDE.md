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

Optimize spans may contain multiple template lines (for example, a header line and a line with output declarations). Output overrides and optimized prompts must be applied to **every** template line in the span, not only the first.

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
│    e. Apply instructions to span template lines                 │
│    f. Call Kedi metric function                                 │
│    g. Kedi metric calls procedure (uses overridden outputs)     │
│    h. Return (score, feedback) to GEPA                          │
│    i. Restore original state                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Post-Optimization                                            │
│    - Extract optimized prompt from best module                  │
│    - Inject prompt into ALL span template lines                  │
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

**Critical Code Section** (lines 253-262):
```python
# Extract optimized prompt for this span
print(f"\n🔍 Extracting optimized prompt...")
extracted = self.prompt_extractor.extract_prompts(optimized_module, [span])
optimized_prompt = extracted.get(span.name)
if optimized_prompt:
    optimized_prompts[span.name] = optimized_prompt
    if span.template_lines:
        # Set optimized prompt on ALL template lines in the span
        # This is necessary because a span can have multiple template lines
        for line in span.template_lines:
            line.optimized_prompt_prefix = optimized_prompt
```

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
- `_restore_output_override()`: Restores original state after metric evaluation

**Critical Code Section** (lines 102-123):
```python
try:
    # Extract outputs from DSPy prediction and inject them into the span
    # This ensures the Kedi procedure uses the DSPy-generated outputs
    # Only apply output override if we have a span (not just output_fields for backward compat)
    if span and prediction_obj:
        output_override = self._extract_prediction_outputs(prediction_obj, span, output_fields)
        if output_override:
            output_override_handle = self._apply_output_override(span, output_override)
    
    if module_ref:
        optimized_prompt = self._extract_current_instructions(module_ref)
        prompt_handle = self._apply_prompt_prefix(span, optimized_prompt)
    
    score, feedback = kedi_metric(input_val, expected_output)
    score_float = float(score)
    feedback_text = self._stringify_feedback(feedback)
    payload = ScoreWithFeedback(score=score_float, feedback=feedback_text or "")
    self._store_cached_feedback(prediction_obj, score_float, feedback_text)
    self._store_cached_feedback(pred, score_float, feedback_text)
    return payload
finally:
    self._restore_prompt_prefix(prompt_handle)
    self._restore_output_override(output_override_handle)
```

---

### 3. StatementCompiler (`src/kedi/lang/compiler/statement_compiler.py`)

**Responsibilities**:
- Compiles Kedi template lines into executable Python functions
- Checks for output overrides before invoking LLM
- Applies optimized prompt prefixes to template segments

**Key Method**: `compile_template_line()`

**Critical Code Section** (lines 45-67):
```python
def _fn(**kwargs: Any) -> Any:
    # If an override is provided, short-circuit LLM invocation and inject values.
    if stmt.optimized_output_override is not None:
        env = self.rt._current_env.get()
        if env is None:
            raise RuntimeError("Template override requires active runtime environment")
        for key, value in stmt.optimized_output_override.items():
            env[key] = value
            kwargs[key] = value
        return stmt.optimized_output_override

    # If there's an optimized prompt prefix, prepend it to the segments
    # This allows GEPA to output plain text while preserving type annotations
    segments_to_use = stmt.segments
    if stmt.optimized_prompt_prefix:
        from ..ast import TextSegment
        # Prepend optimized text, then original seed prompt with types
        # Use two newlines for better separation between optimized and seed prompts
        optimized_text = TextSegment(text=stmt.optimized_prompt_prefix + "\n\n")
        segments_to_use = [optimized_text] + stmt.segments
    
    expr, _ = prepare_segments(self.rt, segments_to_use, kwargs)
    return self.rt.m(expr)
```

---

## Debugging Strategies

### 1. Enable Debug Logging for Output Override

**Location**: `src/kedi/optimization/services/gepa_metric_wrapper.py`

Add this code after line 107 (in `_apply_output_override`):

```python
def _apply_output_override(self, span: Any, outputs: dict[str, Any]) -> Any:
    """Apply output override to ALL template lines in the span."""
    if not span or not outputs:
        return None
    if not hasattr(span, 'template_lines') or not span.template_lines:
        return None
    
    # Store original values for all lines and set override on all of them
    originals = []
    for line in span.template_lines:
        originals.append((line, line.optimized_output_override))
        line.optimized_output_override = outputs
    
    # DEBUG: Verify the override is set
    import os
    if os.getenv("KEDI_DEBUG_GEPA"):
        print(f"    [OUTPUT OVERRIDE] Set on {len(span.template_lines)} line(s): {outputs}")
        for i, line in enumerate(span.template_lines):
            print(f"      Line {i} (id={id(line)}): {line.optimized_output_override}")
    
    return originals
```

**Usage**:
```bash
export KEDI_DEBUG_GEPA=1
kedi examples/your_program.kedi --optimize --optimizer gepa
```

**Expected Output**:
```
[OUTPUT OVERRIDE] Set on 2 line(s): {'v': 25, 'w': 25.0, 'tags': '25'}
  Line 0 (id=4727861136): {'v': 25, 'w': 25.0, 'tags': '25'}
  Line 1 (id=4729265424): {'v': 25, 'w': 25.0, 'tags': '25'}
```

---

### 2. Trace Template Line Execution

**Location**: `src/kedi/lang/compiler/statement_compiler.py`

Add this code at line 46 (beginning of `_fn`):

```python
def _fn(**kwargs: Any) -> Any:
    # DEBUG: Log override check
    import os
    if os.getenv("KEDI_DEBUG_GEPA"):
        print(f"    [TEMPLATE EXEC] Line {id(stmt)}, override={stmt.optimized_output_override}")
    
    # If an override is provided, short-circuit LLM invocation and inject values.
    if stmt.optimized_output_override is not None:
        import os
        if os.getenv("KEDI_DEBUG_GEPA"):
            print(f"    [TEMPLATE EXEC] Using override: {stmt.optimized_output_override}")
        env = self.rt._current_env.get()
        # ... rest of the code
```

**Expected Output**:
```
[TEMPLATE EXEC] Line 4727861136, override={'v': 25, 'w': 25.0, 'tags': '25'}
[TEMPLATE EXEC] Using override: {'v': 25, 'w': 25.0, 'tags': '25'}
[TEMPLATE EXEC] Line 4729265424, override={'v': 25, 'w': 25.0, 'tags': '25'}
[TEMPLATE EXEC] Using override: {'v': 25, 'w': 25.0, 'tags': '25'}
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
base_lm = dspy.LM(model="openai/gpt-4o")
debug_lm = DebugLM(base_lm)
dspy.settings.configure(lm=debug_lm)
```

---

### 4. Verify Metric Evaluation Flow

**Location**: `src/kedi/optimization/services/gepa_metric_wrapper.py`

Add comprehensive logging to the metric wrapper:

```python
def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
    **context: Any,
):
    import os
    debug = os.getenv("KEDI_DEBUG_GEPA")
    
    try:
        # Extract inputs and outputs
        input_val = self._extract_input(gold, proc_params)
        expected_output = getattr(gold, "_expected", None)
        
        if debug:
            print(f"\n[GEPA METRIC] Evaluating:")
            print(f"  Input: {input_val}")
            print(f"  Expected: {expected_output}")
        
        # Get prediction object
        example_obj = gold
        prediction_obj = pred
        if hasattr(pred, "_completions") and pred._completions:
            prediction_obj = pred._completions[0]
        
        prompt_handle = None
        output_override_handle = None
        try:
            # Extract outputs from DSPy prediction
            if span and prediction_obj:
                output_override = self._extract_prediction_outputs(prediction_obj, span, output_fields)
                if debug:
                    print(f"  Extracted outputs: {output_override}")
                if output_override:
                    output_override_handle = self._apply_output_override(span, output_override)
                    if debug:
                        print(f"  Applied output override")
            
            # Apply optimized prompt
            if module_ref:
                optimized_prompt = self._extract_current_instructions(module_ref)
                if debug:
                    print(f"  Optimized prompt length: {len(optimized_prompt) if optimized_prompt else 0}")
                prompt_handle = self._apply_prompt_prefix(span, optimized_prompt)
            
            # Call Kedi metric
            score, feedback = kedi_metric(input_val, expected_output)
            if debug:
                print(f"  Score: {score}, Feedback: {feedback}")
            
            score_float = float(score)
            feedback_text = self._stringify_feedback(feedback)
            payload = ScoreWithFeedback(score=score_float, feedback=feedback_text or "")
            return payload
        finally:
            self._restore_prompt_prefix(prompt_handle)
            self._restore_output_override(output_override_handle)
    except Exception as e:
        if debug:
            import traceback
            print(f"[GEPA METRIC ERROR] {traceback.format_exc()}")
        raise
```

---

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
    optimized_prompt_prefix: Generate numeric outputs using RANGE[0.000,50.000].
    optimized_output_override: {'v': 25, 'w': 25.0, 'tags': '25'}
```

---

### 6. Track GEPA Iteration Progress

**Location**: `src/kedi/optimization/gepa.py`

Add progress tracking after GEPA compilation:

```python
# After line 247 (after optimizer.compile)
optimized_module = optimizer.compile(
    student=dspy_module,
    trainset=train_examples,
    valset=val_examples,
)

# DEBUG: Track GEPA progress
import os
if os.getenv("KEDI_DEBUG_GEPA"):
    print(f"\n[GEPA COMPLETE]")
    print(f"  Module type: {type(optimized_module)}")
    print(f"  Has predictor: {hasattr(optimized_module, 'predictor')}")
    if hasattr(optimized_module, 'predictor'):
        print(f"  Predictor type: {type(optimized_module.predictor)}")
        if hasattr(optimized_module.predictor, 'signature'):
            sig = optimized_module.predictor.signature
            print(f"  Signature instructions length: {len(sig.instructions) if hasattr(sig, 'instructions') else 'N/A'}")
```

---

## Common Issues and Solutions

### Issue 1: Score Not Improving Despite Prompt Changes

**Symptoms**:
- GEPA logs show "Proposed new text for predictor: ..."
- But "New subsample score is not better, skipping"
- Final training score matches initial score

**Diagnosis**:
1. Enable debug logging: `export KEDI_DEBUG_GEPA=1`
2. Check if output override is applied to ALL template lines
3. Verify that template line execution uses the override

**Solution**: Verify output override and optimized prompt are applied to every template line in the span.

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

**Solution**: Verify that the Kedi parser correctly populates `template_lines` for optimize spans. Check `program_parser.py` lines 409-420.

---

### Issue 3: Multiple Template Lines Not Recognized

**Symptoms**:
- Only first line gets optimized prompt
- Second line still uses original prompt

**Diagnosis**:
```python
# Check span structure
print(f"Span '{span.name}' has {len(span.template_lines)} template lines")
for i, line in enumerate(span.template_lines):
    print(f"  Line {i}: {len(line.segments)} segments")
```

**Solution**: Confirm `gepa.py` sets `optimized_prompt_prefix` on every entry in `span.template_lines`.

---

### Issue 4: DSPy Prediction Missing Expected Fields

**Symptoms**:
- `AttributeError` when accessing prediction fields
- Output override extraction returns empty dict

**Diagnosis**:
```python
# In _extract_prediction_outputs
print(f"Prediction type: {type(prediction)}")
print(f"Prediction attributes: {dir(prediction)}")
print(f"Expected fields: {[name for name, _ in output_fields]}")
for field_name, _ in output_fields:
    has_field = hasattr(prediction, field_name)
    value = getattr(prediction, field_name, None) if has_field else None
    print(f"  Field '{field_name}': has={has_field}, value={value}")
```

**Solution**: Ensure DSPy module signature matches span output fields. Check `dspy_module_builder.py` lines 70-81.

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

**Solution**: Verify that `DSPyExampleFormatter` correctly extracts span-level inputs. Check `dspy_example_formatter.py`.

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

Only update changed template lines instead of all lines:

```python
def apply_prompt_diff(span, old_prompt, new_prompt):
    if old_prompt == new_prompt:
        return  # No change
    for line in span.template_lines:
        line.optimized_prompt_prefix = new_prompt
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
  - [ ] Prompt applied to ALL template lines (not just first)
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
2. Enable debug logging (`KEDI_DEBUG_GEPA=1`)
3. Review test cases in `tests/test_gepa_kedi_mocks_*.py`
4. Examine DSPy GEPA logs for iteration details

**Key Files to Review**:
- `src/kedi/optimization/gepa.py` - Main optimizer
- `src/kedi/optimization/services/gepa_metric_wrapper.py` - Metric bridge
- `src/kedi/lang/compiler/statement_compiler.py` - Template execution
- `tests/dspy_mocks.py` - Mock infrastructure for testing
