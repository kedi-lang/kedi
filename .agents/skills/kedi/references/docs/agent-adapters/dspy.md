# DSPy

## Installation

DSPy is an optional integration:

```bash
uv add 'kedi[dspy]'
```

A model is required; DSPy has no model-less Kedi default.

## Language Models

```kedi
> adapter: dspy
> model: openai/gpt-6-luna
```

Model names normalize to LiteLLM form. The adapter establishes a baseline
`dspy.LM`, then uses `dspy.context` for lexical model/profile overrides.

## Reasoning Effort and Settings

Supported settings are `model_type`, `temperature`, `max_tokens`, `cache`,
`callbacks`, `num_retries`, `provider`, `finetuning_model`, `launch_kwargs`,
`train_kwargs`, `use_developer_role`, and `reasoning_effort`.

`> effort:` becomes `reasoning_effort`. Kedi's adapter defaults DSPy LM caching
to false unless explicitly configured.

## Signatures and Structured Outputs

Kedi output fields become a dynamic DSPy `Signature` with `OutputField`
entries. Active system instructions become signature instructions.

```kedi
> adapter: dspy
>> <text> identifies [name] with score [score: float].
```

Reasoning-model thinking wrappers are stripped before DSPy's structured marker
parser sees them; raw traces remain available on the internal LM for debugging.

## ReAct Tool Calls

Without tools, Kedi uses `dspy.Predict`. With scoped tools, it uses
`dspy.ReAct`. Kedi procedure/Python tools are converted to `dspy.Tool` and
wrapped with approval enforcement before child execution.

## Stdio MCP

DSPy MCP support is **stdio only**. Kedi opens each server, initializes a client
session, lists tools, converts them with `dspy.Tool.from_mcp_tool`, and runs
ReAct over local plus MCP tools. SSE or HTTP declarations raise
`NotImplementedError`.

## Blocking Subagents

DSPy supports foreground child delegation and structured child results. It
does not support background subagents or native conversation resume. Continued
turns use Kedi's bounded transcript mechanism.

## GEPA Integration

GEPA is DSPy-backed but distinct from selecting the DSPy runtime adapter.
`--optimizer gepa` configures optimizer and reflection LMs, converts optimize
spans to DSPy modules, and evaluates them through the Kedi metric bridge.

## Current Capability Limits

- MCP is stdio-only.
- Subagents are blocking-only.
- Approval is enforced on Kedi-projected tools, not through a separate native
  DSPy approval protocol.
- Provider-specific settings outside the supported key set are filtered.
