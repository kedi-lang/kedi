# Models and Reasoning

Model, fallback, and reasoning directives refine the active backend. They do not
select a framework or harness by themselves.

## Select a Model

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
```

The model identifier is passed to the selected adapter. Its accepted syntax,
availability, credentials, and provider prefix therefore depend on that
adapter.

A model directive applies to following calls in the current lexical scope:

```kedi
> model: fast-model

@draft(topic: str) -> str:
  >> A draft about <topic> is [text: str].
  = <text>

> model: quality-model

@review(text: str) -> str:
  >> Review feedback for <text> is [feedback: str].
  = <feedback>
```

Each procedure captures the top-level state present when it is defined.

## Dynamic Model Expressions

````kedi
```
selected_model = "openai:gpt-6-luna"
```

> model: `selected_model`
````

The expression must return a string. Prefer literal models or profiles so the
LSP can reason about adapter capability and configuration.

## Ordered Fallback Models

Pydantic AI and LangChain support alternatives to the effective primary model:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
> fallback: openai:gpt-6.1-sol

>> The capital of Turkey is [capital: str].
= <capital>
```

Replace the inline declaration with a block for ordered alternatives:

```kedi
> fallback:
    openai:gpt-6.1-sol
    openai:gpt-6-luna
```

Entries are model IDs, not profiles or adapters. Whole backtick expressions may
return a nonempty model ID or a list/tuple of nonempty IDs, including within a
block. Values are validated before model I/O, not coerced from arbitrary objects.
The same forms work with `> adapter: langchain`; harness adapters reject them.
Fallback is caller-owned model selection, so it is not a capability member.

### Scope and Inheritance

`> fallback:` is independent of `> model:`. The primary can come from an enclosing
scope, an applied profile, Python, or the environment; a model directive need not
immediately precede it. A fallback-only profile is valid:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna

> profile: resilient:
    > fallback: openai:gpt-6.1-sol

> use: resilient

@capital() -> str:
    >> The capital of Turkey is [city: str].
    = <city>

> fallback: `[]`
= <capital()>
```

`capital` keeps the chain captured at its definition site even though the caller
clears its own list. A local declaration likewise leaves the caller unchanged.
Each declaration replaces the list, never appends; changing `> model:` replaces
only the primary. Conditional profile members and subagent profiles follow the
same rules. Omission inherits; ``> fallback: `[]` `` explicitly clears the chain.
`none` is not a reset literal. A primary must resolve before invocation: the
first backup is never an implicit primary.

### Recovery Boundaries

Each candidate first uses `request_retries` for HTTP 408, 429, and 5xx failures
(two additional attempts by default), then advances to the next eligible
candidate. `request_retries: 0` disables same-model retries, not explicit fallback.
`Retry-After` is honored; a delay beyond the bounded retry policy can skip further
requests to that candidate without preventing another configured candidate.
Authentication errors, refusals, argument/output validation failures, tool-body
errors, budget exhaustion, cancellation, and deadline expiry never trigger
fallback. Exhaustion raises a terminal failure with attempt context.

All attempts share admission, request and usage limits, and the enclosing
deadline. Selecting a backup does not restart the template, agent, or child,
consume another child-start slot, or replay completed tools and approvals.
Only stream-opening failures before any response content reaches the caller can
recover; partially delivered output is never replayed or combined across models.

Model changes preserve portable history and tool results, not incompatible
response IDs, server conversation IDs, or cache settings. History-processor
continuation resets stay in effect. Incompatible required server state or
output/tool/media/settings contracts fail instead of being silently discarded.
One invocation's selection does not change another's primary.

For Python constructor and SDK retry controls, see
[Pydantic AI](../agent-adapters/pydantic-ai.md#model-fallback-and-request-retries)
and [LangChain](../agent-adapters/langchain.md#model-fallback-and-request-retries).

## Reasoning Effort

Accepted values are:

- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`
- `max`

```kedi
> effort: high
```

A backtick expression may choose the value dynamically:

```kedi
> effort: `"high" if args.deep else "low"`
```

Not every provider implements every effort level. Pydantic maps `max` to
`xhigh`; harness adapters map or
validate according to their runtime.

## Inheritance and Overrides

An inner scope inherits the outer model and effort until it replaces either:

```kedi
> model: default-model
> effort: low

@deep_review(text: str) -> str:
  > effort: high
  >> A detailed review of <text> is [result: str].
  = <result>
```

The effort override does not discard the inherited model, tools, MCP servers,
or instructions. Applying a profile merges its specified members; omitted
members continue to inherit.

## Choosing Effort

Use low effort for extraction, formatting, classification, and routine tool
selection. Use higher effort for ambiguous planning, multi-source synthesis,
or code reasoning where additional latency and cost are justified. Do not use
effort as compensation for vague instructions or an oversized tool surface.

## Errors

Kedi reports an invalid effort value before execution. Unknown or unavailable
model identifiers, missing credentials, and unsupported model settings are
reported by the selected adapter/provider. A profile that requires a model
override fails if its adapter cannot construct one.
