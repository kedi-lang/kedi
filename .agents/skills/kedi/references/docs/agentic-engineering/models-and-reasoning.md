# Models and Reasoning

Model and reasoning directives refine the active backend. They do not select a
framework or harness by themselves.

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
`xhigh`; DSPy receives the value as `reasoning_effort`; harness adapters map or
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
