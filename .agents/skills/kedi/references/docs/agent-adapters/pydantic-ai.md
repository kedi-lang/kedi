# Pydantic AI


## Installation

Pydantic AI is a core Kedi dependency:

```bash
uv add kedi
```

Provider credentials and optional provider packages follow Pydantic AI's model
requirements.


## Model Names

Select the adapter and a Pydantic-style or LiteLLM-style model:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
```

Strings using `vendor/model` are normalized to Pydantic's model naming form.
An existing Pydantic AI `Model` may be passed to `PydanticAdapter` directly.

CLI and Python API lazy adapter selection defer native provider-name validation
until the model is selected for a call. A program with its own `> model:` does
not require credentials for an unused default provider. This does not waive
credentials for the selected model. Direct `PydanticAdapter(...)` construction
keeps Pydantic AI's eager-validation default; pass `defer_model_check=True` to
opt in. Python API adapter options can explicitly set `defer_model_check=False`
to retain eager validation when the lazy adapter is constructed.


## Model Settings

Supported settings are `max_tokens`, `temperature`, `top_p`, `timeout`,
`parallel_tool_calls`, `tool_choice`, `seed`, `presence_penalty`,
`frequency_penalty`, `logit_bias`, `stop_sequences`, `extra_headers`,
`thinking`, `service_tier`, and `extra_body`.

```kedi
> settings:
    timeout: 120
    max_tokens: 2048
    parallel_tool_calls: true
```

`max` reasoning effort maps to Pydantic AI's `xhigh`.


## Capability Bundles

[Capabilities](../agentic-engineering/capabilities.md) compose tools, instructions,
and owned tool policies through `> use:`. Eager use needs no loading call.
`> defer: true` uses Pydantic AI's native `load_capability` path and withholds
the bundle's callable tools until activation. Activation is run-local; model
retries do not reevaluate a Kedi `defer` expression.

An instruction reset suppresses current Kedi guidance without erasing an older
load result from history. A deferred parent reveals its nested catalog only
after activation; nested deferred bundles and their subagent tools require
their own load. See the capability guide for explicit reuse, history,
MCP connection lifetime, and cache limitations.


## CodeMode

Use `> codemode: enabled` to replace application tool schemas with Kedi's
`search_tools`, `get_tool_schema`, and `execute_code` controls:

```kedi
> adapter: pydantic
> codemode: enabled
```

The outer capability wraps constructor, caller, scoped Kedi, and local MCP
toolsets together. Nested calls retain validation, approval, required-tool,
telemetry, cancellation, and artifact behavior. Provider-native MCP and
external deferred approval are rejected while CodeMode is active. See
[CodeMode](../agentic-engineering/codemode.md) for the complete contract.


## Structured Outputs

Kedi builds a dynamic Pydantic model from each output field:

```kedi
~Finding(severity: str, message: str)

>> The findings from inspecting <code> are [findings: list[Finding]].
```

Field descriptions from `Annotated[T, "description"]` are preserved. Pydantic
AI produces and validates the result before Kedi publishes captured fields.
For completion templates, the generated output schema reinforces that captures
contain replacement values rather than the completed template. This preserves
the provider's output mode and does not trim or otherwise rewrite returned
strings. Raw invokes and decision-model templates do not receive this rule.


## Python and Procedure Tools

Python `@kedi.tool` functions and Kedi procedures selected by `> use:` become
native Pydantic AI tools for one lexical run. Registration is context-local,
so tools do not leak between concurrent calls.


## MCP Toolsets

All Kedi MCP transports are mapped to Pydantic AI toolsets:

- stdio with command, args, and env;
- SSE with URL and headers;
- streamable HTTP with URL and headers.

Application and MCP toolsets are approval-required before execution.


## Native Tool Artifacts

When artifacts are enabled, constructor tools, caller-provided tools, and local
MCP toolsets cross Kedi's artifact-admission boundary before Pydantic AI commits
their successful results to message history. Large values therefore become the
same compact `ArtifactRef` objects used by Kedi-defined tools.

Admission preserves Pydantic AI's native `tool_call_id`, validation, retries,
approval flow, and streaming result handling. Failed tool results remain native
errors, and a tool already wrapped by Kedi is not admitted twice. The exact
returned ref ID must be used with `read_artifact`; IDs must not be predicted.


## Approval Integration

Pydantic's deferred-tool capability is used to resolve Kedi approval requests.
Read-only calls pass automatically; mutating and sensitive calls flow through
the active static/dynamic policy. Edited arguments are supplied as Pydantic
tool overrides after validation.

Nested subagent policies form a ceiling: a child cannot widen a parent's
restriction.


## Foreground and Background Subagents

Pydantic supports both modes and native conversation resume. Usage limits are
translated to Pydantic AI request, tool-call, and token limits. Child tool,
MCP, skills, model, and instruction scopes remain isolated.


## Usage and Retry Behavior

Pydantic run usage is reported to Kedi's subagent budget observer. The adapter
tracks requests, tool calls, and input/output/total tokens.

Unless `retries` is supplied explicitly, `PydanticAdapter` allows three
bounded retries for correctable tool-call failures. The output-validation
retry budget remains Pydantic AI's default. An explicit integer or
`AgentRetries` value overrides Kedi's tool retry default.

`subagent_failure_policy="fail_closed"` is the default. `"recover"` exposes a
sanitized child error to the parent instead of failing the parent run.


## Codex Responses Connections

See [Codex Responses Connections](codex-models.md).

On Codex WebSocket models, Kedi reveals activated capability tools through
Pydantic AI's ordinary tool list instead of native tool-addition/deferral
parameters that the gateway rejects. `load_capability` and run-local activation
still apply. The compatibility profile is scoped to the call: the caller's
model is not mutated, the WebSocket client is retained, and request budgets
remain enforced. HTTP models and other providers keep their original profiles.
Changing the visible tool list is not a prefix-cache preservation guarantee.


## Native History Policy

For application-defined retention or summarization, pass a native
`history_processor=` callback. Pydantic AI's processed history is written back
to the active graph before dispatch, including tool-loop requests. Kedi validates
tool lifecycles and continuation safety around that callback. See
[User-Defined History Processing](../runtime/history.md#user-defined-history-processing).
