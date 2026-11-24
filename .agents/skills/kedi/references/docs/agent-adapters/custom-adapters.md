# Custom Adapters

A custom adapter connects Kedi to a model framework or agent harness that Kedi
does not ship. Kedi renders every model-visible part of a call: the prompt, its
instructions, the output contract, and the tool definitions. The adapter
transports that request to its backend and returns the result. It does not
write prompts of its own.

## One Call End to End

For each model call Kedi:

1. resolves the active profile and the call's tools;
2. renders the prompt and the instruction text;
3. enters the adapter's profile and tool scopes and registers the tools;
4. calls `invoke_sync` or `produce_sync` with the rendered request;
5. validates and converts the returned value to the requested Kedi type;
6. commits the turn to the conversation and records usage.

Kedi drives the synchronous entry points (`invoke_sync`, `produce_sync`) for
program statements. The async entry points (`invoke`, `produce`) are used by
asynchronous callers and subagent execution; implement both.

## The Adapter Contract

```python
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from kedi.agent_adapter import AdapterCapabilities, AgentAdapterKind, ToolSpec


class MyAdapter:
    kind: AgentAdapterKind = "agent-framework"
    shortname = "my-adapter"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_artifacts=True,
    )

    def __init__(self) -> None:
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"my_adapter_tools_{id(self)}", default=()
        )

    def invoke_sync(
        self, *, prompt: str, instructions: str | None = None, **kwargs: Any
    ) -> str: ...

    async def invoke(
        self, *, prompt: str, instructions: str | None = None, **kwargs: Any
    ) -> str: ...

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: Any = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def produce(self, **kwargs: Any) -> Any: ...

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(t for t in tools if isinstance(t, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))
```

Accept unknown keyword arguments with `**kwargs`. Kedi adds keywords over
time, and an adapter must not fail on one it does not use.

## What Each Call Receives

| Kedi statement | Entry point | Arguments |
|---|---|---|
| `[x] << ...` (untyped raw invoke) | `invoke_sync` | `prompt`, `instructions` |
| `[x: T] << ...` (typed raw invoke) | `produce_sync` | `template`, `output_type=T`, `instructions` |
| `>> ... [x] ... [y: T] ...` (template capture) | `produce_sync` | `template`, `output_schema`, `instructions` |
| `> if: <claim>` and other claims | `produce_sync` | `template`, `output_type` (boolean), `instructions` |

- `prompt` and `template` are the **fully rendered prompt**, not the Kedi
  template. For a template capture it already contains Kedi's field list and
  the template text. Send it as the user message unchanged; do not parse or
  re-render it.
- `instructions` is the complete instruction text Kedi built for the call:
  profile `> instructions:`, Kedi's template rules (`<TemplateInstructions>`),
  claim rules, capability, skill, and artifact instructions. It is `None` only
  when there are none. Send it as the system or developer message unchanged.
- `produce_sync` receives exactly one of `output_schema` and `output_type`:
  - `output_schema` maps each output name of a template capture to its
    resolved Kedi type;
  - `output_type` is a Python type for typed raw invokes and claims.

## The Prompt Contract

Kedi owns everything the model reads. An adapter must:

- send `instructions` and the prompt verbatim, each exactly once;
- not add its own template-filling, field-extraction, output-format, or
  tool-usage instructions;
- not rebuild the prompt from the template or append a second copy of the
  instructions.

A `user_prompt_submit` hook edit is already applied to the prompt the adapter
receives, so transport, telemetry, budgeting, cache identity, and history all
see the same text.

When the backend accepts a JSON Schema for structured output, build it with
Kedi's output model so that the descriptions Kedi adds reach the model:

```python
from pydantic import TypeAdapter
from kedi.agent_adapter.output_schema import build_output_model

if output_schema is not None:
    schema = build_output_model(output_schema).model_json_schema()
else:
    schema = TypeAdapter(output_type).json_schema()
```

Advertise `supports_structured_output=True` only when typed nested fields
work, not when the adapter merely asks for JSON in a prompt.

## Return Values

| Call | Return |
|---|---|
| `invoke_sync` / `invoke` | the response text as `str` |
| `produce_sync` with `output_schema` | a mapping (or object) with a value for every output name |
| `produce_sync` with `output_type` | the value itself |

Kedi validates every returned value against the requested type and converts it
to the local Kedi/Python type, so returning the decoded JSON object is enough.
A missing output name fails the call. Do not return provider message objects.
Raw capture stores the returned string.

## Tools

Implement `tool_scope` and `register_tool` (the `SupportsToolRegistration`
protocol). For each call Kedi enters `tool_scope()` and then registers every
tool of the call with `register_tool`, including Kedi's artifact management
tools. Scopes must be concurrency-safe and restore the previous registrations
on exit, as the `ContextVar` above does; never keep active tools in one shared
mutable list.

Each `ToolSpec` carries the model-facing contract: `name`, `description`,
`json_schema` (arguments), `return_schema`, `risk`, and `metadata`. Expose these
to the backend unchanged.

To execute a tool call, call `tool.fn(**arguments)` with keyword arguments and
await the result when it is awaitable. Kedi has already wrapped `fn`, so the
call applies approval policy, required-tool tracking, artifact admission, and
telemetry. A denied call raises `ToolApprovalDenied`; report it, like any
other tool exception, to the model as a failed tool result rather than ending
the run.

Kedi enables artifacts for calls with tools. An adapter that registers every
tool Kedi passes and returns each tool result to the model unchanged supports
Kedi's artifact contract and can advertise `supports_artifacts=True`; Kedi
admits large results and supplies the retrieval tools. Without it, a call with
tools fails unless the program disables artifacts. See
[Artifact Integration](artifact-contract.md).

## Profiles, Models, and Settings

The adapter sees the active profile only through `SupportsProfileOverride`:

```python
from contextlib import AbstractContextManager
from kedi.agent_profile import AgentProfile

def default_model(self) -> str | None: ...

def profile_override(self, profile: AgentProfile) -> AbstractContextManager[None]: ...
```

Kedi enters `profile_override(profile)` around each call. Without this
protocol, a profile that sets `> model:`, `> effort:`, `> mcp:`, `> settings:`,
compaction, or CodeMode fails with `AgentProfileOverrideUnsupported`.
`> instructions:` needs no override; it arrives in `instructions`.

## Media

Prompt media does not arrive as a keyword argument. Advertise
`supports_media_input=True` only when the adapter reads the attachments from
the active request
(`kedi.agent_adapter.request.current_agent_run_request().prompt.media`) and
places each one, with its `name:` label, before the prompt text in the same
user message.

## Framework and Harness Kinds

Set `kind` to:

- `"agent-framework"` for `adapter=`;
- `"agent-harness"` for `agent=`.

Kedi rejects instances passed through the wrong selection parameter.

## Shortnames

`shortname` identifies diagnostics, cache identity, telemetry, and approval
requests. It must be a stable string. Passing an instance directly needs no
registry entry:

```python
from kedi.lang import compile_program, parse_program

runtime = compile_program(parse_program(source), adapter=MyAdapter())
```

## MCP Support

MCP servers arrive through the active `AgentProfile`. Map only transports the
backend truly supports and reject unsupported transports explicitly. Do not
advertise MCP because the underlying SDK happens to have an unrelated global
MCP configuration.

## Capability Metadata

`AdapterCapabilities` records structured output, tools, MCP, profile/model/
effort/settings overrides, codemode, native approvals, subagents, background
subagents, artifacts, media input, history, hooks, accepted setting names, and
JSON Schema string formats.

`None` means unknown; `False` means unsupported. LSP diagnostics and runtime
guards depend on truthful values.

## Parallel Thread Safety

Parallel Kedi execution may call `produce_sync` concurrently. Keep per-run
state in `ContextVar` or explicit context managers, protect process clients and
writes, and never store active tools in one unscoped mutable list.

Calls that share one Kedi conversation are sequenced as complete transactions.
An adapter's conversation scope receives its native resume state before model
execution and writes the next native state only after a successful run. Do not
commit continuation state, tool results, or cleanup ownership in a detached
background callback after the adapter call has returned.

Provider-native tool calls and results remain native messages; do not
serialize them into user-prompt text to emulate history.

For stateful adapters:

- advertise `stateful_history` only when successful continuation can be resumed;
- advertise `history_replay` only when Kedi can preserve the complete causal
  message sequence, including tool calls and results;
- keep mutable provider sessions and live clients in the conversation execution
  context, never in a semantic request snapshot;
- leave `native_artifacts` false unless adapter-native and MCP tool results are
  admitted before they enter model-visible history;
- preserve cancellation and early-close rollback for Kedi-local state even when
  the remote provider cannot roll back its own session side effects;
- keep cache prefixes ordered and rotate the cache epoch only after an accepted
  compaction checkpoint.

## Error Translation

Raise clear exceptions for provider protocol errors, invalid structured output,
unsupported profile fields, timeouts, and disconnects. Preserve original
exceptions as causes when useful. Never convert a failed structured response
to an empty value or silently rerun through raw text.
