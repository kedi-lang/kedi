# Custom Adapters

## The Adapter Contract

Implement `AgentAdapter[T]` from `kedi.agent_adapter`:

```python
from typing import Any
from kedi.agent_adapter import AdapterCapabilities, AgentAdapterKind


class MyAdapter:
    kind: AgentAdapterKind = "agent-framework"
    shortname = "my-adapter"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
    )

    async def produce(
        self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]: ...

    def produce_sync(
        self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]: ...

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str: ...

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str: ...
```

The actual schema values are resolved Kedi/Python types. Return values must map
every requested output name.

## Framework and Harness Kinds

Set `kind` to:

- `"agent-framework"` for `adapter=`;
- `"agent-harness"` for `agent=`.

Kedi rejects instances passed through the wrong selection parameter.

## Shortnames

`shortname` identifies diagnostics, cache identity, telemetry, and approval
requests. It must be a stable string. Passing an instance directly needs no
registry entry; adding a new built-in shortname requires registry and
capability registration.

## Text Invocation

`invoke`/`invoke_sync` receive a fully rendered prompt and return plain text.
Do not return provider message objects. Raw capture stores the returned string.

## Structured Outputs

`produce` receives the rendered template and `output_schema`. Validate provider
output before returning it. Advertise `supports_structured_output=True` only
when typed nested fields work, not when the adapter merely asks for JSON in a
prompt.

## Tool Registration

Implement `SupportsToolRegistration`:

```python
from contextlib import AbstractContextManager
from collections.abc import Sequence
from kedi.agent_adapter import ToolSpec

def register_tool(self, tool: ToolSpec) -> None: ...

def tool_scope(
    self, tools: Sequence[object] = ()
) -> AbstractContextManager[None]: ...
```

Scopes must be concurrency-safe and restore prior registrations. Preserve name,
description, argument/return schema, metadata, risk, and validation.

## MCP Support

MCP servers arrive through the active `AgentProfile`. Map only transports the
backend truly supports and reject unsupported transports explicitly. Do not
advertise MCP because the underlying SDK happens to have an unrelated global
MCP configuration.

## Capability Metadata

`AdapterCapabilities` records structured output, tools, MCP, profile/model/
effort/settings overrides, codemode, native approvals, subagents, background
subagents, accepted setting names, and JSON Schema string formats.

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

Kedi constructs the current user prompt before request assembly. A
`user_prompt_submit` edit is therefore the prompt seen by transport, telemetry,
budgeting, cache identity, and history. Adapters must not independently rebuild
the Kedi template prompt or append a second copy of capability instructions.
Provider-native tool calls and results remain native messages; do not serialize
them into user-prompt text to emulate history.

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
