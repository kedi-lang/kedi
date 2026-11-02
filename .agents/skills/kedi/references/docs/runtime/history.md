# Conversation History

History retains successful turns. It is not response memoization: a new prompt still invokes the model.

## Stateful Conversation History

Kedi model calls are stateless by default. Enable history in a lexical scope
when later template or raw-invoke blocks must receive complete successful turns
from earlier calls:

```kedi
> history: enabled

>> Remember [project_name].
>> A concise tagline for <project_name> is [tagline].

= <tagline>
```

History is valid at top level, in procedures, and in profiles. A nested
`> history: disabled` scope neither reads nor mutates enabled outer history.
Failed and cancelled calls do not commit partial turns.

Prompt hooks run before a turn enters history. When `user_prompt_submit` edits a
prompt, transport, budgeting, telemetry, cache identity, and history all use the
same edited content. A successful prompt/result pair, native continuation,
cleanup ownership, and cache-epoch effect commit atomically. Parallel calls
sharing one conversation run these complete transactions in source order;
separate conversation sessions remain independent.

History is partitioned by concrete adapter and compatible model, settings, MCP,
and tool-contract lane. Each lane keeps its native or portable message sequence,
tool lifecycle, and stable cache identity. Changing an incompatible input starts
a fresh native continuation rather than replaying the previous checkpoint.
Kedi does not translate private provider messages between frameworks, and tool
calls/results remain native causal messages rather than being flattened into a
user prompt. Artifact release and expiry do not delete or reorder existing
messages, so a cached prefix remains append-only within its cache epoch.

## User-Defined History Processing

`PydanticAdapter` and `LangChainAdapter` accept an optional
`history_processor=` callback when an application needs deterministic retention,
redaction, or its own summarization policy. It runs before every logical model
request, including later requests in a tool loop, and receives detached native
messages plus validated editing groups:

```python
from typing import TypeVar

from kedi.agent_adapter import (
    HistoryProcessorContext,
    LangChainAdapter,
    PydanticAdapter,
)

MessageT = TypeVar("MessageT")


def keep_recent_cycles(ctx: HistoryProcessorContext[MessageT]) -> list[MessageT]:
    return ctx.keep_recent_groups(4)


pydantic_adapter = PydanticAdapter(history_processor=keep_recent_cycles)
langchain_adapter = LangChainAdapter(history_processor=keep_recent_cycles)
```

## Native Configuration

Select the same callback from a Kedi program, procedure, or profile:

```kedi
@keep_recent(ctx: Any) -> list[Any]:
    = `ctx.keep_recent_groups(4)`

> history:
    enabled: true
    processor: `keep_recent`

>> The proposed next step is [next_step: str].
= <next_step>
```

Use `Any` for a native callback's framework-specific context and messages.
An unannotated parameter means `str`; `object` is not a supported native Kedi
type name. Python callbacks can use the generic `HistoryProcessorContext` type.

`processor` is a callable reference, not its result. The expanded directive still
requires `enabled: true|false`; `compaction_mode` and `compaction_threshold` remain
sibling fields. Python callers can use
`AgentProfile(history_processing=HistoryProcessingSettings(processor=callback))`
from `kedi.agent_profile` for the same lexical override.

Omitting `processor` inherits the captured policy or adapter constructor default.
An explicit processor value must evaluate to a callable. Omission inherits;
it does not clear an outer processor. Leaving the scope restores the previous
policy. Shared adapter fields are not mutated,
so concurrent invocations can use different processors safely.
Native procedure callbacks bind to their selected definition; a deeper procedure
with the same name cannot replace the captured callback.

## Conditional Processing

Use `processor_condition` to evaluate whether the processor is needed before
each logical model request, including requests after tool calls:

```kedi
@keep_recent(ctx: Any) -> list[Any]:
    = `ctx.keep_recent_groups(4)`

@needs_compaction(state: Any) -> bool:
    = `state.estimated_tokens >= 80_000`

> history:
    enabled: true
    processor: `keep_recent`
    processor_condition: `needs_compaction`

>> A concise next step for this project is [next_step].
= <next_step>
```

The predicate receives a frozen `HistoryProcessorState`, not editable messages.
It has `adapter_shortname`, `model_id`, one-based `request_index`, `cache_epoch`,
`estimated_tokens`, and an opaque `history_version`. The estimate covers the
candidate history, including any system messages it contains. It does not include
separately injected instructions or tool schemas and is not an exact wire-request
budget. A true result builds `HistoryProcessorContext` for the same history
revision; false skips the processor's detached copy and edit validation. Normal
request validation remains active. The predicate may be synchronous or async
and must return exactly `bool`; errors and other return types stop the request
before model transport. It is level-triggered, with no cooldown. Cancellation
does not commit a pending rewrite or cache generation.

Python adapters use `history_processor_condition=needs_compaction` alongside
`history_processor=keep_recent_cycles`. A condition alone does not choose a
processor. A condition-only lexical override can use an outer or constructor
processor. Replacing a processor without specifying a condition makes the new
processor unconditional. An explicit condition must evaluate to a callable.
An effective condition
without a processor is a configuration error.

The LSP warns at active processor declarations, including external callables.
Python constructors warn once per configuration, not per request. A processor
may change earlier history and reduce prefix-cache reuse. A condition reduces
how often the processor runs, but it does not guarantee prefix preservation or
a provider cache hit. A semantic no-op preserves native history and cache
identity. Evaluating the predicate and estimating tokens still cost work.

`enabled: false` disables conversation continuity, not processing of the current
invocation's ephemeral native tool loop. It does not expose an outer conversation
to the callback. Unsupported adapters reject an active processor or condition
before model I/O; the declared capability is `history_processing`.

## Recent Groups and Origins

`ctx.keep_recent_groups(n)` keeps the last `n` closed, unprotected groups and all
protected, unfinished, and current-request groups, in their original order.
It returns a native message list without changing the context. `n` must be a
nonnegative integer, not a boolean. `n=0` still preserves required history; the
total retained group count may exceed `n`. No summary model is called and no
token-budget fit is promised. Normal edit validation still applies.

Protected groups do not count toward `n`. For example, if the newest completed
cycle contains provider-required reasoning, `keep_recent_groups(1)` retains it
and the newest unprotected completed cycle. Retention can therefore differ
between providers even for similar visible text. Inspect `group.protected` and
`group.protected_reasons` when building a more selective policy; do not remove
opaque replay data to force a smaller history.

Each `group.origins` is a tuple of frozen `HistoryOrigin` records:

| Field | Meaning |
| --- | --- |
| `kind` | Existing history kind, such as `user`, `assistant`, `tool-call`, or `tool-result` |
| `run_id` | Runtime-known run identity, or `None` |
| `source_location` | Detached Kedi source span, or `None` for unknown/native-only locations |
| `tool_name` | Exact native or execution-boundary tool name, when known |
| `tool_call_id` | Native tool lifecycle identifier, when known |

Origins are out-of-band metadata, never extra prompt text. They survive supported
edits and successful in-process conversation continuation, including turns where
the condition skips the processor. Imported history,
ambiguous copied duplicates, generated summaries, and calls made while processing
is disabled may lack lineage; Kedi does not infer it from message text. This is
not a durable provenance export protocol, a truth score, or proof of tool side
effects. Metadata edits inside a callback cannot change the retained lineage.

Read origins while returning `list(ctx.messages)` to inspect without rewriting.
A no-op preserves cache identity. Retaining fewer groups can remove an early
prefix and reduce cache reuse; this policy is opt-in, not a universal speedup.

For example, inspect known tool origins without changing the outgoing history:

```python
from kedi.agent_adapter import HistoryProcessorContext


def inspect_origins(ctx: HistoryProcessorContext) -> list:
    for group in ctx.groups:
        for origin in group.origins:
            if origin.tool_name is not None:
                print(origin.run_id, origin.tool_name, origin.tool_call_id)
    return list(ctx.messages)
```

This logs identifiers, not tool arguments or results. Treat identifiers and
source locations as application metadata when choosing a logging destination.

For executable, deterministic coverage of provider-native tool loops, see the
[history-processor regression tests](https://github.com/kedi-lang/kedi/blob/stable/tests/test_pydantic_user_history_processor.py).
From the checkout, run `uv run pytest tests/test_pydantic_user_history_processor.py --record-mode=none`.
These checks need no API key and make no network requests.

## Callback Contract

The callback may be synchronous or asynchronous. Synchronous callbacks run in a
worker thread rather than blocking the adapter event loop. It must return a
non-empty sequence in the owning framework's native message type. Returning the
input or an equivalent fresh sequence is a no-op. Do not slice an arbitrary last
`N` messages: a boundary can split a tool call from its result. `ctx.groups`
exposes the atomic ranges that can be selected safely.

Leaving `history_processor=None` disables this work. An enabled no-op still
copies and inspects messages; it preserves history and continuation rather than
eliminating processing overhead. Native message classes and typed tool payloads
are deep-copied, so the callback must not depend on sharing object identity with
the caller. Serialized dictionaries are not accepted in place of native messages.

The frozen context also provides `adapter_shortname`, `model_id`, one-based
`request_index`, `cache_epoch`, a conservative `estimated_tokens` value, and
per-entry metadata. It does not expose the complete wire request: system
instructions, tool schemas, routing, approvals, and model settings remain owned
by the adapter.

## Editing Boundaries

### Context Budget Accounting

The host request budget is separate from the callback's inspection estimate.
After a successful Pydantic AI or LangChain run, Kedi can bind the last native
response's measured input plus output usage to the exact saved history digest.
Tool-result messages appended after that response are added conservatively.
This avoids treating serialized history bytes as model tokens on an unchanged
continuation. New instructions, schemas and prompt content still enter the next
request's reservation; request and cumulative-input limits are unchanged.

If history is edited, reordered or extended outside the captured run, or native
usage is absent, Kedi falls back to its conservative content estimate. Local
lineage and usage-checkpoint metadata are not model-visible text. Unknown state
is still counted. This uses provider-reported usage, not a tokenizer guarantee.
A2A and subagent persistence retain and validate the digest-bound checkpoint;
older checkpoints without this optional field remain supported.

### Atomic Groups

`group.entry_start` and `group.entry_end` are half-open indices into
`ctx.messages`. Groups cover complete user/assistant cycles, including their
tool exchanges. Keep or replace an eligible group as a whole. Redacting one
message while retaining the rest of its original group is rejected; replace
the entire eligible group with native text summary messages instead.

Use `group.closed`, `group.protected`, and `group.protected_reasons` to decide
what is editable. Retained groups may be reordered within the same protected
boundaries, but their internal message order must remain intact. Keep the last
request message exactly, and preserve system messages, checkpoints, unfinished
work, and provider-required records. A signed reasoning block can therefore
protect its entire group.

This conservative grouping does not expose individual completed tool exchanges
inside an ongoing user cycle for arbitrary editing. Use `history_archive=` for
the existing exact archival of long tool loops. Combining it with this callback
does not allow the callback to remove the resulting checkpoints.

Kedi clones the callback input and accepted result. Before transport it rejects
orphan tool results, partial lifecycle groups, an altered current request,
fabricated provider/checkpoint records, and removal of pending or signed native
state. Callback errors propagate; Kedi never silently sends the old history.

`HistoryProcessingError` wraps callback/copy failures;
`HistoryProcessingValidationError` identifies invalid edits. Cancellation
propagates normally. A running synchronous callback's worker thread cannot be
forcibly stopped: its late result is discarded, but application-owned side
effects still need their own cancellation policy. Callback state shared between
concurrent runs is also the application's responsibility. Kedi keeps its request
counters and pending transport resets separate, including nested native runs on
the same adapter. Calling that same configured adapter recursively from inside
the processor raises an error; use a separate processor-free model for a summary.

## Persistence and Continuation

An accepted edit replaces Pydantic AI's native run history or LangChain's
durable graph state, so later tool steps and successful future turns see it.
Stale response-ID continuation is disabled before an incompatible request and
the conversation's cache generation is staged atomically. Failure or cancellation
does not commit the edit. When `history_archive=` is also enabled, exact archival
runs first and the callback sees the checkpointed native history.

Each actual rewrite stages a generation; two rewrites in one run can advance
it twice. Requests without another rewrite reuse the current generation, and
successful commit does not advance it again. An old server-owned conversation
ID remains disabled on later turns of the same Kedi conversation lane: otherwise
the server could reintroduce removed history. Automatic response-ID continuation
may resume from a response produced after the edit; a fixed old response ID is
not reused. Caller-owned model settings are not mutated.

These cross-turn guarantees require Kedi's conversation state. With native
Pydantic `message_history=` alone, carry forward `result.all_messages()` and
appropriate model settings yourself. Do not append `new_messages()` to the old
unprocessed list, because that would restore the history you removed.

A separately supplied Pydantic `ProcessHistory` capability is rejected when
`history_processor` is configured. Compose the policies in one callback instead.
Other capabilities must also respect these message boundaries.

The Python callback and native `processor` field share one implementation.
Neither introduces a second DSL compaction mode or a built-in summarizer.
