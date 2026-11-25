# Capability Reference

`AdapterCapabilities` (`kedi.agent_adapter.AdapterCapabilities`) is the
adapter's declaration of what it implements. Kedi reads it before relying on
an optional protocol, `> requires:` checks it before any model I/O, and the
language server and `kedi explain` read it for built-in adapters.

## Values

Most fields are `bool | None`:

| Value | Meaning | Kedi |
|---|---|---|
| `True` | implemented and tested | relies on it |
| `False` | not implemented | refuses programs that need it |
| `None` | unknown, for example until a remote peer is asked | refuses it, unless the name is in `negotiated_capabilities` |

Declare only what the adapter really implements, not what the underlying SDK
could do. A capability that Kedi relies on but the adapter only approximates
fails in ways Kedi cannot detect.

## Fields

| Field | Means the adapter | Kedi, when it is `True` | When a program needs it and it is not `True` |
|---|---|---|---|
| `supports_structured_output` | constrains answers to a JSON Schema | — | `> requires: structured_output` fails |
| `supports_tool_registration` | implements `tool_scope` / `register_tool` | registers each call's tools | tools on an adapter without the protocol: `AgentToolOverrideUnsupported: Active adapter does not support tool registration` |
| `supports_capabilities` | applies `> capability:` bundles | — (Kedi merges a bundle's instructions and tools into the profile; using one adds an implicit `capabilities` requirement) | `MissingAdapterCapabilityError` for `capabilities` |
| `supports_deferred_tools` | activates `defer: true` capability tools on demand | registers tools with a pending `deferred_capability` | `MissingAdapterCapabilityError` for `deferred_tools` |
| `supports_mcp` | connects profile MCP servers | — (`mcp_servers` reaches the adapter through the profile either way) | `> requires: mcp` fails; the adapter must refuse servers it cannot open |
| `supports_profile_override` | implements `SupportsProfileOverride` | — (the protocol, not the flag, is checked at run time) | `> requires: profile_override` fails |
| `supports_model_override` | honors `> model:` | — | `> requires: model_override` fails |
| `supports_model_fallback` | tries `> fallback:` models on transient failures | passes `fallback_models` | `> fallback:` fails with `MissingAdapterCapabilityError` for `model_fallback` |
| `supports_effort` | honors `> effort:` | — | `> requires: effort` fails |
| `supports_settings` | honors the names in `settings` | — | `> requires: settings` fails |
| `settings` | the accepted setting names | allows `tool_reason` only when listed | `tool_reason is not supported by adapter '<shortname>'` |
| `supports_codemode` | runs the CodeMode controls | allows `> codemode:` | `CodeMode is not supported by adapter '<shortname>'` |
| `supports_native_approvals` | applies approval policies itself | stops wrapping tools with approval; passes `approval_ceiling` to children | — |
| `supports_native_approval_handler` | drives an interactive approval handler natively | — | `> requires: native_approval_handler` fails |
| `supports_subagents` | runs children (`SupportsSubagentExecution`) | offers `delegate_task` and `continue_subagent` | `AgentSubagentUnsupported: Active adapter does not support Kedi subagent delegation` |
| `supports_background_subagents` | lets children run while the parent continues | offers `background`, `wait_subagent`, `cancel_subagent`, `subagent_status` | `background: true` fails as a tool error |
| `supports_live_subagent_input` | delivers input to running children | offers `send_subagent_input`, `send_parent_message` | `> requires: live_subagent_input` fails |
| `supports_subagent_interrupt` | stops and restarts a child attempt | accepts `interrupt: true` | `> requires: subagent_interrupt` fails |
| `supports_artifacts` | passes Kedi tool results through unchanged and registers artifact tools | adds the artifact contract to calls with tools | a call with tools fails: `ArtifactCapabilityError: Active adapter does not support Kedi artifact references and tools` |
| `supports_native_artifacts` | admits adapter-native and MCP tool results before they enter history | — | `> requires: native_artifacts` fails |
| `supports_stateful_history` | continues native conversations (`SupportsConversationState`) | enters `conversation_scope` and keeps continuation state | `> history: enabled` fails: `Adapter '<shortname>' does not support stateful history` |
| `supports_history_replay` | rebuilds the full causal message sequence from Kedi history | — | `> requires: history_replay` fails |
| `supports_history_processing` | runs history processors | allows processors and conditions | `MissingAdapterCapabilityError` for `history_processing` |
| `supports_native_compaction` | compacts history natively | allows `compaction: native` | `native compaction is not supported by adapter '<shortname>'` |
| `supports_stream_events` | emits streaming events | — | `> requires: stream_events` fails |
| `supports_request_budget` | admits each model request against the run budget | allows `> budget: request_limit` | `Adapter '<shortname>' cannot enforce request run budget for this execution` |
| `supports_tool_attempt_budget` | admits each tool attempt against the run budget | allows `> budget: tool_attempt_limit` (not with MCP or external tools) | `Adapter '<shortname>' cannot enforce tool_attempt run budget for this execution` |
| `supports_media_input` | accepts multimodal input: places prompt images, audio, video, and documents in the request | allows multimodal prompts | `MediaError: Adapter '<shortname>' does not accept media input` |
| `supports_image_output` | returns an `Image` for `[img: Image] << ...`: one the model draws, or one of the request's media it names | allows image output | the model can only name an attached image; a request with none fails: `MediaError: Adapter '<shortname>' cannot produce images` |
| `supports_audio_output` | returns an `Audio` for `[speech: Audio] << ...`, from a model that speaks; the request carries no media and no tools | allows speech output | `MediaError: Adapter '<shortname>' cannot produce speech` |
| `hooks` | dispatches the listed hook events (`HookCapabilities`) | runs profile hooks | `Agent hooks are not supported by adapter '<shortname>'`, or `Agent hook events <names> are not supported` |
| `supported_json_schema_formats` | enforces these JSON Schema `format`s | the language server flags other formats for built-in adapters | — |
| `negotiated_capabilities` | checks these `None` capabilities itself, before remote work | lets `> requires:` pass for them | — |

A `—` in the third column means Kedi changes nothing at run time; the flag
still answers `> requires:`, the language server, and `kedi explain`.

## Invariants

The constructor enforces two dependencies:

- `supports_artifacts=True` requires `supports_structured_output=True` and
  `supports_tool_registration=True`;
- `supports_native_artifacts=True` requires `supports_artifacts=True`.

## `> requires:` and Negotiation

```kedi
> requires:
    structured_output
    tool_registration
```

Kedi checks each requirement against `capability_status(name)`:

- `True` passes;
- `False` fails with `MissingAdapterCapabilityError: Adapter '<shortname>'
  does not satisfy required capability '<name>'`;
- `None` fails too, unless the name is in `negotiated_capabilities`. An adapter
  that can only learn a capability from its peer, such as an A2A adapter
  reading an Agent Card, lists it there and must refuse the request itself,
  before starting remote work, when the peer turns out not to support it.

## Built-in Shortnames

The language server and `kedi explain` resolve capabilities from the adapter
registry, by shortname. An adapter passed as an instance
(`compile_program(adapter=...)`) is known only at run time, so its
capabilities produce run-time errors, not editor diagnostics.

## Running It

```python title="run_capabilities.py"
import asyncio
from dataclasses import replace

from kedi.agent_adapter import MissingAdapterCapabilityError
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter


class Echo:
    async def complete(self, request: BackendRequest) -> Reply:
        return Reply(text="ok")


def run(source: str, adapter: ChatAdapter) -> object:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


SOURCE = "> requires:\n    tool_registration\n    stream_events\n[a] << Hi.\n= <a>\n"

try:
    run(SOURCE, ChatAdapter(Echo(), model="m1"))
except MissingAdapterCapabilityError as error:
    assert error.requirement == "stream_events"
    assert "does not satisfy required capability 'stream_events'" in str(error)
else:
    raise AssertionError("stream_events is unknown, so it must fail")


class NegotiatingAdapter(ChatAdapter):
    capabilities = replace(
        ChatAdapter.capabilities, negotiated_capabilities=frozenset({"stream_events"})
    )


assert run(SOURCE, NegotiatingAdapter(Echo(), model="m1")) == "ok"
```
