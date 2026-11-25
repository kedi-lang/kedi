# Profiles, Models, and Settings

A program configures each call through its active profile: `> model:`,
`> effort:`, `> settings:`, `> fallback:`, `> mcp:`, and the profile
directives that select tools, approval, history, and children. The adapter
sees the profile only if it implements `SupportsProfileOverride`.

## The Protocol

`kedi.agent_adapter.meta.SupportsProfileOverride`:

```python
from contextlib import AbstractContextManager
from kedi.agent_profile import AgentProfile

def default_model(self) -> str | None: ...

def profile_override(self, profile: AgentProfile) -> AbstractContextManager[None]: ...
```

- `default_model()` returns the model the adapter uses when the profile sets
  none. Kedi records it in telemetry and run records.
- `profile_override(profile)` applies `profile` to every call made inside the
  context and restores the previous profile on exit. Kedi enters it around
  each call (see [Call Lifecycle](call-lifecycle.md)), and the adapter enters
  it for a child in `subagent_scope` (see [Subagents](subagents.md)).
  Overrides nest, so keep a stack in a `ContextVar`, as the [reference
  adapter](reference-adapter.md#profiles) does.

## Without the Protocol

An adapter that does not implement `SupportsProfileOverride` runs every call
with its own configuration. Kedi then refuses, before the call, any profile
that sets a model, fallback models, effort, MCP servers, settings, compaction,
or CodeMode:

```text
AgentProfileOverrideUnsupported: Active adapter does not support agent profile override
(requested model=..., effort=..., ...)
```

`> instructions:`, tools, approval, and children need no override: they reach
the adapter as `instructions`, registered tools, wrapped tools, and
delegation tools.

## With the Protocol

Once the adapter implements the protocol, Kedi stops refusing profile fields
for it. The adapter must refuse every field it cannot honor, in
`profile_override`, with `AgentProfileOverrideUnsupported` naming the field.
Silently ignoring a field runs the program with a configuration it did not
ask for.

| Profile field | Directive | Adapter's job |
|---|---|---|
| `model` | `> model:` | Use it instead of `default_model()`. Advertise `supports_model_override`. |
| `effort` | `> effort:` | Map the level to the backend, or refuse it. Advertise `supports_effort`. |
| `settings` | `> settings:` | Apply the names listed in `capabilities.settings`; refuse other names except Kedi's own. Advertise `supports_settings`. |
| `fallback_models` | `> fallback:` | Try the alternatives on transient failures. Advertise `supports_model_fallback`; Kedi refuses `> fallback:` otherwise. |
| `mcp_servers` | `> mcp:` | Connect the servers, or refuse them. See [MCP, Skills, and Capability Tools](mcp-and-skills.md). |
| profile instructions | `> instructions:` | Nothing: Kedi renders them into the `instructions` argument. Do not read them from the profile. |
| `approval`, `hooks`, `history`, `artifacts`, `run_budget`, `compaction`, `codemode` | | Owned by Kedi; the capability pages say what the adapter must add. |

Kedi still checks capability-gated features for every adapter before the call:
`> requires:`, `> fallback:` (`model_fallback`), history processors, native
compaction, CodeMode, hook events, and `tool_reason`.

## Effort

`effort` is one of `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
(`kedi.effort.EffortValue`). Map it onto the backend's reasoning control. When
the backend has fewer levels, map to the nearest one and document it; Kedi's
Pydantic AI adapter maps `max` to `xhigh` (`kedi.effort.pydantic_effort_value`).
When the backend has no reasoning control, refuse `effort` rather than
dropping it.

## Settings

`> settings:` names are validated by the parser against one closed list: the
union of the settings that Kedi itself and the built-in adapters accept (see
[the settings inventory](../agentic-engineering/instructions-and-settings.md#settings-inventory)). A custom
adapter therefore chooses its supported settings from those names, for
example `temperature`, `max_tokens`, or `timeout`, and declares them:

```python
capabilities = AdapterCapabilities(
    supports_settings=True,
    settings=frozenset({"temperature"}),
)
```

Five settings belong to Kedi and may appear in any profile
(`kedi.agent_profile.settings.KEDI_RUNTIME_SETTING_KEYS`):

| Setting | Owner | Adapter's job |
|---|---|---|
| `context_limit` | Kedi's request-size check | none |
| `media_limit` | Kedi's media check | none |
| `request_retries` | the adapter's request retries | read it with `request_retry_limit(settings, default)` |
| `tool_reason` | approval reasons | only adapters that list `tool_reason` in `capabilities.settings` may receive it; Kedi refuses it for the others |
| `voice` | speech output | an adapter with `supports_audio_output` reads it with `speech_voice(settings)` |

## Model Fallback

`fallback_models` is an ordered tuple of model IDs. Kedi's adapters try the
next model when a request to the current one fails with a transient provider
error (`kedi.agent_adapter.request_retry.transient_provider_error`: status
408, 429, or 5xx), per request, after request retries are exhausted. They do
not fall back after a response has started streaming, and an alternative
never replaces an unresolved primary model.

## Running It

```python title="run_profiles.py"
import asyncio

from kedi.agent_adapter import MissingAdapterCapabilityError
from kedi.agent_profile import AgentProfileOverrideUnsupported
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter


class Recorder:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return Reply(text="ok")


def run(source: str) -> list[BackendRequest]:
    backend = Recorder()
    runtime = compile_program(parse_program(source), adapter=ChatAdapter(backend, model="m1"))
    try:
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    return backend.requests


[default] = run("[a] << Hi.\n= <a>\n")
assert default.model == "m1"

[fast] = run(
    "> profile: fast:\n"
    "    > model: m-fast\n"
    "    > settings:\n"
    "        temperature: 0.2\n"
    "> use: fast\n"
    "[a] << Hi.\n"
    "= <a>\n"
)
assert fast.model == "m-fast" and fast.temperature == 0.2

for directive, field in (
    ("> effort: high", "effort"),
    ("> settings:\n    max_tokens: 10", "max_tokens"),
):
    try:
        run(f"{directive}\n[a] << Hi.\n= <a>\n")
    except Exception as error:
        assert isinstance(error, AgentProfileOverrideUnsupported) or isinstance(
            getattr(error, "original", None), AgentProfileOverrideUnsupported
        ), error
        assert field in str(error)
    else:
        raise AssertionError(f"{field} must be refused")

try:
    run("> model: m1\n> fallback: m2\n[a] << Hi.\n= <a>\n")
except Exception as error:
    assert "model_fallback" in str(error), error
    assert isinstance(error, MissingAdapterCapabilityError) or isinstance(
        getattr(error, "original", None), MissingAdapterCapabilityError
    )
else:
    raise AssertionError("fallback needs supports_model_fallback")
```
