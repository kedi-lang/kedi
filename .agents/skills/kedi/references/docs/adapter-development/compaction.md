# Compaction and Prompt Caching

A long conversation outgrows the context window and gets expensive to resend.
Two mechanisms address that: compaction shortens the history, and prompt
caching lets the provider reuse an unchanged prefix. Both depend on the
adapter's native history, so both are adapter work that Kedi configures and
checks.

## Native Compaction

A program asks for compaction in its history settings:

```kedi
> history:
    enabled: true
    compaction_mode: native
    compaction_threshold: `100_000`
```

`compaction_mode` is `native` or `disabled`. With `native`, the provider or
harness compacts the history itself: OpenAI and Anthropic context management,
Codex thread limits, Claude Agent SDK session compaction. Kedi does not
replace an unsupported provider with a summarizer of its own.

Kedi checks before the call:

| Condition | Error |
|---|---|
| `compaction_mode: native` without history | `CompactionSettingsError: native compaction requires history to be enabled` |
| the adapter does not advertise `supports_native_compaction` | `CompactionSettingsError: native compaction is not supported by adapter '<shortname>'` |

The adapter receives the policy as `profile.compaction`, a
`kedi.agent_profile.compaction.CompactionSettings` with `mode` and an optional
`threshold` in input tokens, in `profile_override`. It must:

- **map the policy to the backend**: enable the provider's compaction and pass
  `threshold` when the backend has one;
- **refuse what it cannot honor**: a backend with no configurable threshold
  must reject an explicit `threshold` (Kedi's Claude adapter does), instead of
  silently using its own;
- **treat `disabled` as "add no compaction"**: it does not switch off
  compaction that a harness performs internally;
- **mark rewritten history**: when a turn's history was compacted, set
  `conversation.cache_prefix_changed = True` so Kedi starts a new cache epoch
  (see below);
- **store the compacted history** in `result_state` like any other turn (see
  [Conversations and History](conversations.md)).

## Prompt Caching

Providers cache a request prefix that repeats exactly: tool definitions,
instructions, and earlier turns. Kedi keeps its side stable:

- instruction parts are ordered from the most to the least stable and do not
  change between calls with the same profile and tools (see [Prompts and
  Instructions](prompt-contract.md));
- each conversation lane has a `cache_key`, available as
  `AdapterConversationContext.cache_key` in `conversation_scope` and as
  `request.conversation.cache_key` on the assembled request.

The adapter's part:

- send tools, then `instructions`, then the history, in the same order and
  serialization on every request;
- pass `cache_key` to the provider's cache-routing option when it has one
  (Kedi's Pydantic AI adapter sends it as OpenAI's `prompt_cache_key`);
- apply provider cache-control markers only to stable content;
- never change earlier history silently. When a processor, compaction, or a
  reset rewrites it, set `cache_prefix_changed = True`. Kedi then increments
  the conversation's `cache_epoch`, derives a new `cache_key`, and drops the
  state of other adapters in the conversation, whose history no longer
  matches.

## Running It

```python title="run_compaction.py"
import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from kedi.agent_adapter import AdapterConversationContext
from kedi.agent_profile.compaction import CompactionSettingsError
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from stateful_adapter import StatefulAdapter


class CompactingAdapter(StatefulAdapter):
    capabilities = replace(StatefulAdapter.capabilities, supports_native_compaction=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policies: list[object] = []
        self.cache_keys: list[str] = []

    @contextmanager
    def profile_override(self, profile: Any) -> Generator[None, None, None]:
        self.policies.append(profile.compaction)
        with super().profile_override(profile):
            yield

    @contextmanager
    def conversation_scope(
        self, conversation: AdapterConversationContext
    ) -> Generator[None, None, None]:
        self.cache_keys.append(conversation.cache_key)
        with super().conversation_scope(conversation):
            yield


class Echo:
    async def complete(self, request: BackendRequest) -> Reply:
        return Reply(text="ok")


SOURCE = """
> history:
    enabled: true
    compaction_mode: native
    compaction_threshold: `50_000`
[a] << First.
[b] << Second.
= <b>
"""


def run(source: str, adapter: StatefulAdapter) -> None:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


adapter = CompactingAdapter(Echo(), model="m1")
run(SOURCE, adapter)
assert {(p.mode, p.threshold) for p in adapter.policies} == {("native", 50_000)}
first_key, second_key = adapter.cache_keys
assert first_key and first_key == second_key

try:
    run(SOURCE, StatefulAdapter(Echo(), model="m1"))
except Exception as error:
    assert "native compaction is not supported by adapter 'chat'" in str(error), error
else:
    raise AssertionError("native compaction needs supports_native_compaction")
```
