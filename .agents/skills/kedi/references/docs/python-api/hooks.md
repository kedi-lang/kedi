# Hooks from Python

Hooks intercept supported agent events. They differ from stream-event observers: selected hooks can edit or deny an operation.

## Lifecycle Hooks

Register a hook in the current Python API configuration with `@kedi.on(...)`:

```python
import kedi
from kedi import UserPromptSubmitDecision


@kedi.on("user_prompt_submit")
def redact_prompt(event):
    return UserPromptSubmitDecision.edit(event.content.replace("secret", "[redacted]"))
```

The decorator accepts one event name or a sequence of names. `configure()`,
`context()`, `query()`, and `bind()` also accept a `hooks=` event-to-handler
mapping. Direct Python API registrations are inherited by delegated subagents
as runtime policy.

Every supported adapter instance exposes the same decorator:

```python
from kedi.agent_adapter import PydanticAdapter

adapter = PydanticAdapter("openai:gpt-6-luna")
observed = []


@adapter.on(("post_tool_use", "post_tool_use_failure"))
def observe_terminal_tool_event(event):
    observed.append((event.event, event.tool_name, event.tool_call_id))
```

Adapter constructors additionally accept `hook_handler=` for one catch-all
handler. Adapter handlers run before lexical/profile handlers. See
[Agent Lifecycle Hooks](../agentic-engineering/hooks.md) for decision types,
event payloads, ordering, persistence, and backend support.
