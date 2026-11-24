# Artifacts

A tool can return more data than a model should read at once: a log file, a
search dump, a large table. Kedi stores such a result as an artifact and gives
the model a compact reference and tools to read it in bounded pieces. The
program itself still receives native values; artifacts only change what the
model sees.

## What Kedi Does

For every call with tools, when artifacts are enabled (the default):

1. it adds the `artifacts.instructions` part, which explains references and
   the retrieval tools, to `instructions`;
2. it registers the artifact management tools with the call's other tools:
   `read_artifact`, `search_artifacts`, `release_artifact`, and
   `run_artifact_code`, plus `query_artifact` when `query_artifacts` is
   enabled;
3. it wraps each tool's `fn`: when a successful result serializes to more than
   the policy's `threshold` (100 kB by default), the result is stored in the
   session's artifact store and `fn` returns the reference instead.

A reference is a JSON object (`ArtifactRef.model_dump(mode="json")`):

| Field | Content |
|---|---|
| `ref_id` | the opaque id the model passes to `read_artifact` |
| `logical_type`, `media_type` | what the value is |
| `summary`, `preview` | a short description and the first `preview_chars` characters |
| `size_bytes`, `char_count` | the stored size |
| `created_at`, `expires_at` | lifetime under `ttl` |
| `sensitive` | whether the value must not be echoed |
| `usage_hint` | how to read the rest |

## The Adapter's Part

Advertise `supports_artifacts=True` when the adapter:

- **registers every tool Kedi passes**, including the management tools;
- **returns each `fn` result to the model as is**: a reference stays a
  reference, and the adapter never fetches, expands, or truncates it;
- **keeps every tool-call id paired with its result**: a later `read_artifact`
  call is a new tool call with a new result, not an edit of the old message;
- **leaves storage to Kedi**: no artifact store or manager of its own, and no
  re-running of the source tool to get the full value.

The constructor requires `supports_structured_output` and
`supports_tool_registration` alongside it. Without `supports_artifacts`, a
call with tools fails with `ArtifactCapabilityError: Active adapter does not
support Kedi artifact references and tools`, unless the program disables
artifacts for the profile:

```kedi
> artifacts:
    enabled: false
```

A disabled profile sends every tool result to the model in full.

The [reference adapter](reference-adapter.md) satisfies all four rules
without artifact-specific code: it registers what it is given and serializes
whatever `fn` returns.

## Native and MCP Tool Results

Kedi wraps only the tools it registers. Results from backend-native tools
(a harness's file or shell tools, provider built-ins) and from MCP servers the
adapter connects itself bypass that wrapper. To bring them under the same
policy, admit each result before it enters model-visible history:

```python
from kedi.agent_adapter.tools.execution import admit_current_artifact_result

model_visible = await admit_current_artifact_result(
    result, tool_name=native_name, call_id=call_id
)
```

The function returns the reference for a large result, the result unchanged
for a small one, and the result unchanged outside a Kedi request (no active
artifact scope). Send `model_visible` to the model in place of `result`.
Advertise `supports_native_artifacts=True` only when every native and MCP
result goes through it.

## Failures

| Situation | Result |
|---|---|
| the session's `session_quota` or `max_artifacts` is exhausted | `fn` returns an error object for the model instead of the reference; the run continues |
| a result is too large to admit at all | `ToolOutputTooLargeError` propagates from `fn` |
| the model reads an unknown or expired `ref_id` | the management tool fails with a correctable error the model can read |

A failed admission never falls back to sending the full payload.

## Running It

```python title="run_artifacts.py"
import asyncio
import json

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter

SOURCE = """
@read_log(day: str) -> str:
  ###
  Return the audit log of one day.
  ###
  > tool:
    risk: read_only
  = `"line " * 1000 + "APPROVED BY ada"`

> artifacts:
    threshold: 1kb
> use: read_log
[answer] << Who approved the change on 2026-11-20?
= <answer>
"""


class Script:
    def __init__(self) -> None:
        self.results: list[dict] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not self.results:
            return Reply(tool_calls=(ToolCall("c1", "read_log", {"day": "2026-11-20"}),))
        if len(self.results) == 1:
            ref = self.results[0]["result"]
            arguments = {"ref_id": ref["ref_id"], "pattern": "APPROVED BY"}
            return Reply(tool_calls=(ToolCall("c2", "read_artifact", arguments),))
        return Reply(text=json.dumps(self.results[1]["result"]))


backend = Script()
runtime = compile_program(parse_program(SOURCE), adapter=ChatAdapter(backend, model="m1"))
try:
    answer = runtime.run_main()
finally:
    asyncio.run(runtime.aclose())

ref = backend.results[0]["result"]
assert set(ref) >= {"ref_id", "preview", "size_bytes", "usage_hint"}
assert ref["size_bytes"] > 1000 and "APPROVED" not in ref["preview"]
assert "APPROVED BY ada" in answer
```

The model received a reference instead of 5 kB of log, then found the line it
needed with one bounded read.
