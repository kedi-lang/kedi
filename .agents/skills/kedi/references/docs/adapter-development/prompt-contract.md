# Prompts and Instructions

Kedi writes every piece of text a model reads for a Kedi call. The adapter
sends that text to its backend unchanged. This page lists what the text
contains, in which order, and the few places where an adapter influences it.

## The Two Strings

Each call carries two strings:

| Argument | Send it as | Contents |
|---|---|---|
| `instructions` | the system or developer message | instruction parts, joined in order with a blank line |
| `prompt` (`invoke`) / `template` (`produce`) | the user message | the rendered prompt for this call |

Send each exactly once. Do not parse them, re-render them, translate them into
another format, or add text of your own around them.

## Instruction Parts

`instructions` is built from parts. Each has a stable identity and a source,
and appears only when it applies:

| Order | Identity | Source | Present when |
|---|---|---|---|
| 1 | `template.instructions` | runtime | the call is a template capture in completion mode; wrapped in `<TemplateInstructions>` |
| 2 | `profile.instructions` | profile | the profile has `> instructions:` |
| 3 | `capability:<identity>` | capability | an active capability contributes instructions |
| 4 | `grounding.instructions` | runtime | the profile has tools, MCP servers, or skills; wrapped in `<ToolCallInstructions>` |
| 5 | `skills.instructions` | skills | skills are enabled |
| 6 | `artifacts.instructions` | artifacts | the artifact contract is active for the call |
| 7 | `runtime.claim` | runtime | the call is a claim |

Two consequences for adapters:

- **The text is cache-stable.** Parts are ordered from the most stable to the
  most call-specific, and their content does not change between calls with the
  same profile and tools. Keep `instructions` first in the provider request so
  prompt caching works.
- **Profile instructions are already included.** Do not also read the
  profile's `> instructions:` from the active profile and send them; they would
  appear twice.

## The Prompt

| Call | Prompt |
|---|---|
| raw invoke `<<` | the rendered text of the statement |
| template capture, completion mode | `Fields to be substituted into the template string for final construction: (a, b).` followed by `The template string:` and the rendered template |
| template capture, decision mode | `Evaluate each marked decision field using its question, criteria, and the context below.` followed by the field list and the rendered template as context |
| claim | `Claim: <rendered claim>` |
| child run | the delegated task; see [Subagents](subagents.md) |

The completion rules (what a field value is, how to treat the surrounding
template text, escaped brackets, and so on) live in
`<TemplateInstructions>`, not in the prompt. Models that should not receive
them, such as a classifier, use decision mode.

## Decision Mode

An adapter chooses the template prompt mode per profile by implementing
`SupportsTemplatePromptMode`:

```python
from kedi.agent_adapter.meta import TemplatePromptMode
from kedi.agent_profile import AgentProfile

def template_prompt_mode(self, profile: AgentProfile) -> TemplatePromptMode:
    return "completion"  # or "decision"
```

In decision mode the prompt asks the model to evaluate each field from the
context, and `template.instructions` is left out. Kedi's decision models
(TypeSafe Jev and Laya) use it. Adapters for general chat models do not
implement the protocol and get completion mode.

## Hook Edits

`user_prompt_submit` hooks run before the request is assembled. When a hook
replaces the prompt, the adapter receives the replaced text, and so do
telemetry, budgeting, cache identity, and history. The adapter must not apply
hooks again or keep a copy of the original prompt. See [Hooks](hooks.md).

## What an Adapter Must Not Add

- template-filling or field-extraction rules, examples, or reminders;
- output-format instructions such as "answer in JSON with fields ...": send
  the JSON Schema through the backend's structured-output feature instead (see
  [Structured Output](structured-output.md));
- tool-usage instructions or tool catalogs when the backend accepts tool
  declarations;
- a second copy of the profile instructions or of any part above;
- correction turns of its own. Where Kedi supplies correction text, as
  `accept_subagent_output` does for child results, send that text.

Text that identifies a harness and does not depend on the request, such as an
agent harness's own fixed system prompt, is not part of this contract.

## Seeing the Text

```python title="run_prompt_contract.py"
import asyncio
from typing import Any

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter


class Recorder:
    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return self.reply


def run(source: str, reply: Reply, adapter_type: type[ChatAdapter] = ChatAdapter) -> BackendRequest:
    backend = Recorder(reply)
    runtime = compile_program(parse_program(source), adapter=adapter_type(backend, model="m1"))
    try:
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    [request] = backend.requests
    return request


PROFILE = """
@lookup(city: str) -> str:
  ###
  Look up facts about a city.
  ###
  > tool:
    risk: read_only
  = `'facts'`

> instructions: Answer as a careful geographer.
> use: lookup
"""

capture = run(PROFILE + ">> The capital of France is [city].\n", Reply(data={"city": "Paris"}))
system = capture.instructions or ""
order = [
    system.index("<TemplateInstructions>"),
    system.index("Answer as a careful geographer."),
    system.index("<ToolCallInstructions>"),
]
assert order == sorted(order), system
assert capture.messages[0]["content"].startswith(
    "Fields to be substituted into the template string for final construction: (city)."
)
assert system.count("Answer as a careful geographer.") == 1

claim = run(PROFILE + "> if: Paris is in France\n  = yes\n", Reply(data=True))
claim_system = claim.instructions or ""
assert "<TemplateInstructions>" not in claim_system
assert claim_system.index("Answer as a careful") < claim_system.index("Determine whether")
assert claim.messages[0]["content"] == "Claim: Paris is in France"
```

A decision-mode adapter receives a different prompt and no completion rules:

```python title="run_prompt_contract.py"
class DecisionAdapter(ChatAdapter):
    def template_prompt_mode(self, profile: Any) -> str:
        return "decision"


decision = run(
    ">> The ticket [urgent: bool] needs a reply today.\n",
    Reply(data={"urgent": True}),
    DecisionAdapter,
)
assert decision.messages[0]["content"].startswith("Evaluate each marked decision field")
assert "<TemplateInstructions>" not in (decision.instructions or "")
```
