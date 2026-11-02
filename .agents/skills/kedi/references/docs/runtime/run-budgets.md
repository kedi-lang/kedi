# Run Budgets

A run budget bounds managed work across one root invocation, its procedures,
and its subagents. It is an execution ceiling, not an estimate of token usage
or a provider billing limit.

## Declare a Shared Ceiling

```kedi
> budget:
    tool_attempt_limit: 12

@summarize(note: str) -> str:
    > budget:
        tool_attempt_limit: 2
    [summary] << Summarize <note> in one sentence.
    = <summary>

[note] = Unit tests pass; integration tests remain unverified.
= <summarize(note)>
```

The root allows at most twelve managed tool-body attempts in total. This
procedure allows at most two within its own scope and still debits the root.
The example exposes no tools, so it consumes no tool attempts; it illustrates
scope, not a requirement that the model use tools. Supply a model and credentials
to run it. Add `request_limit` to cap generations too, after configuring the
provider retry boundary described below.

Both fields accept nonnegative integer literals. At least one field is required.
Zero forbids the next operation in that
dimension. Omitting a field leaves its inherited ceiling in force.

The directive is valid at top level, in procedures, and in profiles. A local
ceiling never raises the ancestor's remaining allowance. Each new root
invocation receives a fresh ledger; sharing an adapter does not share its run
budget. Concurrent sibling admissions atomically debit their common ancestors.

## What Gets Counted

| Dimension | Charged work | Not charged |
| --- | --- | --- |
| `request_limit` | Each managed generation attempt, including tool-loop turns, output repairs, and fallback model attempts | Parsing, deterministic code, and work rejected before admission |
| `tool_attempt_limit` | Each entry into a managed tool body, including each selective retry | Argument validation, approval denial, and CodeMode discovery/control wrappers |

Nested application tools inside CodeMode still consume tool attempts. Successful
tool execution followed by an output-validation or post-hook failure still
consumed its body attempt. Failed admitted operations are not refunded.

`RunBudgetExceeded` stops the next operation before it begins. Its `dimension`,
`limit`, `consumed`, and `scope_depth` attributes identify the exhausted boundary.
The exception is not an instruction to retry with an empty result. This is not
rollback: already admitted siblings or tool side effects may have completed.

## Verify the Boundary Without a Provider

This complete Python example uses Pydantic AI's deterministic `FunctionModel`.
The first generation succeeds; the dependent second generation is blocked.
It needs no credentials and makes no network requests.

```python
import asyncio

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from kedi import RunBudget, RunBudgetExceeded
from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

calls = []


async def respond(messages, info):
    calls.append(messages)
    return ModelResponse(parts=[TextPart("Tests need review.")])


program = parse_program("""
[summary] << Summarize the test report.
[action] << Propose one action for <summary>.
= <action>
""")
runtime = compile_program(
    program,
    adapter=PydanticAdapter(FunctionModel(respond)),
    run_budget=RunBudget(request_limit=1, tool_attempt_limit=2),
)
try:
    try:
        runtime.run_main()
    except RunBudgetExceeded as exc:
        assert exc.dimension == "request"
        assert exc.limit == exc.consumed == 1
    else:
        raise AssertionError("The second generation must be blocked")
    assert len(calls) == 1
finally:
    asyncio.run(runtime.aclose())
```

## Provider Retries Must Be Observable

A strict generation count cannot silently ignore retries inside a provider SDK.
For a finite `request_limit`, configure its actual client with `max_retries=0`.
Putting that field in Kedi `> settings:` does not configure every SDK's client.
For example, a Pydantic OpenAI model can be constructed in the embedding host:

```python
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import PydanticAdapter

client = AsyncOpenAI(max_retries=0)
model = OpenAIResponsesModel(
    "gpt-6-luna", provider=OpenAIProvider(openai_client=client)
)
adapter = PydanticAdapter(model)
```

Supply `OPENAI_API_KEY`, pass this adapter to `compile_program`, and close the
client with `await client.close()` when the embedding host finishes. LangChain's
OpenAI model similarly accepts `ChatOpenAI(model="gpt-6-luna", max_retries=0)`.
Kedi does not mutate a shared client's retry settings. If a provider's retry
control is opaque or enabled, the finite request budget rejects it before
generation rather than claiming a ceiling it cannot enforce.

## Capabilities and Other Limits

Adapters advertise `request_budget` and `tool_attempt_budget` separately.
Pydantic AI and LangChain support the managed boundaries; unsupported adapters
or opaque provider tools reject an applicable finite limit. Arbitrary SDK calls
inside embedded Python are outside this accounting boundary.

Run budgets do not replace per-child `SubagentUsageLimits`, descendant-start
limits, deadlines, or context/token guards. Those remain independently enforced;
whichever boundary is reached first stops its operation. See
[Subagent Limits](../agentic-engineering/subagent-limits.md) and
[Capability Matrix](../reference/capability-matrix.md).
