# Local Laya

Laya evaluates typed decisions locally. It supports boolean judgments, finite
choices, probabilities, described rubrics, and constrained extraction. It does
not generate free-form text or replace a general-purpose coding agent.

## Installation

The development packages live in the separate
[kedi-decisions monorepo](https://github.com/kedi-lang/kedi-decisions).
They are not published to PyPI yet. From your Kedi checkout, clone it into
`decisions/` if that directory does not already exist, then install into the
active Kedi environment:

```sh
git clone https://github.com/kedi-lang/kedi-decisions.git decisions
uv pip install -e decisions -e 'decisions/packages/kedi-laya[mlx,pydantic]'
```

Use `[torch,pydantic]` for the upstream PyTorch runtime, or include `langchain`
for that adapter. Neither installation requires TypeSafe or a Jev API key.
Installation does not download weights; the first inference loads the selected
checkpoint. An existing local checkpoint directory avoids network downloads.

## A Kedi Program

```kedi
> import: decisions
> adapter: pydantic
> model: laya/aac6fef/laya-multilingual-mlx

> settings:
  decision_threshold: 0.85

>> A customer reports a duplicate charge and asks for a refund.
The responsible team is [team: Literal["billing", "technical"]].
The probability that a refund is requested is [refund: Probability].

= <team> receives the request, with refund probability <refund>.
```

Change the adapter directive to `> adapter: langchain` to use the same program
through LangChain. `> import: decisions` exposes `Probability`, `Rubric`,
`BooleanCriteria`, and `ChoiceCriteria`; importing these types does not select
a model or start inference. Python code can import them from `kedi.decisions`.
The same import works for Jev; only the model and its provider settings change.

## Provider and Backend

`laya/<checkpoint>` always identifies the `laya` provider. Its automatic backend
selection prefers installed MLX on Apple Silicon and otherwise selects PyTorch.
The `laya-mlx/<checkpoint>` compatibility alias forces MLX. Python callers can
choose explicitly:

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent
from kedi_laya import LayaClient

class Route(BaseModel):
    team: Literal["billing", "technical"]

with LayaClient("path/to/checkpoint", backend="mlx") as client:
    agent = Agent(client.as_pydantic_model(), output_type=Route)
    result = agent.run_sync("Please refund my duplicate charge.")
    print(result.output.team)
```

For LangChain, use
`client.as_langchain_model().with_structured_output(Route)`. The direct
`client.as_evaluator()` API returns values, usage, and decision evidence without
either framework. Framework integrations are optional dependencies.

## Decision Policy and Evidence

Booleans use the strict rule `probability > decision_threshold`, default `0.85`.
`decision_tool_call_threshold`, default `0.6`, controls finite tool selection.
Parameterized tool selections require an explicit argument-producing handler;
the model does not fabricate arbitrary tool arguments. Kedi's implicit artifact
toolset is disabled for decision models.

`decision_info("team")` exposes the provider, checkpoint, confidence, and
probabilities through the normal [decision evidence API](jev-evidence.md).
Laya uses the `decisions` metadata envelope; Jev retains `typesafe`. Their
confidence values are not interchangeable calibrated probabilities.

## Operational Limits

- One client loads weights once and serializes its predictions. Async inference
  runs off the event loop; cancelling the waiter does not stop a native kernel.
- Closing a model wrapper does not close a borrowed client. The owner should
  close the client when its session ends. No hosted fallback is used.
- Owned MLX predictors reject input exceeding the checkpoint's actual token
  budget instead of silently truncating it. Injected predictors and PyTorch
  retain their own context handling.
- Real inference has been verified with the multilingual MLX checkpoint.
  PyTorch loader contract tests do not establish correctness for every checkpoint.
