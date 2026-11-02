# TypeSafe Jev

Jev evaluates claims, selects finite options, and scores described rubrics. It is
not a free-form text generator. Kedi exposes it through the Pydantic AI and
LangChain adapters using the optional `kedi-typesafe` package.


## Installation

For a published compatible release:

```sh
uv add 'kedi[typesafe]'
# For LangChain, also install the classifier integration:
uv add 'kedi[langchain,typesafe]' 'kedi-typesafe[langchain]'
```

The current `typesafe` and `codex-model`/`terminal-bench` extras pin incompatible
Pydantic AI release lines. Install them in separate environments; uv rejects the
unsupported combined selections explicitly.

Set `TYPESAFE_API_KEY` in the environment. The extended API described here requires
`kedi-typesafe` 0.2.1 or later, which includes the decision evidence used by Kedi bindings.
The development 0.3.0 package lives in the `kedi-decisions` monorepo and reuses
its shared schema implementation without changing these imports. Local Laya
inference is provided by a separate [Laya extension](laya.md), not TypeSafe.


## Claims and Thresholds

See [Claims and Thresholds](jev-claims.md).

## Custom Endpoint

The development SDK integration accepts `TYPESAFE_BASE_URL` for both adapters.
Set it to the HTTPS service origin; the SDK appends `/v1/systemone`. Keep the
key in `TYPESAFE_API_KEY`. An explicit `base_url=` on `TypeSafeModel` or
`TypeSafeChatModel` overrides the environment value.

```python
import os
from kedi_typesafe import TypeSafeModel

model = TypeSafeModel(
    "jev-latest",
    api_key=os.environ["TYPESAFE_API_KEY"],
    base_url=os.environ["TYPESAFE_BASE_URL"],
)
```

A failed custom-endpoint request does not fall back to the public service.
This integration targets TypeSafe SDK 0.7.2 and Pydantic AI 2.54.0; it is not
available in the previously published `kedi-typesafe` package. Use the reviewed
development checkout until its compatible release is published.

## Typed Criteria

See [Typed Criteria](jev-criteria.md).

## Outputs and Limitations

See [Outputs and Limitations](jev-criteria.md).

## Decision Evidence in Kedi

See [Decision Evidence in Kedi](jev-evidence.md).

## Python API

```python
from typing import Annotated
from pydantic import BaseModel
from pydantic_ai import Agent
from kedi_typesafe import Probability, TypeSafeModel

class Assessment(BaseModel):
    supported: bool
    probability: Probability

async def assess():
    async with TypeSafeModel(threshold=0.9) as model:
        result = await Agent(model, output_type=Assessment).run(
            "Reference: Ankara is Turkey's capital. Claim: Turkey's capital is Ankara."
        )
        return result.output
```

LangChain uses `kedi_typesafe.integrations.langchain.TypeSafeChatModel` and
`with_structured_output(Assessment, include_raw=True)`. Raw results expose decision
evidence, provider probabilities/confidence when available, and usage. Evidence is
metadata, not an additional generated output field. Streaming exposes the completed
assessment; it does not simulate token-by-token Jev generation.

Reuse model instances for warm connections. Close owned models using their context
manager or `aclose`; caller-supplied clients remain caller-owned. Configure transport
timeout on the constructor. Unsupported generation settings are rejected.


## Capture Decisions from Python Calls

See [Capture Decisions from Python Calls](jev-evidence.md).

## Explicit Tool Routing

See [Explicit Tool Routing](jev-tools.md).

## Migration from 0.1

See [Migration from 0.1](jev-claims.md).
