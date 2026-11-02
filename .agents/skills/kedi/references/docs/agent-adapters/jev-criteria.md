# Jev Criteria and Output Types

Describe finite choices, binary evidence, probabilities, and ordered score levels.
Import the primitives directly with `> import: decisions`, or select only the names
you need. The types come from the optional `kedi-decisions` package and also
work with Laya. Importing them does not select a model, load weights, or require
a provider SDK. Python callers use `kedi.decisions`; standalone callers without
Kedi can import from `kedi_decisions`.

The shared package is currently available from the
[decision-model monorepo](https://github.com/kedi-lang/kedi-decisions).
It is included in Kedi's source development environment. For other environments,
install that checkout with `uv pip install -e /path/to/kedi-decisions`.

```kedi
> import: decisions:
  Probability
  Rubric

~Assessment(
  correctness: Probability,
  quality: Annotated[
    float,
    "Assess the answer against the reference",
    `Rubric(["Incorrect", "Partly correct", "Fully correct"])`
  ]
)
```

Native metadata calls use backticks; arbitrary bare function calls are not added
to the type grammar. Python aliases remain an alternative:

## Typed Criteria

````kedi
```
from typing import Annotated
from kedi.decisions import Probability, Rubric

Correctness = Annotated[
    Probability,
    "Does the answer agree with the reference?",
]
Quality = Annotated[
    float,
    "Assess the answer against the reference",
    Rubric(["Incorrect", "Partly correct", "Fully correct"]),
]
```
> adapter: pydantic
> model: typesafe/jev-latest

~Assessment(
  correctness: `Correctness`,
  quality: `Quality`
)
>> The reference is Ankara and the answer is Ankara. The assessment is [assessment: Assessment].
= `assessment`
````

`Probability` is a finite float between 0 and 1. A rubric with N ordered levels
produces a score between 0 and N-1; float scores retain their fractional part.
Integer rubric outputs round to the nearest level, with ties rounded upward.
Rubric bounds are automatically included and validated; no separate range is needed.

`ChoiceCriteria` supplies descriptions for each exact `Literal`/enum option.
`BooleanCriteria(true=..., false=...)` supplies descriptions for positive and negative
evidence. Import them with `> import: decisions` for Kedi, or from `kedi.decisions` in
Python. They can be used in native backtick metadata or Python `Annotated` aliases.
Importing criteria without the optional package fails with installation advice.
Ordinary `import kedi` and decision-evidence inspection remain available without
it. The former provider-specific Kedi type-import modules are removed; provider
model IDs, settings, and standalone model package APIs are unchanged.


## Outputs and Limitations

- Nested required objects are reconstructed after evaluation.
- Lists of finite string choices are multi-label decisions, in declaration order.
- Nullable finite choices include an explicit no-match option.
- Email, phone, regex, and custom extractors produce finite candidates locally;
  Jev chooses among them. These are extraction, not arbitrary text generation.
- Nullable extraction without candidates returns `None` without a provider request.
- General unconstrained strings, free-form invokes, arbitrary numeric extraction,
  media input, recursive schemas, and arbitrary arrays are unsupported.
- Independent fields share one request. Dependent decisions need separate calls.
- Unsupported schemas and malformed provider output raise errors; they are not
  silently converted to false, repaired, or sent to another model.
