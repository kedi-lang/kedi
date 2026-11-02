# Python Interop

Python is Kedi's deterministic execution layer. Use it for arithmetic,
validation, data shaping, library calls, control flow, and local side effects;
use Kedi templates where model judgement is required.

## Where Python Runs

Python appears in five public forms:

- inline expressions in single backticks;
- bare inline segments inside rendered text;
- one-line side-effect statements;
- fenced multiline blocks;
- the first top-level fenced block, which acts as the prelude.

Python also supplies runtime type expressions, directive values, defaults, and
Python API integrations.

## Values Crossing the Boundary

A sole Python expression in a variable initialization, assignment, return, or native call argument
preserves its Python value:

```kedi
[limit: int] = `2 * 5`

@take(values: list[str], count: int) -> list[str]:
  = `values[:count]`
```

Once literal Kedi text is mixed with a value, the result is rendered as `str`.
This boundary is deliberate; do not serialize to text and parse back when a
native value can cross directly.

## Runtime Names

Python can access:

- standard built-ins allowed by the active executor;
- names created by the prelude;
- imported modules and exported values;
- generated Kedi custom types;
- compiled Kedi procedures;
- top-level Kedi values;
- parameters and visible lexical values for the current procedure.

The exact execution environment can vary by executor. Code that requires a
library, filesystem, or network capability should import it explicitly and
document that runtime requirement.

## Type Resolution

Backtick type annotations evaluate in the Python type environment:

````kedi
```
from typing import Literal
Stage = Literal["dev", "staging", "prod"]
```

[stage: `Stage`] = `"prod"`
````

Direct Kedi type syntax is preferable for static visibility. Runtime type
expressions are intended for Python-defined or dynamically selected types.

## Scope Model

Kedi values are exposed to an executed Python block as module globals. Reading a
bare name works normally, but reflection and nested rebinding follow global
rather than function-local semantics. New names created by one Python block do
not become Kedi variables and do not persist into a later block.

Read [Prelude, Globals, and Scope](prelude-globals-and-scope.md) before writing
blocks that mutate existing Kedi state or define nested Python functions.

## Safety and Side Effects

Embedded Python is application code, not a sandbox by default. It can call
libraries and perform side effects allowed by the selected executor and host
process. Treat `.kedi` source with Python blocks as trusted code.

Keep model prompts declarative and side effects explicit. A Python block should
return data or perform a clearly named operation; hidden I/O inside a value
formatting expression makes tests and optimization harder to reason about.

## Choosing a Form

| Need | Use |
| --- | --- |
| One calculation | Single-backtick expression |
| Insert a calculation into text | ``<`expression`>`` or bare backtick segment |
| Several statements that produce a value | Fenced block with `return` |
| Update an existing Kedi variable | Side-effect line or fenced block |
| Share imports and helper functions | Prelude |
| Semantic extraction or generation | `>>` model template, not Python |
