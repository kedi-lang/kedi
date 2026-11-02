# Prelude, Globals, and Scope

The prelude establishes shared Python state before Kedi declarations and
statements are compiled. Ordinary blocks, by contrast, are isolated execution
units whose new Python locals do not persist.

## The Prelude Block

When the first program content is a top-level Python fence, Kedi treats it as
the prelude:

````kedi
```
from pathlib import Path
from typing import Literal

Environment = Literal["dev", "staging", "prod"]

def normalize_path(value: str) -> str:
    path = Path(value)
    return path.as_posix()
```

@artifact(path: str, environment: Environment) -> str:
  = <environment>:<`normalize_path(path)`>
````

The fence must be the first content, apart from comments and permitted source
metadata. A later top-level fence is an ordinary runtime block.

## Source-Relative Files

A file-backed Kedi program receives `__file__` as its absolute source path.
It is available in the prelude, inline expressions, and procedure bodies:

````kedi
```
from pathlib import Path
```

[notes] = ```
path = Path(__file__)
path = path.with_name("notes.md")
return path.read_text(encoding="utf-8")
```
````

Every imported module receives its own `__file__`; it does not inherit the
caller's file path. Kedi leaves the process working directory unchanged, so
ordinary `Path("notes.md")` calls remain CWD-relative. This distinction lets
programs use both bundled resources and paths relative to the shell invocation
without changing the behavior of concurrent programs.

Unnamed programs and synthetic sources such as notebook cells do not receive
an inferred `__file__` binding.

## Startup Execution

The prelude executes during program compilation/startup, before the main Kedi
body runs. Its names become available to type resolution, defaults, procedures,
inline expressions, and later Python blocks.

Because it runs early:

- a prelude cannot reference a Kedi type or procedure declared later;
- import and initialization failures prevent the program from starting;
- expensive or network-bound work delays every startup;
- declaration-time defaults observe the prelude state at definition time.

Keep the prelude declarative: imports, constants, lightweight helper functions,
and Python types. Put request-specific work in procedures.

## Shared Imports and Helpers

Use the prelude for helpers shared across several Kedi statements:

````kedi
```
import re

SLUG_RE = re.compile(r"[^a-z0-9]+")

def slug(value: str) -> str:
    return SLUG_RE.sub("-", value.lower()).strip("-")
```

@artifact_name(title: str, version: str) -> str:
  = <`slug(title)`>-<version>
````

Do not use it as an unstructured dumping ground. A helper that forms a reusable
project API usually belongs in a Python module imported by the package.

## Kedi Variables as Python Globals

Visible Kedi values are injected as Python globals:

````kedi
[limit: int] = `5`

[result: int] = ```
return globals()["limit"] * 2
```
````

Bare `limit` and `globals()["limit"]` both read the value. `locals()` does not
contain Kedi bindings because the block is not modeled as a Python function
closure.

## Reflection and Promises

In parallel mode, bare reads and `globals()["name"]` resolve a pending Kedi
promise. Non-resolving mapping operations such as `globals().get("name")`,
`globals().items()`, `globals().values()`, and `dict(globals())` intentionally
expose the raw promise so advanced code can forward it without forcing the
dependency.

Use `kedi.force(value)` to resolve a raw promise explicitly. It is a no-op for
ordinary values. Most programs should use bare names and never observe a
promise.

## Rebinding and `global`

An ordinary Python block can update an existing Kedi binding directly:

````kedi
[count: int] = `1`
```
count += 1
```
= `count`
````

A nested Python function needs `global count`:

````kedi
[count: int] = `0`
```
def increment():
    global count
    count += 1

increment()
```
````

Do not use `nonlocal`: there is no enclosing Python function binding. Python
write-back targets the nearest lexical owner and validates all changed values
before committing any of them. New names created in the block remain local and
disappear after the block.

Control-flow bodies add Kedi value scopes. If a branch or iteration declares
`[count] = ...`, later Python in that body updates the child binding and the
outer `count` remains unchanged. Without that local declaration, Python
`count = ...` updates the visible outer owner. Use `[count] := ...` for the
same assignment in native Kedi syntax.

## Procedure and Closure Scope

Inline Python inside a procedure sees parameters, locals, top-level values,
imports, custom types, and prelude helpers:

```kedi
[tax_rate: float] = `0.2`

@gross(net: float) -> float:
  [rounded: float] = `round(net * (1 + tax_rate), 2)`
  = `rounded`
```

Nested Kedi procedures use lexical closures. Their Kedi scope is distinct from
Python's nested-function rules, even though both can access outer data.

Local Kedi names do not leak from a procedure invocation. A nested type or
procedure is likewise unavailable after its enclosing invocation returns.

## Separate Python Blocks

Each ordinary block gets a fresh local namespace:

````kedi
[marker] = ordinary blocks follow a Kedi statement

```
temporary = "not persistent"
```

```
# `temporary` is not defined here.
```
````

If later Kedi code needs a value, assign the block's return:

````kedi
[temporary: str] = ```
return "persistent Kedi value"
```
````

This explicit data boundary keeps execution analyzable and allows parallel
dependency discovery.

## Sequential and Parallel Snapshots

Sequential execution observes writes in source order. In parallel mode,
independent model templates may run concurrently, but each scheduled call
captures its value environment by value. A later write cannot change the inputs
of an already scheduled call.

Bare Python reads create dependency joins and resolve pending values. The
declared value/type contracts remain the same, but stochastic answers and
independent side-effect ordering need not match. A visible `KediPromiseLeak`
indicates an interpreter bug or unsupported advanced promise manipulation, not a
value the application should serialize.
