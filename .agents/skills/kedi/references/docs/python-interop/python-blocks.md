# Python Blocks

Fenced Python blocks execute statement suites. They are the right boundary for
control flow, local imports, multiple calculations, exception handling, and
deterministic integration code.

## Multiline Python Blocks

The fences and Python source align with the surrounding Kedi indentation:

````kedi
@statistics(values: list[float]) -> dict[str, float]:
  = ```
  if not values:
      return {"count": 0, "mean": 0.0}

  return {
      "count": len(values),
      "mean": sum(values) / len(values),
  }
  ```
````

The opening and closing ````` markers must be alone on their lines. Do not add
`python` after the opening marker. Kedi dedents the contents relative to the
current Kedi scope before execution.

## Value-Returning Blocks

Use `return` in a block assigned to an L-value:

````kedi
[records: list[dict[str, bool]]] = `[{"ok": True}, {"ok": False}]`
[healthy: int] = ```
return sum(1 for record in records if record["ok"])
```
````

The block result is validated against the initialization annotation. There is no
implicit coercion: returning `"1"` for `[healthy: int]` is an error.

A direct procedure return can also be a block:

````kedi
@load_config(path: str) -> dict[str, str]:
  = ```
  import json
  from pathlib import Path

  config_path = Path(path)
  return json.loads(config_path.read_text(encoding="utf-8"))
  ```
````

This loader expects a JSON object with string values; the return contract
rejects other shapes. Use this form when the block computes a procedure result. Use an
initialization block when later Kedi statements need the value.

## Side-Effect-Only Blocks

A standalone fenced block runs for side effects and ignores its Python return:

````kedi
@record(message: str) -> str:
  ```
  print(f"audit: {message}")
  ```
  = recorded
````

Use standalone blocks for explicit local effects. Avoid hiding writes or network
calls in formatting expressions.

## Single-Line Side Effects

One backtick-delimited line executes as a Python statement:

```kedi
@collect() -> list[int]:
  [values: list[int]] = `[]`
  `values.append(1)`
  `values.extend([2, 3])`
  = `values`
```

This is concise for one mutation. Use a fenced block when several statements
belong together or need control flow.

## Existing and New Names

Python blocks receive visible Kedi names as globals. Assigning to an existing
Kedi variable updates that binding:

````kedi
@normalize() -> str:
  [value] = `"  Ready "`
  ```
  value = value.strip().lower()
  temporary = value.upper()
  ```
  = <value>
````

`value` becomes `"ready"`. `temporary` is local to that execution and is not
available as a Kedi variable or in a later block. Surface new results through a
block initialization or create the Kedi L-value before the block.

## Local Imports and Helpers

Imports and helper definitions inside an ordinary block are local to that
execution:

````kedi
[digest: str] = ```
import hashlib

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

return sha256("payload")
```
````

Put shared imports and helpers in the prelude instead. Keep operation-specific
dependencies local so the program's global namespace stays small.

## Control Flow and Exceptions

Python's normal statement semantics apply:

````kedi
@parse_port(raw: str) -> int:
  = ```
  try:
      port = int(raw)
  except ValueError as exc:
      raise ValueError(f"invalid port: {raw!r}") from exc

  if not 1 <= port <= 65535:
      raise ValueError("port must be between 1 and 65535")
  return port
  ```
````

Catch failures only when translating or recovering from them. Kedi maps an
uncaught Python exception back to the fenced block's source span and preserves
the causal exception chain.

## Nested Python Functions

Kedi bindings are module globals to Python blocks. A nested Python function that
rebinds one must declare `global`, not `nonlocal`:

````kedi
@counter() -> int:
  [count: int] = `0`
  ```
  def bump() -> None:
      global count
      count += 1

  bump()
  bump()
  ```
  = `count`
````

Reading a Kedi value from a nested function needs no declaration. `nonlocal`
fails because the Kedi binding is not in an enclosing Python function scope.

## Executor Constraints

Blocks run through the configured executor. The default local executor has host
Python access; alternate executors may restrict modules, files, networking, or
serialization. Keep portable Kedi logic free of unnecessary host assumptions,
and document executor requirements when a block intentionally depends on them.
