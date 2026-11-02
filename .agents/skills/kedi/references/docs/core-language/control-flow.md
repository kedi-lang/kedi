# Control Flow

Kedi supports deterministic and model-classified branches, conditional loops,
and sequential iterable loops. A trailing `:` after an inline Python condition
selects deterministic execution. Without that trailing colon, the header is a
Kedi template claim classified by the active agent adapter.


## Conditional Branches

```kedi
[score: int] = `72`
[result] = pending

> if: `score >= 60`:
  [result] := passed
> else:
  [result] := failed

= <result>
```

The condition is evaluated exactly once. Its result must have exact Python
type `bool`; values that are merely truthy or falsey are rejected. The closing
`:` after the embedded Python expression is required. Kedi runs only the
selected body, so Python statements, procedure definitions, and model calls in
the other body have no effect.

`> else:` is optional. Kedi has no `elif` form; nest another `> if:` when a
second condition is needed:

```kedi
[grade] = unknown

> if: `score >= 90`:
  [grade] := A
> else:
  > if: `score >= 75`:
    [grade] := B
  > else:
    [grade] := C
```

Each selected body owns a child value scope. A Kedi `=` initialization remains
inside that branch, while `:=` or embedded Python may update a binding already
owned by a containing scope. New Python-only names never become Kedi bindings.
Agent-profile directives inside the body are also lexical. The first evaluated
return ends the enclosing procedure or regular program; later conditions and
statements do not execute. Use `> show:` to display a value and continue.


## Template Conditions

```kedi
[city] = Ankara
[minimum_population: int] = `5_000_000`
[result] = unknown

> if: <city> has more than <`minimum_population`> residents
  [result] := major city
> else:
  [result] := smaller city

= <result>
```

A template condition has no trailing `:`. Plain text, `<name>` substitutions,
procedure calls, and Python substitutions such as ``<`minimum_population`>``
use normal Kedi rendering semantics. Bare backtick expressions are rejected
during parsing; angle brackets make their place in the rendered claim
explicit. `[output]` fields are rejected because a condition does not create a
binding.

Kedi evaluates the rendered claim using the current agent profile and its
available context. A claim that cannot be established is treated as `false`.
The evaluation does not create a Kedi binding.

The trailing colon distinguishes a deterministic condition from a claim even
when both contain one inline Python segment:

```kedi
> if: `is_ready`:
  [kind] = deterministic

> if: <`is_ready`>
  [kind] = model-classified
```


## Conditional Loops

See [Conditional Loops](loops-and-map.md).

## Sequential Loops

See [Sequential Loops](loops-and-map.md).

## Map Continuations

See [Map Continuations](loops-and-map.md).
