# Loops and Map Continuations

Conditional loops repeat a condition; iterable loops visit values; map adds a continuation per retained iteration. None creates an implicit result collection.

## Conditional Loops

```kedi
[remaining: int] = `3`

> loop: `remaining > 0`:
  `remaining -= 1`

= `remaining`
```

The exact-`bool` condition is evaluated before every iteration. A false first
result executes no body. Every body execution receives a fresh child scope.
State needed by the next condition must update an existing outer binding with
`:=` or owner-aware Python write-back; new body-local declarations disappear
after that iteration.

Template conditional loops use the same claim syntax and re-render the claim
before every iteration:

```kedi
> loop: <work> remains unfinished
  >> Continue the unfinished work.
```

Conditional loops allow 10,000 completed iterations by default. If their
condition is still true, Kedi raises `LoopIterationLimitError` before starting
the next body. Nested loops have independent counters. Python callers can set a
positive `loop_iteration_limit` through `compile_program()`, `configure()`,
`context()`, or `interactive()`.


## Sequential Loops

```kedi
[items: list[int]] = `[1, 2, 3, 4, 5]`
[total: int] = `0`

> loop [n]: `items`:
  `total += n`

= `total`
```

The expression is evaluated once and must return an `Iterable`. Kedi traverses
it sequentially in its native order without materializing it or adding implicit
parallelism. Lists, tuples, dictionaries, strings, generators, ranges, and
custom iterables therefore retain their normal iteration behavior. If the
iterator exposes `close()`, Kedi calls it after normal traversal and on an
early exit caused by a loop-body failure.

The binder receives each yielded value without coercion and is visible to every
statement and nested block in the current iteration. Every iteration owns a
fresh child scope, so its binder and local declarations cannot leak into or
overwrite another iteration. The binder's lifetime is restricted to the loop:

```kedi
[n: int] = `99`
[values: list[int]] = `[]`

> loop [n]: `range(3)`:
  `values.append(n)`

= `(values, n)`
```

This returns `([0, 1, 2], 99)`. Each iteration binder belongs to a fresh child
scope, so the outer `n` is never overwritten and the iteration binding is
discarded when that scope ends. Nested loops may shadow the same binder safely.
Outer values change only through `:=`, owner-aware Python write-back, or
mutation of an outer object.


## Map Continuations

One sibling `> map:` clause may immediately follow an iterable loop:

```kedi
[selected: list[str]] = `[]`
[candidates: list[str]] = `["Ada <ada@example.org>", "Missing address"]`

> loop [candidate]: `candidates`:
  >> Eligible candidates have a named contact and a usable email address.
  It is [qualified: bool] that <candidate> meets these criteria.
  Their contact email, if present, is [email: str | None].
> map:
  > if: `qualified and email is not None`:
    `selected.append(email)`

= `selected`
```

This is a deferred continuation stage, not Python's collection-producing
`map()`. Kedi first traverses the iterable and starts every loop-body job. It
then schedules one continuation for each retained iteration scope. Each map
execution sees only its own binder, declarations, and model outputs plus the
containing scopes.

The configured execution engine controls concurrency. Sequential execution
preserves source order; a parallel engine may finish independent records out of
order. Kedi joins and drains the entire stage before continuing after the loop,
then surfaces the first failure.

Parallel map continuations may overlap. Source-order side effects and atomic
read-modify-write operations are not implied: shared aggregates must use
operations or synchronization appropriate for the active execution engine.

Kedi accepts one map stage on an immediately preceding
binder-based loop. It has no binder and creates no implicit output collection.
Orphan maps, maps attached to conditional loops, and chained maps are parse
errors.

## Nested Stages

Nested loops and map stages retain their own binders. This deterministic
example uses indexed writes so its result does not depend on completion order:

```kedi
[rows: list[list[int]]] = `[[1, 2], [3, 4]]`
[squares: list[list[int]]] = `[[0, 0], [0, 0]]`

> loop [row_index]: `range(len(rows))`:
  > loop [column_index]: `range(len(rows[row_index]))`:
    [square: int] = `rows[row_index][column_index] ** 2`
  > map:
    `squares[row_index][column_index] = square`

= `squares`
```

The result is `[[1, 4], [9, 16]]`. The inner stage completes before that outer
iteration finishes. Independent map callbacks can overlap; use disjoint
destinations as above or explicit synchronization. A shared increment such as
`total += square` is not an implicit atomic reduction.
