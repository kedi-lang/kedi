# Scopes and Binding Lifetime

A binding belongs to the scope that declares it. A procedure invocation, a
selected branch, and each loop iteration have their own value environments.
Visibility is not ownership: seeing an outer value does not make a new local
declaration overwrite it.

## Initialization versus Assignment

```kedi
[status] = pending
[visits: int] = `0`

> if: `True`:
  [status] = local review
  [visits] := `visits + 1`
  [scratch] = temporary

= `(status, visits)`
```

The result is `("pending", 1)`. `status` inside the branch shadows the outer
binding. `visits :=` updates the nearest existing owner. `scratch` disappears
with the branch. Assigning an unknown name with `:=` fails; adding a type
annotation to `:=` is invalid because the owner already defines the contract.

## Python Write-Back

````kedi
[flag: bool] = `False`

> if: `True`:
  ```
  flag = True
  scratch = "not a Kedi binding"
  ```

= `flag`
````

The existing `flag` can be updated. New Python locals such as `scratch` are not
exported into the Kedi environment. If no outer Kedi `flag` existed, writing a
Python local of that name would not create a global Kedi binding. Parameters
and reserved names are not writable through lexical assignment.

## Mutable Objects and Aliases

Lexical isolation is not an object-memory sandbox. A visible list can be
mutated in place, and aliases can observe that mutation. Binding validation
does not prove that arbitrary Python code cannot mutate the interior of an
object into an invalid shape. Revalidate external or mutated data when it
crosses a typed boundary; copy it explicitly when ownership must be isolated.

Scheduled model requests have their own input snapshot behavior. Do not infer
ordinary Python alias behavior from [Concurrency](../runtime/concurrency.md).

## Iteration and Captures

Each iteration gets a fresh binder and local declarations. An outer name with
the same spelling survives unchanged. A map continuation retains its own
iteration environment until its stage is joined. See [Loops and
Map](loops-and-map.md) for nested execution and shared side effects.

Nested procedures can read enclosing value bindings. Model/profile
configuration is captured at definition and local directives affect following
calls in that scope. Value lookup, configuration capture, and definition-time
parameter defaults are different rules; see [Procedures](procedures.md) and
[Parameters and Returns](parameters-and-returns.md).
