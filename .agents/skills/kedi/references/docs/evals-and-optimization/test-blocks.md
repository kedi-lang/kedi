# Test Blocks

## Define `@test:` Blocks

A test suite is a top-level block named after the procedure it validates. Each
case has a unique label and contains Python code:

````kedi
@normalize_tag(tag: str) -> str:
  = `tag.strip().lower()`

@test: normalize_tag:
  > case: lowercases:
    `assert normalize_tag("KEDI") == "kedi"`

  > case: strips_whitespace:
    ```
    result = normalize_tag("  docs  ")
    assert result == "docs"
    ```
````

Use a backtick line for a short statement. Use a fenced Python block for setup,
multiple assertions, or control flow.

## Target Procedures

The name after `@test:` identifies the procedure in reports:

```kedi
@add(left: int, right: int) -> int:
  = `left + right`

@test: add:
  > case: positive:
    `assert add(2, 3) == 5`
```

The suite is validation metadata; it does not invoke the target automatically.
Each case must call the procedure itself when that is part of the test.

## Arrange Runtime Values

Cases execute against the compiled Kedi runtime. They can use prelude values,
custom types, imported exports, prelude helpers, and compiled procedures:

````kedi
```
def canonical(value):
    return value.casefold().strip()

prefix = "docs"
```

@qualified_name(name: str) -> str:
  = `prefix + ":" + canonical(name)`

@test: qualified_name:
  > case: uses_runtime_environment:
    ```
    assert qualified_name(" API ") == "docs:api"
    ```
````

Keep setup local to a case unless it is genuinely part of the program's public
runtime environment.

The validation runner compiles the program but does not execute top-level main
statements as setup. Put shared fixtures in the prelude or initialize them
inside each case; a top-level Kedi variable initialization has not run merely
because the test suite was loaded.

## Assertions in Python

Any exception fails the case. Python `assert` is conventional because it keeps
the expected condition visible:

````kedi
@ratio(total: int, count: int) -> float:
  = `total / count`

@test: ratio:
  > case: computes_fraction:
    ```
    value = ratio(3, 2)
    assert isinstance(value, float)
    assert value == 1.5
    ```
````

Kedi records the exception message in the result. A failed case does not stop
the remaining cases from running.

## Test Execution Order

Suites and their cases run in source order. Each case uses the same compiled
runtime, so mutations to shared Python objects can leak into later cases. Avoid
order-dependent tests; construct fresh mutable data inside each case.

## Test Failures

`kedi program.kedi --test` prints one line per case:

```text
Tests:
- normalize_tag::lowercases: OK
- normalize_tag::strips_whitespace: OK
```

The command exits with status `1` when one or more cases fail, or when parsing,
compilation, code generation, or optimized-prompt loading fails.

## CLI Test Runs

Run all test suites in a file:

```bash
kedi program.kedi --test
```

Backend options still apply because a tested procedure may call a model:

```bash
kedi program.kedi --test \
  --adapter pydantic \
  --adapter-model openai:gpt-6-luna
```

`--test` cannot be combined with `-c/--command`; validations require a source
file. Extra CLI arguments remain available through the runtime `args` binding.

## Cache Behavior

Before tests run, Kedi generates any `> auto:` procedures. The default
`program.cache.kedi` is reused. `--no-cache` disables reuse and removes the
temporary codegen cache after the command:

```bash
kedi program.kedi --test --no-cache
```

This flag does **not** disable response caching configured through the Python
API and does **not** ignore `program.kedi.optimized.json`. File-backed tests
load valid optimized prompt prefixes when that artifact exists, and invalid
optimization JSON fails loudly.
