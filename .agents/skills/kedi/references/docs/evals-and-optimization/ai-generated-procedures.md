# AI-Generated Procedures

## Declare an `> auto:` Block

`> auto:` defines a procedure by contract instead of a handwritten body:

```kedi
@unique_in_order(items: list[str]) -> list[str]:
  > auto:
    Remove duplicate strings while preserving the first occurrence order.
    Return an empty list for an empty input.
```

The indented lines form one natural-language specification. The old implicit
unknown-`>` form is not accepted; use the explicit `> auto:` directive.

`> auto:` accepts indented specification lines without `>>`; these describe
generated code rather than an ordinary executable template. In contrast,
`> optimize:` requires a template starting with `>>`. Bare prompt beginnings
remain invalid in ordinary procedure bodies.

## Procedure Signatures as Contracts

Parameter names, type annotations, and the return type constrain generated
tests and code:

```kedi
~Summary(title: str, bullets: list[str])

@build_summary(title: str, facts: list[str]) -> Summary:
  > auto:
    Keep the title unchanged and convert each non-empty fact into one bullet.
```

Use precise native types. An omitted parameter or return annotation defaults to
`str`, which may tell the generator the wrong contract.

## Available Context

Code generation receives Python-shaped context derived from:

- the top-level Python prelude;
- Kedi custom type definitions;
- signatures and procedure docstrings for non-generated procedures.

Generated procedures are excluded from the available procedure-signature list
to avoid presenting unresolved implementations as dependencies.

## Generate Test Cases

The codegen agent first emits pytest-style Python tests. Kedi translates them
into an `@test:` suite, writes them into the generated cache block, and verifies
that the merged source parses.

Generated tests are provisional acceptance criteria, not a replacement for
handwritten regression tests. Review important contracts and add explicit
`@test:` cases to the source.

Module-level helpers and setup are preserved inside each generated test case.
Tests must be synchronous, undecorated, zero-argument functions; pytest fixtures
and parametrization are not supported. Invalid Python or unsupported test shapes
are reported back to the generator for another attempt.

## Generate an Implementation

After tests compile, the agent emits a Python function **body**, not a `def`
wrapper. Kedi translates that body into the original Kedi procedure signature,
merges it with the generated tests, parses the result, and runs all tests.

A returned `def` wrapper or otherwise untranslatable body is rejected and fed
back to the next attempt.

## Retry Stages

`--codegen-retries N` creates `N` escalating stages:

- stage 1 allows one test attempt and one implementation attempt;
- stage 2 allows up to two of each;
- stage N allows up to N of each.

Compilation and test errors are passed back into later attempts. The default is
five stages; total possible calls are therefore greater than five.

## Passing Criteria

An implementation is accepted only when:

1. generated tests translate into valid Kedi;
2. the implementation translates into valid Kedi;
3. merged source and cache parse successfully;
4. all validation tests pass.

If every stage fails, Kedi removes that procedure's failed cache block and the
command fails.

## Cached Implementations

For `program.kedi`, generated code is stored in `program.cache.kedi`. Each
procedure has comment-delimited markers, so regenerating one procedure does not
replace unrelated generated procedures.

Tests and candidate implementations are written provisionally. A block becomes
reusable only after validation succeeds and Kedi writes its completion marker.
Tests-only, invalid, and older unmarked cache blocks are regenerated rather than
treated as finished implementations. This means existing unmarked caches need
one successful regeneration after upgrading.

Exceptions and cancellation remove the current procedure's incomplete block,
without removing other procedures. If the process is terminated before cleanup
can run, the provisional block may remain, but it is not reused as a completed
cache entry.

The cache is reused by procedure name:

```bash
kedi program.kedi
```

Force generation without retaining or reusing the cache:

```bash
kedi program.kedi --no-cache
```

Kedi still creates a temporary cache because generated tests and implementation
must be merged for execution; it removes that procedure's block afterward.

## Codegen Models and Agents

The production codegen agent is `pydantic_ai`:

```bash
kedi program.kedi \
  --codegen-agent pydantic_ai \
  --codegen-model openai/gpt-6-luna \
  --codegen-retries 5
```

The `mock` agent exists for deterministic framework tests:

```bash
kedi program.kedi --codegen-agent mock
```

It generates generic placeholder behavior and is not a production
implementation strategy.

## Failure Modes

Generation fails when the model is unavailable, the contract is underspecified,
tests cannot be translated, implementation translation fails, merged source is
invalid, or generated tests do not pass.

Passing generated tests does not prove correctness beyond those tests. The same
model can misunderstand both the contract and its tests. Keep independent
handwritten boundary and failure cases, and review generated code before giving
it access to sensitive resources.

Cached code is executable Kedi/Python. Treat it as generated source: review it,
decide deliberately whether to commit it, and regenerate after changing a
procedure's contract. The cache lookup is keyed by procedure name, not a digest
of the specification.
