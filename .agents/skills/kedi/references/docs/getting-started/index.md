# Learn Kedi { #start-with-kedi }

Start with [Installation](installation.md), then work through
[Your First Program](first-program.md). You will distinguish values computed by
Python from values requested from a model, preserve their types across procedure
calls, and use the result in ordinary control flow. Agents and tools come later,
when the task needs them.

## What You Will Build

The first workflow accepts a topic from the command line, asks a model for a
short structured brief, and returns the captured text:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna

@brief(topic: str) -> str:
  >> A two-sentence brief for a software engineer about <topic> is [summary: str].
  = <summary>

= <brief(`args.topic`)>
```

This is intentionally small, but it demonstrates the main execution model:

1. `args.topic` is passed to `brief` as a native Python string.
2. `<topic>` substitutes that runtime value into the prompt.
3. `[summary: str]` captures one typed field from the model response.
4. `= <summary>` returns its rendered text.

## Prerequisites

- Python 3.11 or newer;
- a Kedi installation;
- credentials required by the selected model provider;
- optionally, an agent harness installation when using Codex, Claude, or ACP.

Parsing does not contact a provider. Use it to validate syntax before setting up
credentials.

For the explicit OpenAI model above, follow the provider SDK and
`OPENAI_API_KEY` setup in [Installation](installation.md). Do not assume the
CLI's default model matches the credentials you configured.

## The Smallest Useful Program

For a fixed prompt with one result, capture the output explicitly:

This fragment inherits the adapter and model directives from the program above.

```kedi
>> The importance of idempotency, explained in one paragraph, is [answer: str].
= <answer>
```

The typed field keeps the response and makes its contract visible to the
adapter. A plain template without an output field does **not** keep the response:

```kedi
>> In one paragraph, idempotency matters because
```

That form is appropriate only when the call's side effects or trace matter and
the text is intentionally discarded. It is usually the wrong choice for a
user-facing answer.

## Run, Parse, and Validate

Save the first example as `brief.kedi`, then parse it:

```bash
kedi parse brief.kedi
```

Run it with an application argument:

```bash
kedi brief.kedi --topic "distributed locks"
```

Unknown CLI options after the source file are normalized into the reserved
`args` object. For example, `--dry-run` becomes `args.dry_run`. The `args`
binding cannot be assigned from Kedi or embedded Python.

You can also parse inline source without any provider setup:

```bash
kedi -p -c "= ready"
```

Parse-only mode checks syntax and structural rules. Compilation and execution
can additionally fail on type resolution, backend capability validation,
provider errors, or approval decisions.

For example, `kedi -p -c "@broken("` deliberately fails parsing. By contrast,
a valid program can still fail later if an input is missing or a provider rejects
the request. Keep these failure stages distinct when diagnosing a program.

## Where to Go Next

- [First Program](first-program.md) expands the example and compares structured
  capture with raw capture.
- [Projects and Execution](projects-and-execution.md) explains source loading,
  adapters, and generated artifacts.
- [Templates and Invokes](../core-language/templates-and-invokes.md) defines
  the exact `>>` and `<<` semantics.
- [Outputs, Initialization, and Assignment](../core-language/outputs-and-assignments.md)
  explains output capture, `=` initialization, and `:=` assignment.

## In This Section

**Pages**

- [Installation](installation.md)
- [First Program](first-program.md)
- [Projects and Execution](projects-and-execution.md)
- [Find a Reference](../reference/index.md)
