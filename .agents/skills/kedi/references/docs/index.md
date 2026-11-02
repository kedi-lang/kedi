---
manual_home: true
hide:
  - toc
---

<span id="the-kedi-language" hidden></span>

# The Kedi language { #kedi-programming-language }

A language for typed programs with natural language. Learn the syntax, follow
the runtime, and find the rules behind your programs.

<div class="kedi-manual-intro" markdown="1">

[Start with installation](getting-started/installation.md){ .kedi-start-link }

</div>

<!-- manual-directory -->

<aside class="kedi-manual-specimen" markdown="1">

## A template, at a glance

With an adapter and model configured:

```kedi
>> The capital of France is [city: str].

= <city>
```

<div class="kedi-example-output">
<span>Illustrative output</span>
<samp>Paris</samp>
</div>

## Read the notation

| Syntax | Meaning |
| --- | --- |
| `>>` | Begin a model template |
| `[city: str]` | Capture a typed output |
| `<city>` | Substitute a value |

## Keep close

- [Syntax index](reference/syntax.md)
- [Directive index](reference/directives.md)
- [Errors and diagnostics](reference/diagnostics-and-troubleshooting.md)

</aside>

<div class="kedi-manual-overview" markdown="1">

## Why Typed LLM Programs

Prompts are useful for fuzzy transformations; Python is useful for deterministic
logic. Kedi keeps both in one dataflow without pretending they are the same
thing.

This definition is a fragment: it does not call a model until the procedure is
invoked. The [first program](getting-started/first-program.md) supplies model
selection, command-line inputs, and a complete executable entry point.

```kedi
~Ticket(category: str, urgency: int)

@classify(message: str) -> Ticket:
  >> The classification of support message <message> is [ticket: Ticket].
  = `ticket`
```

`<message>` is a substitution: Kedi renders an existing value into the prompt.
`[ticket: Ticket]` is an output capture: the adapter asks the model for a value
matching the generated schema. ``= `ticket` `` returns the native `Ticket`
object. Writing `= <ticket>` instead would stringify it.

Use output capture whenever a model result participates in the program's
dataflow, including plain text. A `str` capture still gives the adapter an
explicit output contract:

```kedi
@summarize(message: str) -> str:
  >> A two-sentence summary of <message> is [summary: str].
  = <summary>
```

Raw capture is an escape hatch for deliberately unstructured text, not the
default form of a Kedi model call.

Typed does not mean factually correct. Validation checks the declared contract;
it cannot prove that a model's answer is true. Deterministic checks, evidence,
and [evaluation](evals-and-optimization/index.md) address different questions.

## Choose a Starting Point

- New to Kedi: begin with [Learn Kedi](getting-started/index.md).
- Learning the DSL: use the [Language Reference](core-language/index.md).
- Embedding Kedi in Python: use the [Python API](python-api/index.md).
- Building tool-using or delegated agents: use
  [Agents and Orchestration](agentic-engineering/index.md).
- Looking up exact syntax or behavior: use the [Reference](reference/index.md).

After the first program, use [Projects and Execution](getting-started/projects-and-execution.md)
to split it across files, then choose a task from the [Cookbook](examples/index.md).
You do not need tools, profiles, or delegated agents to use typed templates.

## Language, Python API, and Model Integrations { #core-language-python-api-and-agent-adapters }

The same runtime semantics are available through two authoring surfaces:

- `.kedi` files are best for workflows where prompts, types, procedures, tests,
  and profiles should be visible together.
- `@kedi.query` and `@kedi.bind` are best when Python owns the public function
  signature and Kedi provides the implementation.

Adapters are boundary implementations, not alternative Kedi dialects. The
language semantics stay stable, while capability validation reports whether a
selected backend supports structured output, tools, MCP, approvals, or
subagents.

## Documentation Conventions

Code blocks marked `kedi` are Kedi source. Backticks have two different roles:

- single backticks, such as `` `items` ``, evaluate a Python expression;
- triple backtick blocks execute multiline Python.

The documentation distinguishes:

- **substitution** (`<value>`), which reads and renders a value;
- **output capture** (`[value: Type]`), which asks the model to produce a value;
- **native return** (``= `value` ``), which preserves the Python object;
- **rendered return** (`= <value>`), which returns text.

Those choices are called out in examples because replacing one with another can
change types, validation, or whether a model response is retained.

</div>
