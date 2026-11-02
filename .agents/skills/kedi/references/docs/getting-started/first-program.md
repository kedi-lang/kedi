# Your First Kedi Program

Your first program asks a model to complete a sentence, captures its answers as
typed values, and uses those values in its output. No Python block or SDK call
is needed.

Complete [Installation](installation.md), including OpenAI provider support and
`OPENAI_API_KEY`, before running it. This program makes a real model request.

## Write Your First Template

Create `film.kedi`:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna

[film: str] = Spirited Away

>> The film <film> was directed by [director: str] and first released in [year: int].

= <film> (<year>), directed by <director>.
```

```bash
kedi film.kedi
```

In a uv project, use `uv run kedi film.kedi`. A typical result is:

```text
Spirited Away (2001), directed by Hayao Miyazaki.
```

The program supplies the film title, not the answers. The model generates the
director and release year; wording and factual correctness are not guaranteed.
This example does not look up a film database.

- `[film: str] = ...` binds the input title without calling a model.
- `>>` opens a natural-language template. `<film>` substitutes the existing title.
- `[director: str]` and `[year: int]` declare the answers to generate and their
  types. Both belong to the same template, not two separate prompts.
- `= ...` returns the completed output, which the CLI prints. Reading the
  captured values resolves the template's deferred model request.

The completed template forms a sentence, while its captures become program
values: `director` is a string and `year` is an integer. Kedi validates their
types; that validation does not establish that the facts are true.

To choose the film at runtime, replace the input binding with:

```kedi
[film: str] = `args.film`
```

Then run `kedi film.kedi --film "My Neighbor Totoro"`. The template stays the
same; only its input changes.

<a id="create-a-kedi-file"></a>

## Grow the Program with a Typed Procedure

Once the template/capture distinction is familiar, a procedure can package a
model interaction for reuse. This second example captures a structured review
instead of individual film facts. Create `review.kedi`:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna

~Review(decision: Literal["approve", "revise"], summary: str)

@review_change(title: str, diff_summary: str) -> Review:
  >> For change <title> with diff summary <diff_summary>, the review result is [review: Review].
  = `review`

= `review_change(args.title, args.diff_summary).model_dump_json()`
```

Run it:

```bash
kedi review.kedi \
  --title "Reject unsafe paths" \
  --diff-summary "Adds containment checks before file access"
```

## Add Inputs

The output is a JSON object with `decision` and `summary`, for example:

```json
{"decision":"revise","summary":"Show tests for traversal and symlink escape cases."}
```

This is illustrative, not an exact expected answer. The model sees only the
title and summary, not the code or test results. Its recommendation is advisory:
schema validation limits the decision to two values but does not prove that the
change is safe or grant permission to merge it.

`title` and `diff_summary` are typed procedure parameters. The final call uses
a single-backtick Python expression:

```kedi
[review: Review] = `review_change(args.title, args.diff_summary)`
```

This passes native strings and preserves the native `Review` return. An angle
call renders its result to text, so it is appropriate for procedures returning
`str`, not for carrying a `Review` object through the dataflow.

## Write a Template

Continuation rows after `>>` are joined with newlines and sent as one model
request. This example needs only one row:

```kedi
>> For change <title> with diff summary <diff_summary>, the review result is [review: Review].
```

`<title>` and `<diff_summary>` are substitutions. They read existing values;
they do not ask the model to generate anything. Use substitutions for runtime
facts, user input, prior procedure results, or deterministic Python values.

## Capture a Typed Output

`[review: Review]` is an output capture. The selected adapter receives a schema
derived from the Kedi type and must return a matching object. Capture output
when downstream logic needs typed fields or when the response must be validated.

Kedi also exposes raw capture for deliberately unstructured provider text:

```kedi
@review_change(title: str, diff_summary: str) -> str:
  [review] << Review of <title> with diff summary <diff_summary>:
  = <review>
```

Do not put output fields inside a raw `<<` prompt. Raw captures always produce a
string; types other than `str` are rejected. Prefer the typed version above for
normal application dataflow, including typed `str` results.

## Return the Result

Inside the typed version, ``= `review` `` returns the native `Review` model.
The top-level expression serializes it deliberately:

```kedi
= `review_change(args.title, args.diff_summary).model_dump_json()`
```

Use a native return when Python or another Kedi procedure needs the object. Use
a rendered return when the program's final output is text.

## Use the Typed Result

Replace the final expression of `review.kedi` with this block; keep its directives,
type and procedure definitions above it:

```kedi
[review: Review] = `review_change(args.title, args.diff_summary)`
[next_step: str] = Request another revision

> if: `review.decision == "approve"`:
  [next_step] := Queue for human review

= <next_step>: <`review.summary`>
```

This branch performs a Python comparison, not another model judgment. The
trailing `:` makes that explicit. `=` initializes `next_step`; `:=` updates its
existing outer binding from the branch's child scope. Neither branch publishes
or merges anything. For natural-language conditions and loop/map dataflow, see
[Control Flow](../core-language/control-flow.md) and
[Loops and Map](../core-language/loops-and-map.md).

## Pass Command-Line Arguments

Application flags belong after the Kedi source:

```bash
kedi review.kedi --title "Update parser" --diff-summary "Adds package syntax"
```

Dashed names become underscore attributes, so `--diff-summary` is available as
`args.diff_summary`. Flags without values become booleans. Kedi's own options,
such as `--adapter` and `--test`, are parsed by the CLI rather than exposed as
application arguments.

## Parse Before Running

```bash
kedi parse review.kedi
```

Parsing catches malformed syntax, duplicate selective imports, invalid
directives, and other structural errors. It cannot prove that provider
credentials exist or that a runtime-computed type is valid. Those checks happen
during compilation or execution.

## Continue the Program

- Add deterministic checks with [Testing](../evals-and-optimization/test-blocks.md).
- Move reusable procedures into [Modules](../modules-and-packaging/modules.md).
- Call the same runtime from [Python](../python-api/embedding.md).
- Add [Tools](../agentic-engineering/tools-and-use.md) only when external evidence
  or actions are needed; a template alone does not fetch files or browse the web.
