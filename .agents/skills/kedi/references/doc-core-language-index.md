> Offline source: `kedi-docs-main/docs/core-language/index.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Program Anatomy](#program-anatomy)
- [Prompt and Native Execution](#prompt-and-native-execution)
- [Dataflow at a Glance](#dataflow-at-a-glance)
- [Types and Structured Results](#types-and-structured-results)
- [Procedures and Scope](#procedures-and-scope)
- [Complete Language Map](#complete-language-map)
- [In This Section](#in-this-section)

# Language Reference { #core-language }

## Program Anatomy

A Kedi program is an ordered sequence of declarations and executable
statements. The public language surface includes:

- imports, exports, and package metadata;
- custom types and procedures;
- template and raw model calls;
- variable initialization, assignment, and returns;
- deterministic and model-classified conditionals, conditional loops, and sequential loops;
- embedded Python;
- model, profile, tool, MCP, skill, and subagent directives;
- test, eval, optimization, and generated-procedure blocks.

Indentation defines scope. There is no brace-delimited alternative.

## Prompt and Native Execution

Kedi uses explicit syntax for the model boundary:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
>> <notes> contains [count: int] action items.
```

This performs a model call and captures `count`. By contrast:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
[count: int] = `len(notes.splitlines())`
```

is deterministic variable initialization and does not contact a model. Use a template
when the transformation needs model judgement; use Python when the answer is
deterministic and locally computable.

## Dataflow at a Glance

Angle brackets read values; square brackets introduce output fields or binding
targets:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
[topic] = API compatibility
>> A brief explanation of <topic> is [summary: str].
= <summary>
```

`<topic>` is an R-value substitution. `[summary: str]` is an L-value output
capture. The same bracket syntax can appear on the left of `=` for native
variable initialization, where no model is involved. `:=` assigns to an
existing binding.

## Types and Structured Results

Types can annotate outputs, variable initializations, parameters, returns, and
custom type fields. An assignment inherits the target binding's existing type
contract. Kedi resolves built-in names, Python type expressions, and custom
types, then validates values at runtime. Adapters receive structured schemas
when they support them.

## Procedures and Scope

Procedures create reusable lexical scopes:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@normalize(value: str, lower: bool = `True`) -> str:
  = `value.strip().lower() if lower else value.strip()`
```

Parameters and local initializations do not leak to callers. Explicit
assignments can update a visible outer binding. Top-level agent state is
captured by following procedures; directives inside a procedure affect only
the remainder of that procedure's lexical block.

## Complete Language Map

Follow the topic groups [below](#in-this-section) for the language reading order.

Modules, agent directives, tests, and Python embedding are documented in their
own sections because each has independent scoping and runtime rules.

## In This Section

**Source and Syntax**

- [Source Structure](doc-core-language-source-structure.md)
- [Multiline Syntax and Escaping](doc-core-language-multiline-and-escaping.md)
- [Syntax Index](doc-reference-syntax.md)
- [Directive Index](doc-reference-directives.md)

**Values and Types**

- [Types and Constraints](doc-core-language-types.md)
- [Outputs and Bindings](doc-core-language-outputs-and-assignments.md)
- [Scopes and Binding Lifetime](doc-core-language-scopes-and-bindings.md)

**Model Interaction**

- [Templates and Invokes](doc-core-language-templates-and-invokes.md)
- [Substitutions and Calls](doc-core-language-substitutions-and-calls.md)

**Procedures**

- [Procedures and Closures](doc-core-language-procedures.md)
- [Parameters and Returns](doc-core-language-parameters-and-returns.md)

**Control Flow**

- [Branches and Claims](doc-core-language-control-flow.md)
- [Loops and Map](doc-core-language-loops-and-map.md)

**Python Interop**

- [Python Interop](doc-python-interop-index.md)
- [Inline Expressions](doc-python-interop-inline-expressions.md)
- [Python Blocks](doc-python-interop-python-blocks.md)
- [Prelude and Scope](doc-python-interop-prelude-globals-and-scope.md)

**Pages**

- [Coverage and Limitations](doc-reference-semantic-coverage.md)
