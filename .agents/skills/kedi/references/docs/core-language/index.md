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

```kedi
>> <notes> contains [count: int] action items.
```

This performs a model call and captures `count`. By contrast:

```kedi
[count: int] = `len(notes.splitlines())`
```

is deterministic variable initialization and does not contact a model. Use a template
when the transformation needs model judgement; use Python when the answer is
deterministic and locally computable.

## Dataflow at a Glance

Angle brackets read values; square brackets introduce output fields or binding
targets:

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

- [Source Structure](source-structure.md)
- [Multiline Syntax and Escaping](multiline-and-escaping.md)
- [Syntax Index](../reference/syntax.md)
- [Directive Index](../reference/directives.md)

**Values and Types**

- [Types and Constraints](types.md)
- [Outputs and Bindings](outputs-and-assignments.md)
- [Scopes and Binding Lifetime](scopes-and-bindings.md)

**Model Interaction**

- [Templates and Invokes](templates-and-invokes.md)
- [Substitutions and Calls](substitutions-and-calls.md)

**Procedures**

- [Procedures and Closures](procedures.md)
- [Parameters and Returns](parameters-and-returns.md)

**Control Flow**

- [Branches and Claims](control-flow.md)
- [Loops and Map](loops-and-map.md)

**Python Interop**

- [Python Interop](../python-interop/index.md)
- [Inline Expressions](../python-interop/inline-expressions.md)
- [Python Blocks](../python-interop/python-blocks.md)
- [Prelude and Scope](../python-interop/prelude-globals-and-scope.md)

**Pages**

- [Coverage and Limitations](../reference/semantic-coverage.md)
