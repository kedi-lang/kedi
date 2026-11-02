# Execution and Dataflow

Kedi executes source in order while preserving lexical declaration state.
Concurrency changes scheduling, not the declared value and type contracts.
It does not promise identical stochastic outputs or side-effect ordering.

## Compilation Phases

Compilation performs these conceptual phases:

- resolve source documents and module paths;
- parse syntax into an AST with source spans;
- validate declarations, directives, package/profile graphs, and types;
- execute the prelude and build the type environment;
- compile custom types and procedures with captured lexical state;
- construct the runtime's adapter, tool, MCP, approval, and execution engine.

Some failures are necessarily deferred. A dynamic Python type, backend
expression, model connection, or tool call cannot be fully validated until
execution reaches it.

## Source Order

Initializations, assignments, imports, directives, and executable statements
take effect where they appear:

```kedi
[stage] = draft

@before() -> str:
  = <stage>

[stage] = final

@after() -> str:
  = <stage>
```

Procedures capture top-level agent/declaration state at definition, while value
lookup follows the runtime's lexical environment. Use distinct names when an
intentional snapshot should be obvious.

## Value Environments

Lookup combines the current procedure frame, lexical closures, top-level
globals, imported exports, prelude names, initial Python API globals, generated
types, and compiled procedures according to their precedence.

Procedure-local names do not leak. A Python block receives visible Kedi values
as globals, but new Python locals do not become Kedi bindings.

## Native and Rendered Values

```kedi
[count: int] = `3`
[message] = count\=<count>
```

`count` is a native integer; `message` is rendered text. A sole Python
expression or procedure call on an initialization, assignment, or return
boundary can preserve its native result. Mixed literal text always renders.

This distinction drives dependency tracking and type validation. Avoid
stringifying a collection merely to pass it to another typed procedure.

## Model Templates

One `>>` block is one adapter call:

```kedi
>> Incident: <incident>.
The affected service is [service: str].
The severity is [severity: str].
```

Kedi prepares substitutions, builds a combined output schema, invokes the
active adapter, validates returned fields, and publishes them into the current
environment.

Continuation lines in the same block cannot read outputs from that same call.
Start another `>>` block for a dependency:

```kedi
>> Incident <incident> affects [service: str].
>> The recommended owner for <service> is [owner: str].
```

## Procedure Frames and Joins

Each invocation gets a frame for parameters, locals, nested declarations,
agent state, and scheduled work. A procedure return is a join point: pending
model results required by the return are forced and the native result is checked
against the return annotation.

When a procedure exits, Kedi restores outer tool/profile frames. Scheduled work
is drained before an exception leaves the frame so background model failures
cannot disappear.

## Failure Propagation

Parsing and structural failures prevent execution. Runtime type, Python,
approval, tool, adapter, and model errors stop the affected flow and acquire
Kedi trace frames as they propagate.

In parallel mode, all already-scheduled jobs are drained. The first failure is
raised and additional concurrent failures are logged. Unconsumed calls still
run and can fail; discarding a model response does not discard its error.
