> Offline source: `kedi-docs-main/docs/python-api/caching-runtime-and-executors.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Inspect Cache State](#inspect-cache-state)
- [Clear Caches](#clear-caches)
- [`KediRuntime`](#kediruntime)
- [Compile Programs](#compile-programs)
- [Runtime Input, Output, and Context](#runtime-input-output-and-context)
- [Low-Level Expressions](#low-level-expressions)
- [Custom Executors](#custom-executors)
- [Default Executor](#default-executor)
- [Debug Exporters](#debug-exporters)
- [Subagent State Persistence](#subagent-state-persistence)

# Caching, Runtime, and Executors


## Inspect Cache State

See [Inspect Cache State](doc-python-api-cache-control.md).

## Clear Caches

See [Clear Caches](doc-python-api-cache-control.md).

## `KediRuntime`

`KediRuntime` is the compiled execution container:

```python
from kedi import KediRuntime
```

Most applications should use `query` or `bind`. Use the runtime directly when
building an embedding, custom compiler flow, executor integration, or engine.

Important public methods include:

- `run_main()` to execute the compiled top-level program and return a forced
  result;
- `set_initial_globals(mapping)` to seed runtime values before execution;
- `procedure` and `main` decorators for low-level program construction;
- `m(expressions)` and `invoke(expressions, capture=...)` inside an active Kedi
  procedure context;
- `drain()` to await every scheduled job;
- `current_trace_frames()` and `build_execution_error(...)` for diagnostics.


## Compile Programs

Parse and compile explicitly:

```python
from kedi.lang import compile_program, parse_program

source = """
@greet(name: str) -> str:
  = Hello, <name>

= <greet(World)>
"""

program = parse_program(source, source_path="<memory>")
runtime = compile_program(program)
result = runtime.run_main()
```

A model adapter is optional only when the program never executes a template or
raw invoke. For model-backed programs, pass an `AgentAdapter`.


## Runtime Input, Output, and Context

Seed native values:

```python
program = parse_program("= `customer_id`")
runtime = compile_program(
    program,
    runtime_globals={"customer_id": "cus_123"},
)
assert runtime.run_main() == "cus_123"
```

`compile_program(...)` accepts:

- `adapter`;
- `executor`;
- `runtime_globals`;
- `engine`;
- `default_agent_profile`;
- subagent depth, concurrency, timeout, usage limits, and state path.

The runtime owns procedure/global environments and uses `ContextVar` for the
current invocation, trace stack, and subagent stack.

Runtime failures are raised as `KediExecutionError`. Catch the exception when
embedding Kedi, call `exc.render()` for the formatted Kedi traceback, or inspect
`frames`, `python_traces`, and `original` programmatically. See
[Errors, Frames, and Tracebacks](doc-runtime-errors-and-debugging.md) for the
complete error model.


## Low-Level Expressions

The root package exports constructors used with `runtime.m(...)`:

```python
from kedi import c, i, o

expressions = [
    "Find the capital of ",
    i("country"),
    ": ",
    o("capital", str),
]
```

`i(name)` reads an input, `o(name, type)` declares an output, and
`c(procedure, *args)` invokes a Kedi procedure. `runtime.m(...)` and
`runtime.invoke(...)` require an active procedure environment; calling them
arbitrarily outside Kedi execution raises.


## Custom Executors

See [Custom Executors](doc-python-api-executors.md).

## Default Executor

See [Default Executor](doc-python-api-executors.md).

## Debug Exporters

See [Debug Exporters](doc-python-api-executors.md).

## Subagent State Persistence

Low-level compilation can configure subagents:

```python
from kedi import SubagentUsageLimits

limits = SubagentUsageLimits(request_limit=8, tool_calls_limit=16)
runtime = compile_program(
    program,
    adapter=adapter,
    subagent_max_depth=5,
    subagent_max_concurrency=4,
    subagent_timeout_seconds=120.0,
    subagent_usage_limits=limits,
    subagent_state_path=".kedi/subagents.json",
)
```

The state file belongs to the subagent coordinator and is separate from
response, codegen, and optimization caches. Pending or running work restored
after process loss is marked interrupted rather than falsely reported as
completed.

For a complete program, defined tools, execution context, and cleanup, see
[Embedding Subagent Programs in Python](doc-agentic-engineering-subagent-python.md).
