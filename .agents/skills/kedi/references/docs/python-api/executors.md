# Executors and Debug Exporters

An executor owns embedded-Python execution. Replacing it must preserve environment, return, and error contracts; an interface stub is not a secure sandbox.

## Custom Executors

Implement the `Executor` protocol from `kedi`. This interface sketch is not an
implementation and must not be used as a sandbox:

```python
from typing import Any, Callable
from kedi import Executor, ExecutorDebugExporter, KediRuntime


class RestrictedExecutor:
    def set_debug_exporter(
        self, exporter: ExecutorDebugExporter | None
    ) -> None: ...

    def evaluate_inline(
        self, rt: KediRuntime, code: str, local_env: dict[str, Any]
    ) -> Any: ...

    def execute_block(
        self, rt: KediRuntime, code: str, local_env: dict[str, Any]
    ) -> Any: ...

    def execute_side_effects(
        self,
        rt: KediRuntime,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None: ...

    def execute_prelude(
        self, rt: KediRuntime, code: str
    ) -> dict[str, Any]: ...

    def create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable[..., Any],
        defaults: dict[str, Any] | None = None,
    ) -> Callable[..., Any]: ...

    def evaluate_type_expression(
        self, rt: KediRuntime, code: str, env: dict[str, Any]
    ) -> Any: ...
```

The protocol is runtime-checkable. A custom executor must preserve Kedi's
environment and return semantics, not merely evaluate isolated strings.

Incremental sessions additionally need `execute_prelude_incremental(rt, code,
env)` to extend an existing Python environment and return its updated mapping.
The `env` argument is an existing mapping or `None`. An executor that only
implements `Executor` cannot run incremental preludes; the runtime raises an
error rather than silently restarting the environment. `DefaultExecutor`
supports this operation.


## Default Executor

`DefaultExecutor` uses Python `eval` and `exec`:

```python
import asyncio
from kedi import DefaultExecutor
from kedi.lang import compile_program, parse_program

program = parse_program("= `2 + 3`")
runtime = compile_program(program, executor=DefaultExecutor())
try:
    assert runtime.run_main() == 5
finally:
    asyncio.run(runtime.aclose())
```

It is **not sandboxed**. Embedded Python has the host process's authority,
imports, filesystem access, network access, and credentials. Use a specialized
executor and operating-system isolation for untrusted Kedi source.


## Debug Exporters

Attach a Markdown event exporter when compiling the `program` above:

```python
from kedi import DefaultExecutor, MarkdownDebugExporter

executor = DefaultExecutor(
    debug_exporter=MarkdownDebugExporter("runtime-debug.md")
)
runtime = compile_program(program, executor=executor)
```

Events include executor step, code, inputs, local environment, outputs, and
errors. Sanitization makes values printable; it does **not redact secrets**.
Debug exports can contain prompts, credentials, user data, and tool results.
Store and share them accordingly.

`default_debug_export_path("program.kedi")` creates a timestamped path in the
current working directory.
