# Embedding Kedi in Python

## Choose the owner

| Need | Entry point |
| --- | --- |
| A Kedi procedure body inline in Python | `@kedi.query` |
| A full `.kedi` file with Python call signature | `@kedi.bind(file=...)` |
| Explicit source/adapter/runtime lifecycle | `parse_program` then `compile_program` |
| Incremental declarations and values | `kedi.interactive()` |
| Shared conversation/artifact lifetime across calls | `kedi.session()`; not persistent Kedi variables |

## Query and bind

A query docstring starts with a standalone `kedi` line; its content is a **procedure body**, not a whole program wrapper. A decorated Python body is ignored; use `...`. Use the Kedi native return for objects, e.g. a sole backtick expression. Python annotations do not coerce a rendered result into a model/int. `async def` gets an async wrapper that offloads blocking work.

Bind files are full programs and may define types/profiles/imports/procedures. A relative `file=` resolves from the defining Python source file, falling back to CWD where that identity is unavailable. The file main return is the call result. `reload=True` rereads each call; the initial decoration still requires valid source. Hash-based parse caching never makes invalid new source fall back to old code.

`query`/`bind` accept backend/profile/tools/env/MCP/approval/skills/cache options; read [public parameters](doc-python-api-public-parameters.md) before passing anything else. Use `adapter=` for frameworks, `agent=` for harnesses, never both. Configured/local `env` can override function arguments; avoid accidental collisions. Arguments cannot use registered tool names. `tools=` makes callables available; `> use:` exposes the intended callable in source.

Run [python_api.py](../assets/examples/python_api.py) for a deterministic query/bind example. Read [query](doc-python-api-query.md), [bind](doc-python-api-bind.md), and [types/tools](doc-python-api-types-and-tools.md) for dynamic output types and `@kedi.type` registration.

## Low-level lifecycle

```python
import asyncio
from pathlib import Path
from kedi.lang import compile_program, parse_program

path = Path("program.kedi").resolve()
runtime = compile_program(parse_program(path.read_text(encoding="utf-8"), source_path=str(path)))
try:
    value = runtime.run_main()
finally:
    asyncio.run(runtime.aclose())
```

This minimal loader is appropriate for ordinary source. File execution with generated/optimized companions should follow Kedi's source-loading/CLI pipeline; `parse_program` alone does not perform every file-run preprocessing step. Pass an explicit adapter and `runtime_globals` when the host supplies models/inputs. Preserve source paths so imports/resources and tracebacks work.

In an async host, keep reusable async clients on one owner loop. Use `use_async_owner_loop()`, run `runtime.run_main` via `asyncio.to_thread`, and `await runtime.aclose()` on that loop; do not nest `asyncio.run()` or reuse loop-bound clients across separate runs. See [embedding](doc-python-api-embedding.md). Closing a runtime does not transfer ownership of every caller-created adapter/client; follow its explicit integration contract.

## Configuration, sessions and executors

`configure` sets defaults; `context` temporarily merges/restores them. Neither is a live session. Low-level executors and subagent limits belong to `compile_program`/interactive runtime parameters, not arbitrary adapter kwargs on configure. Read [configuration](doc-python-api-configuration-and-context.md).

Interactive `execute` returns `None`; use `> show:` at the root, and normal returns in procedures. Fragments execute once and preserve earlier completed side effects on failure. No concurrent/reentrant execution. Strict dump/load snapshots reject unsupported state, imports/preludes and live objects instead of replaying or silently dropping them. See [interactive execution](doc-runtime-interactive-execution.md).

`DefaultExecutor` uses Python eval/exec with host authority. A protocol stub is not a sandbox. Read [executors](doc-python-api-executors.md) before implementing one; incremental preludes require an additional supported operation.

## Native adapter APIs

`PydanticAdapter` also exposes Pydantic AI native run/iter/stream calls; these retain Kedi profile/policy behavior. Native framework inputs are not the same as `produce(template=..., output_schema=...)`. The structured low-level schema maps names to Python types, not JSON dictionaries. `invoke` returns text; `produce` returns typed output.

Do not invent asynchronous, tool, approval, stream or persistence methods from another framework. For external deferred approval, retain exact ToolSpecs and use Kedi's argument-validating conversion helpers. See [Pydantic adapter](doc-agent-adapters-pydantic-ai.md), [LangChain](doc-agent-adapters-langchain.md), [custom adapters](doc-agent-adapters-custom-adapters.md), and the [public export list](doc-python-api-public-exports.md).
