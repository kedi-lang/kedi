# Browser and Playground

## Browser-Compatible Execution

Kedi parsing/compilation can remain on a Python backend while embedded Python
and model inference are delegated through bridges. This avoids pretending the
entire Python runtime or provider SDK runs natively in the browser.

For the supported local cell interface, use [Notebook](notebook.md). The bridge
API and WebGPU demo below are separate integration surfaces; Notebook does not
depend on the playground repository or this example server.

## Pyodide Executor

Integration sketch: `bridge` must implement the synchronous request protocol,
and `program` is an already parsed Kedi program. This is not a browser launcher:

```python
from kedi.executors import PyodideExecutor
from kedi.lang import compile_program

executor = PyodideExecutor(bridge, timeout=60)
runtime = compile_program(program, executor=executor)
```

`PyodideExecutor` subclasses `PlaygroundExecutor`. Its synchronous bridge
receives operation, code, referenced environment values, synchronization names,
and source offset; it returns result, environment updates, stdout, or an error.

The embedding application owns worker startup, timeouts, interruption and
cleanup. In particular, creating an executor does not download Pyodide or start
a worker. Notebook supplies that lifecycle for its own browser executor.

## Supported Language Surface

Inline expressions, Python blocks, side effects, preludes, and type expressions
are forwarded. JSON-compatible values cross by value. Worker-owned unsupported
objects cross as opaque `PlaygroundReference` handles when the bridge supports
them.

Only names referenced by the Python code are serialized, reducing bridge
traffic. Existing Kedi variables are synchronized back after side effects.

## Browser Model Bridge

`WebGPUAdapter` sends prompts, JSON Schema, settings, and iterative tool calls
through a `BrowserModelBridge`. It supports structured output, scoped tools,
profile/model/effort/settings overrides, and optionally selected MCP
transports.

It is not registered as a normal DSL shortname; construct and pass the adapter
instance from Python.

## Interactive Example

The repository's `examples/webgpu_kedi_demo` runs a GGUF model with wllama/
WebGPU in a browser worker while the Python server executes Kedi:

```bash
python examples/webgpu_kedi_demo/server.py
```

Open `http://127.0.0.1:8787`. The demo supports Hugging Face download, a local
server model, or browser file selection.

## Runtime Isolation

A bridge is an execution boundary, not automatically a security boundary.
Security depends on the worker runtime, exposed operations, serialization
policy, server endpoints, and browser origin controls.

The demo's Env Manager stores values in browser `localStorage`. `HF_TOKEN` may
be sent for model downloads; other displayed BYOK fields are currently stored
but not wired into adapters.

## Browser Limitations

- Browser model quality and structured/tool reliability depend on the loaded
  GGUF model.
- Response caching is disabled in the demo.
- Model files are excluded from Docker images.
- The demo adapter/HTTP polling bridge is a proof of integration, not a
  production multi-user service.
- Debug/stdout and opaque references need explicit lifecycle handling.
