# Use Kedi from Python

An operations update becomes a typed `Incident`, then a customer-facing notice.
The caller works with Python objects rather than extracting values from JSON text.

## Run

Complete the [shared setup](../README.md#setup), then:

```bash
cd examples/python_api
python main.py
```

Expect an incident identifying checkout timeouts with `resolved=false`, followed
by a short notice. The notice must not invent a recovery time or say that the
planned rollback has already fixed the incident.

## The two integration styles

- `@kedi.query` compiles the Kedi docstring of `extract_incident`.
- `@kedi.type` exposes the annotated Python `Incident` class to Kedi's type system.
- `@kedi.bind` gives `draft_notice` its implementation from [notice.kedi](notice.kedi).

The second call consumes the first call's typed result. `Path(__file__)` resolves
the bound source relative to the Python module, not the caller's working
directory. The program uses the configured adapter/model; it does not pick an
SDK-specific client behind the caller's back.

Use `kedi.context(adapter=...)` to supply a model instance in an application or
test. The regression test does this with deterministic Pydantic AI models and
verifies that `resolved=False` reaches the second prompt unchanged. Two calls
are intentional here: the example demonstrates a typed boundary, not an
argument that every extraction needs a separate rewrite pass.
