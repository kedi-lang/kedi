# Custom Adapters

A custom adapter connects Kedi to a model framework or agent harness that
Kedi does not ship. The complete contract, with tested code for every part,
is the [Adapter Development](../adapter-development/index.md) section. Start
with the [Reference Adapter](../adapter-development/reference-adapter.md): a
working adapter that every page of the section extends.

Pass an adapter instance when compiling a program:

```python
from kedi.lang import compile_program, parse_program

runtime = compile_program(parse_program(source), adapter=MyAdapter())
```

## The Adapter Contract

An adapter declares `kind`, `shortname`, and `capabilities`, and implements
`invoke`, `invoke_sync`, `produce`, and `produce_sync`. Kedi renders the
prompt and instructions; the adapter sends them unchanged and returns text or
the decoded value. See [Call Lifecycle](../adapter-development/call-lifecycle.md)
and [Prompts and Instructions](../adapter-development/prompt-contract.md).

## Framework and Harness Kinds

`"agent-framework"` adapters are selected with `adapter=` / `> adapter:`,
`"agent-harness"` adapters with `agent=` / `> agent:`. See the
[overview](../adapter-development/index.md#kinds).

## Shortnames

`shortname` names the adapter in diagnostics, telemetry, approvals, and
conversation state. Keep it stable.

## Text Invocation

`invoke_sync(prompt=..., instructions=...)` returns the response text. See
[Call Lifecycle](../adapter-development/call-lifecycle.md#which-statement-calls-what).

## Structured Outputs

`produce_sync` receives `output_schema` for template captures and
`output_type` for claims and typed raw invokes, and returns the decoded value.
See [Structured Output](../adapter-development/structured-output.md).

## Tool Registration

`tool_scope` and `register_tool` receive the call's tools; executing a call
goes through the tool's wrapped `fn`. See
[Tools](../adapter-development/tools.md).

## MCP Support

See [MCP, Skills, and Capability Tools](../adapter-development/mcp-and-skills.md).

## Capability Metadata

See the [Capability Reference](../adapter-development/capabilities.md).

## Parallel Thread Safety

See [Concurrency and Event Loops](../adapter-development/concurrency.md).

## Error Translation

See [Errors, Retries, and Cancellation](../adapter-development/errors.md).
