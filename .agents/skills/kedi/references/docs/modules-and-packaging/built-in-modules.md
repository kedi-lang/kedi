# Built-In Modules

Kedi ships a small set of ordinary `.kedi` modules. They use the same explicit
exports, profiles, Python interop, tool metadata, and resolution rules as project
modules.


## Resolution and Shadowing

Import a bundled module by name:

```kedi
> import: filesystem:
  readonlyfs
  read_text_file
```

A sibling `filesystem.kedi` takes precedence over the bundled module. Use this
deliberately; an accidental same-name file changes the imported API.


## `errors`

The `errors` module exports `ModuleNotInstalledError`:

````kedi
> import: errors

```
raise ModuleNotInstalledError("httpx", required_by="HTTP reports")
```
````

The exception subclasses `ModuleNotFoundError` and formats singular or plural
missing package names. Use it when a feature has a clear optional Python
dependency.


## `require`

`require` exports a Python-callable helper:

````kedi
> import: require

```
require(["httpx", "pydantic"], required_by="Remote reports")
```
````

It checks import availability with Python's module discovery and returns `True`
when all names are present. Missing modules raise `ModuleNotInstalledError`.
It does not install packages or validate their versions.


## `filesystem`

See [`filesystem`](filesystem.md).

## `sandbox`

The optional `sandbox` module exports `execute_code` and a `sandbox` profile:

```kedi
> import: sandbox

[result: Any] = `execute_code("sum(values)", {"values": [1, 2, 3]})`
```

It requires the Python package `pydantic_monty`. Importing the module checks that
dependency immediately. `execute_code(code, inputs={}, fail_fast=False)` executes
with Monty and returns the native final result; the backtick expression preserves
that value. Inputs default to an empty mapping. Execution failures are returned as
text by default so an agent can inspect them; pass `fail_fast=True` to raise instead.

This sandbox is for intentionally constrained generated code. It is not the
execution mechanism for ordinary Kedi Python blocks, which use the configured
Kedi executor.

## `helpers`

`helpers` exports `llm_approval(request: ApprovalRequest) -> ApprovalDecision`,
an experimental model-backed approval handler. It requires tool reasons to be
enabled on the calling tool surface; the model's reason is untrusted context,
not proof of user authorization. The helper produces and displays assessed
risk, authorization confidence, an allow/deny decision, and a concise
rationale. A denial returns the same report to the calling model through the
adapter's denied tool result. It does not run for ordinary read-only calls.

```kedi
> import: helpers:
  llm_approval

> settings:
  tool_reason: enabled
> approval: `llm_approval`
```

This configures the handler; it does not invoke a tool by itself. Select a
compatible model and explicit tool scope before using it. Keep deterministic
allowlists or human review for decisions that require stronger guarantees.
See [Tool Reasons](../agentic-engineering/tool-reasons.md).


## `this` and Example Modules

`this` is a bundled demonstration/easter-egg module whose import executes its
encoded output. It is not an application API.

`wordle` exports its game profile and game procedures; it requires a graphical
environment and optional packages. See [Example Modules](example-modules.md)
for its actual export list. These are demonstrations, not general-purpose
stdlib contracts.
