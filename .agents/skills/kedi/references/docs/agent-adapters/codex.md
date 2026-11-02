# Codex

## Codex App Server

The Codex adapter starts:

```text
codex app-server --stdio
```

It communicates directly over App Server JSON-RPC. The default thread is
ephemeral, so ordinary Kedi calls do not become persistent Codex threads.

## Authentication

Install the Codex CLI, ensure `codex` is on `PATH`, and authenticate it before
running Kedi. The App Server uses the CLI's existing authentication and
configuration; Kedi does not accept or persist Codex credentials.

## Model and Reasoning Settings

```kedi
> agent: codex
> model: gpt-6.1-sol
> effort: high
```

Supported settings are `approval_policy`, `config`, `cwd`, `model_provider`,
`personality`, `sandbox`, `service_tier`, `summary`, and `timeout`.

## Working Directory

`cwd` defaults to the Kedi process working directory:

```kedi
> settings:
    cwd: /workspace/project
```

The directory governs Codex built-in filesystem and shell behavior.

## Sandbox Policy

Kedi defaults Codex to `workspace-write`. Accepted aliases normalize to Codex
App Server sandbox values. Use `read-only` for analysis-only tasks or a broader
mode only when the environment and approval policy justify it.

## Approval Policy

Kedi leaves the App Server in approval-requesting mode by default and resolves
native shell/file approval methods through Kedi policy. A static/dynamic Kedi
policy also guards projected dynamic tools.

App Server native approvals cannot apply Kedi's edited arguments. An `edit`
decision for such a native request is declined rather than silently treated as
allow.

## Built-In Tools

Codex retains its harness-native shell and filesystem tools. Kedi procedures
and Python tools become App Server dynamic tools for the active run.

Kedi MCP declarations are currently unsupported by `CodexAdapter` and raise;
this does not mean the Codex installation itself has no MCP configuration.

## CodeMode

`> codemode: enabled` publishes only `search_tools`, `get_tool_schema`, and
`execute_code` as App Server dynamic tools. Kedi procedures and Python tools
move behind the run-scoped catalog, while Codex's native filesystem, shell, and
search tools remain available as harness controls. CodeMode settings and the
catalog participate in thread reuse identity.

Nested calls use Kedi inline approval exactly once. CodeMode does not add MCP
support to this adapter; Kedi MCP declarations still fail explicitly.

## Structured Output

Kedi passes JSON Schema through `turn/start.outputSchema`, parses the returned
JSON, and validates Pydantic models. Object schemas are closed with
`additionalProperties: false`.

Supported string formats are `date`, `date-time`, `duration`, `email`, and
`time`. Other JSON Schema string formats fail before the turn; use a plain or
`Annotated` string when semantic guidance is sufficient.

## Subagent Lifecycle

Codex supports foreground/background children. Dynamic tools, sandbox, cwd,
model, instructions, and approval ceilings are rebuilt per child scope.

Each native turn owns its active Kedi tools and approval callbacks. Turn timeout
or cancellation stops async callbacks and waits for their cleanup before the
call returns. Repeated cancellation does not interrupt that cleanup a second
time, and sibling turns retain their own bindings. Closing an event consumer
alone does not cancel the underlying run.

Cancellation does not undo committed effects or forcibly stop arbitrary
synchronous Python code. Cleanup that cannot finish within its bounded deadline
raises an error instead of claiming the work stopped. An interrupted native
turn may end without a final token-usage notification; missing usage is not
evidence that no inference occurred.

## Capability Limits

- Kedi MCP server declarations are unsupported.
- Native approval handlers support allow/deny, not argument editing.
- The `codex` executable and compatible App Server are required.
