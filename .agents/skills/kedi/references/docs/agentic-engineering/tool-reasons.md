# Tool Reasons and LLM Approval

A tool justification is model-provided context for the approval handler. It is
not authorization, a new tool effect, or a separate reasoning-model call.
The feature is disabled by default and applies only to tools that may require
approval.

## LLM-backed Approval Helper

The built-in `helpers` module provides an LLM-backed dynamic handler that uses
the active model. Enable tool reasons with Pydantic AI or LangChain:

```kedi
> settings:
    tool_reason: enabled

> import: helpers
> approval: `llm_approval`
```

!!! warning "Experimental and limited-context"
    `llm_approval` raises an error when `tool_reason` is disabled. With the option
    enabled, it evaluates the tool name, description, declared risk, arguments,
    and optional model-provided reason. The reason is untrusted context, not proof
    of user authorization. This helper is not a complete authorization boundary.

The helper asks for four required outputs: assessed risk (`low`, `medium`, or
`high`), authorization confidence at the same three levels, an `allow`/`deny`
decision, and a concise rationale. Kedi prints this assessment when the decision
is made. It also stores the report in `ApprovalDecision.reason`; on denial,
Pydantic AI receives it as `ToolDenied` and LangChain receives it as an error
`ToolMessage`, allowing the calling model to understand why execution stopped.
Authorization confidence summarizes the supplied evidence and does not promote
the model-authored tool reason into trusted user permission.

## Optional Tool Reasons

Both `PydanticAdapter` and `LangChainAdapter` accept `tool_reason=False` by
default. In Kedi, use `> settings:` with `tool_reason: enabled` or
`tool_reason: disabled`. The setting follows lexical/profile scope and overrides
the constructor value without changing the shared adapter. If it is omitted,
the constructor default applies. It is not forwarded to provider model settings.
Unsupported adapters reject activation rather than silently ignoring it.

A backtick Python expression must return a boolean:

````kedi
```
import os
```

> settings:
    tool_reason: `os.getenv("TOOL_REASON_FOO_BAR") is not None`
````

Plain `true` / `false` are not accepted for this setting. Python constructors
and `settings={"tool_reason": True}` still use real Python booleans.

Enabling it adds an optional `reason: string` argument only to tools
that may require approval. Statically read-only tools without a risk resolver
keep their original schemas. Argument-aware tools expose the optional field,
but their resolved risk still determines whether the handler runs.

Instructions for writing a brief action justification are supplied once in the
shared model instructions, not repeated in every tool schema. Producing the
reason needs no additional model call. Omitting it is valid.

Handlers read `request.reason`; `request.arguments` contains only the real tool
arguments. Approval edits and risk rechecks retain the original justification;
it does not automatically justify an edited action.

This also works with MCP and CodeMode. For MCP, Kedi projects the extra field
locally and removes it before calling the server; the server's own schema does
not change. External MCP tools retain Kedi's existing conservative approval
policy. Provider-hosted MCP that bypasses Kedi's local approval boundary is not
supported by this option. An approval-applicable tool that already declares a
business argument named `reason` fails explicitly rather than losing or
overwriting that argument. Unmodified read-only schemas keep their own fields.

## Choose the Boundary

Use a deterministic approval handler for hard rules: allowed paths, tools,
operations, tenants, or spending limits. Add LLM approval only where a fallible
semantic judgment is acceptable. A persuasive justification must not override
a deterministic deny or an inherited safety ceiling.

## Failure Cases

Unsupported adapters, an existing business parameter named `reason`, invalid
configuration values, and LLM approval without enabled tool reasons fail
explicitly. Missing optional justification alone is not an invalid tool call.
Handlers should decide how to treat that absence for their application.
