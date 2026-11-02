# Claude

## Claude Agent SDK

Install the optional SDK:

```bash
uv add "kedi[claude]"
```

In a source checkout use `uv sync --extra claude`.

## Authentication

The Claude Agent SDK owns authentication. Configure the SDK-supported
credentials before running Kedi; `ANTHROPIC_API_KEY` is the common
non-interactive path. Authentication and other non-retryable 4xx SDK events
fail immediately with actionable context; transient retry events may continue.

## Model and Effort

```kedi
> agent: claude
> effort: high
```

With no model override, the Claude Agent SDK selects its default Claude model.
Effort is normalized through the shared Kedi effort mapping.

Luna is an OpenAI model, not a Claude model. To use Luna, select a compatible
backend instead:

```kedi
> adapter: pydantic
> model: openai/gpt-6-luna
> effort: high
```

## System Presets

Claude Code's system prompt preset remains active. `> instructions:` is appended to
that preset rather than replacing it, preserving coding/file-tool behavior.

## Working Directory

```kedi
> settings:
    cwd: `str(project_root)`
```

Other supported settings are `allowed_tools`, `disallowed_tools`, `env`,
`fallback_model`, `max_budget_usd`, `max_turns`, `permission_mode`, and
`tools`.

## Permission Mode

Non-interactive runs default to `permission_mode: acceptEdits`. Override it
explicitly when a narrower Claude SDK mode is required. Kedi still installs its
approval hook; SDK permission mode and Kedi approval are separate layers.

## Built-In File Tools

Claude Code tools are enabled through the SDK preset by default. Kedi-projected
tools are exposed through an in-process SDK MCP server and added to
`allowed_tools`. External stdio/SSE/HTTP MCP servers are also mapped.

Native `Read`, `Glob`, `Grep`, `WebFetch`, and `WebSearch` are classified
read-only. Other native/MCP operations require policy.

## CodeMode

`> codemode: enabled` exposes the three shared CodeMode controls through Kedi's
in-process Claude SDK MCP server. Kedi procedures and declared stdio/SSE/HTTP
MCP tools are materialized into the run-scoped catalog and are callable only
from `execute_code`. Claude's built-in filesystem, search, web, and shell tools
remain native control-plane tools.

The SDK approval hook skips the three controls; the nested application call is
the single approval owner. Kedi closes the local MCP clients and Monty session
on completion, error, or cancellation.

## Subagent Lifecycle

Claude supports foreground/background children and native resume using the SDK
session ID. Child options preserve model, tools, system, MCP, approval ceiling,
and usage constraints.

## Capability Limits

The SDK must be installed for this adapter. Exact built-in tool availability
and model names follow the installed Claude Agent SDK. Kedi validates its own
setting keys but does not emulate missing SDK capabilities.
