# ACP Agents

## Stdio Agent Commands

The generic ACP adapter launches an Agent Client Protocol process and speaks
newline-delimited JSON-RPC over stdio:

```kedi
> agent: acp:
    command: npx @agentclientprotocol/codex-acp
```

Command strings are shell-split without invoking a shell. Python may pass a
sequence to avoid quoting ambiguity.

## Explicit Commands

ACP does not have an implicit command source. Every runnable ACP configuration
must bind the command in a typed `> agent: acp:` body or construct
`ACPAdapter(command=...)` in Python. Plain `> agent: acp`, CLI command options,
and environment command fallbacks are intentionally unsupported.

## Working Directory and Environment

```kedi
> settings:
    cwd: /workspace/project
    env: `{"MODE": "review"}`
    timeout: 300
```

Clients are reused by `(command, cwd, env)` and each prompt opens a fresh ACP
session. Stdio process cwd and `session/new.cwd` receive the configured cwd.

## Timeouts

`timeout` is the prompt request timeout and must be a positive number.
`request_timeout` on `ACPAdapter` governs protocol requests by default.
Disconnect errors include the last 40 stderr lines for diagnosis.

When a prompt times out or its async caller is cancelled, Kedi sends
`session/cancel` for that prompt's session and waits for the agent to stop.
Other sessions are not cancelled. If the agent does not acknowledge within
five seconds, Kedi closes the connection; other calls sharing that unresponsive
process then fail as disconnected. Cancellation during session setup does not
submit a prompt after the setup finishes.

Closing an adapter first closes its input stream so the agent can stop its own
children. On POSIX systems Kedi also cleans up the process group it created,
then joins the reader threads. This does not terminate unrelated agent processes.

Only an `end_turn` response is a completed answer. Token/request exhaustion,
refusal, cancellation, and invalid stop reasons raise an error rather than
returning partial text as success. Unsupported inbound client requests receive
a protocol error; permission requests are denied with the ACP `cancelled`
outcome. Kedi does not silently approve remote tools.

## Model Selection

Generic ACP model selection is not currently mapped. `> model:` and
`> effort:` are unsupported capability overrides. Configure the child ACP
agent's own default model through its command/environment.

## Prompt and Structured Results

ACP supports raw text only:

```kedi
> agent: acp:
    command: npx @agentclientprotocol/codex-acp
[answer] << Inspect the repository and summarize the risk.
= `answer`
```

Kedi concatenates `agent_message_chunk` text updates. Structured `>>` output
raises `NotImplementedError`; no prompting shim fabricates a schema.

## ACP Capability Mapping

Kedi sends protocol initialization, `session/new` with cwd and an empty
`mcpServers` list, `session/prompt`, streamed updates, then `session/close`.
Each call is independent.

Unsupported model/effort overrides and local tool or MCP registration are
rejected rather than ignored. Remote tools configured inside the child agent
are separate from Kedi's local tool registry.

## Current Limits

- no structured output;
- no Kedi tool registration;
- no Kedi MCP projection;
- no model or effort override;
- no Kedi subagents;
- only text `agent_message_chunk` updates are collected.
