# Adapter Development

An agent adapter is the boundary between Kedi's runtime and a model framework
or agent harness. Every `>>` template, `<<` raw invoke, claim, delegation, and
tool call a program makes passes through it. This section is the complete
contract for writing one.

## Kedi Renders, the Adapter Transports

Kedi decides what the model sees and what a result must look like:

- the rendered prompt and the instruction text: profile instructions,
  template rules, claim rules, and capability, skill, and artifact
  instructions;
- the output contract: a field schema for template captures, a Python type for
  claims and typed raw invokes, a wrapped JSON Schema for structured children;
- the tools, already wrapped with approval, required-tool tracking, artifact
  admission, and telemetry;
- validation of every returned value, budgets, limits, history, and
  conversation commits.

The adapter maps that onto its backend and back: messages, tool calls,
structured output, provider state, and usage. It must not write task prompts
of its own. See [Prompts and Instructions](prompt-contract.md).

## Kinds

| `kind` | Selected with | Typical backend |
|---|---|---|
| `"agent-framework"` | `> adapter:` or `compile_program(adapter=...)` | a model framework where Kedi supplies the tools and the loop is in-process (Pydantic AI, LangChain) |
| `"agent-harness"` | `> agent:` or `compile_program(agent=...)` | an agent runtime with its own tools, sandbox, and loop (Codex, Claude Agent SDK, ACP, A2A) |

Kedi rejects an instance passed through the wrong parameter.

## The Surface

Only the first row is required. Every other protocol is optional and is
switched on by an `AdapterCapabilities` flag; Kedi checks the flag before it
relies on the protocol, and the LSP uses the same flags for diagnostics.

| Protocol | Methods | Capability | Page |
|---|---|---|---|
| `AgentAdapter` | `invoke`, `invoke_sync`, `produce`, `produce_sync` | `supports_structured_output` | [Call Lifecycle](call-lifecycle.md), [Structured Output](structured-output.md) |
| `SupportsProfileOverride` | `default_model`, `profile_override` | `supports_profile_override`, `supports_model_override`, `supports_effort`, `supports_settings`, `settings` | [Profiles, Models, and Settings](profiles.md) |
| `SupportsTemplatePromptMode` | `template_prompt_mode` | — | [Prompts and Instructions](prompt-contract.md) |
| `SupportsToolRegistration` | `tool_scope`, `register_tool` | `supports_tool_registration`, `supports_deferred_tools`, `supports_capabilities` | [Tools](tools.md) |
| native approvals | `approval_ceiling` and the adapter's approval hook | `supports_native_approvals`, `supports_native_approval_handler` | [Approvals](approvals.md) |
| hooks | `HookRegistrationMixin`, `hooked_agent_run`, tool hook dispatch | `hooks` | [Hooks](hooks.md) |
| MCP | profile `mcp_servers` | `supports_mcp` | [MCP, Skills, and Capability Tools](mcp-and-skills.md) |
| CodeMode | `execute_code` controls | `supports_codemode` | [CodeMode](codemode.md) |
| artifacts | registered artifact tools, native admission | `supports_artifacts`, `supports_native_artifacts` | [Artifacts](artifacts.md) |
| `SupportsSubagentExecution` | `subagent_scope`, `subagent_run_kwargs`, `produce_subagent`, `telemetry_model_name` | `supports_subagents`, `supports_background_subagents`, `supports_live_subagent_input`, `supports_subagent_interrupt` | [Subagents](subagents.md) |
| `SupportsConversationState` | `conversation_scope` | `supports_stateful_history`, `supports_history_replay`, `supports_history_processing` | [Conversations and History](conversations.md) |
| compaction and caching | history codecs, compaction backends | `supports_native_compaction` | [Compaction and Prompt Caching](compaction.md) |
| multimodal input | `PromptDraft.media` | `supports_media_input` | [Multimodal Input](multimodal-input.md) |
| budgets | usage reporting, run-budget admission | `supports_request_budget`, `supports_tool_attempt_budget` | [Budgets and Usage](budgets.md) |
| events | `AgentEventSink`, stream events | `supports_stream_events` | [Events, Streaming, and Telemetry](events.md) |

Every capability is `None` (unknown), `True`, or `False`. Kedi treats only
`True` as support. The [Capability Reference](capabilities.md) lists every
field, what Kedi does differently because of it, and the error it raises when
a program needs a capability the adapter lacks.

## Reading Order

1. [Reference Adapter](reference-adapter.md): a complete, tested adapter that
   every later page extends.
2. [Call Lifecycle](call-lifecycle.md), [Prompts and Instructions](prompt-contract.md),
   [Rendered Request](rendered-request.md) and [Structured Output](structured-output.md):
   what each call receives and must return.
3. The pages for the protocols your backend can support.
4. [Concurrency and Event Loops](concurrency.md) and [Testing an
   Adapter](testing.md) before you ship.

The Python on these pages runs in the documentation test suite. A
`python title="name.py"` block is part of the module it names.
