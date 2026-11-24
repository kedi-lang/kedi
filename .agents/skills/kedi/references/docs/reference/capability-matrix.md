# Capability Matrix

Capabilities are declared contracts, not feature guesses. Kedi validates the
active profile against them and fails when a required surface is unavailable.

The tables between the generated markers come directly from the built-in
adapters' `AdapterCapabilities`. Run
`python scripts/sync_capability_matrix.py` after changing adapter metadata.
Every built-in adapter also implements raw text invocation because `invoke` is
part of the base adapter protocol rather than an optional capability.

<!-- BEGIN GENERATED ADAPTER CAPABILITIES -->
## Framework Adapters

| Capability | Pydantic AI | LangChain |
| --- | :---: | :---: |
| Capability composition | yes | yes |
| On-demand capability tools | yes | no |
| Structured output | yes | yes |
| Kedi tool registration | yes | yes |
| Kedi-declared MCP | yes | yes |
| Profile override | yes | yes |
| Model override | yes | yes |
| Effort | yes | yes |
| Settings | yes | yes |
| Code mode | yes | yes |
| Native approvals | yes | yes |
| Dynamic native approval handler | yes | yes |
| Foreground subagents | yes | yes |
| Background subagents | yes | yes |
| Live subagent input | yes | yes |
| Subagent interruption | yes | yes |
| Artifacts | yes | yes |
| Native artifacts | yes | yes |
| Stateful history | yes | yes |
| History replay | yes | yes |
| Native compaction | yes | yes |
| Request budget | yes | yes |
| Tool attempt budget | yes | yes |
| Semantic stream events | yes | yes |
| Native history processing | yes | yes |
| Media input | yes | yes |
| Agent lifecycle hooks | yes | yes |

## Agent Harnesses

| Capability | Claude Agent SDK | Codex App Server | ACP | A2A |
| --- | :---: | :---: | :---: | :---: |
| Capability composition | no | no | no | no |
| On-demand capability tools | no | no | no | no |
| Structured output | yes | yes | no | conditional |
| Kedi tool registration | yes | yes | no | no |
| Kedi-declared MCP | yes | no | no | no |
| Profile override | yes | yes | yes | yes |
| Model override | yes | yes | no | no |
| Effort | yes | yes | no | no |
| Settings | yes | yes | yes | no |
| Code mode | yes | yes | no | no |
| Native approvals | yes | yes | no | no |
| Dynamic native approval handler | yes | no | no | no |
| Foreground subagents | yes | yes | no | yes |
| Background subagents | yes | yes | no | yes |
| Live subagent input | yes | yes | no | no |
| Subagent interruption | yes | yes | no | no |
| Artifacts | yes | yes | no | no |
| Native artifacts | no | no | no | no |
| Stateful history | yes | yes | no | yes |
| History replay | no | no | no | no |
| Native compaction | yes | yes | no | no |
| Request budget | no | no | no | no |
| Tool attempt budget | no | no | no | no |
| Semantic stream events | yes | yes | yes | yes |
| Native history processing | no | no | no | no |
| Media input | no | no | no | no |
| Agent lifecycle hooks | yes | yes | yes | yes |
<!-- END GENERATED ADAPTER CAPABILITIES -->

ACP is intentionally a raw-text harness. It starts a fresh ACP session for each
Kedi invocation and consumes text chunks. Structured captures, Kedi tools, MCP
projection, model selection, effort, and subagents are rejected rather than
simulated.

A2A delegates execution to a remote agent. Structured output is conditional:
generic peers remain text-only, while peers advertising Kedi's versioned
structured-output extension accept typed captures. A2A does not project local
tools, MCP, model settings, effort, CodeMode, or approval handlers into the
remote process.

“Native approvals” means the adapter can project Kedi's policy into its own
tool loop. Kedi still performs argument validation and risk classification
around registered tools. Transport modes, provider support, and other
adapter-specific constraints are not represented by the boolean capability
metadata; consult the corresponding [adapter page](../agent-adapters/index.md).

## Structured Schema Notes

Framework support still depends on the selected provider/model. Codex accepts
string schema formats `date`, `date-time`, `duration`, `email`, and `time`; it
rejects Kedi output schemas requiring `Regex`, `HttpUrl`, or `FileUrl`. Claude
accepts the documented built-in formats. Validate the exact production model,
not only the adapter.

## Capability Meanings

| Capability | Required by |
| --- | --- |
| `capabilities` | `> use:` of a [capability bundle](../agentic-engineering/capabilities.md) |
| `deferred_tools` | Capability bundles with `> defer: true` |
| `structured_output` | Any `>>` block containing output fields |
| `tool_registration` | `> use:` of a procedure or Python callable |
| `mcp` | One or more `> mcp:` servers |
| `profile_override` | Lexical/profile state changes |
| `model_override` | Explicit `> model:` |
| `model_fallback` | `> fallback:` ordered alternatives |
| `effort` | Explicit `> effort:` |
| `settings` | Nonempty `> settings:` |
| `codemode` | `> codemode: enabled` progressive tool discovery and bounded execution |
| `native_approvals` | Adapter-owned risky tool loop |
| `native_approval_handler` | Dynamic handler projected natively |
| `subagents` | `> subagent:` |
| `background_subagents` | Background lifecycle tools |
| `live_subagent_input` | `> send` to a running child |
| `subagent_interrupt` | `> interrupt` of a running child |
| `artifacts` | `> artifacts:` enabled |
| `native_artifacts` | Native artifact admission for adapter-owned tools |
| `stateful_history` | History kept across requests by the adapter |
| `history_replay` | Replaying stored history into a new run |
| `history_processing` | `> history:` `processor` or `processor_condition` |
| `native_compaction` | Adapter-native history compaction |
| `request_budget` | `> budget:` `request_limit` |
| `tool_attempt_budget` | `> budget:` `tool_attempt_limit` |
| `media_input` | Media values attached to a prompt |
| `stream_events` | Completed semantic commentary/final messages and run lifecycle events |
| `hooks` | Agent lifecycle interception; support is declared separately for each event |

Untyped raw invokes can run on a text-only adapter; a typed raw invoke capture
needs `structured_output`. A profile
may be syntactically valid but capability-invalid for its selected backend;
that mismatch is an error before Kedi pretends the feature is active.

`codemode` replaces ordinary application-tool disclosure with explicit tool
search, schema hydration, and bounded code execution. Pydantic AI, LangChain,
Claude Agent SDK, and Codex App Server implement the Kedi-owned surface; Kedi
does not infer it from ordinary tool registration.

Hook support is event-specific. ACP and A2A expose only `user_prompt_submit`.
Pydantic AI, LangChain, Claude, Codex, and WebGPU expose
the event matrix documented in
[Agent Lifecycle Hooks](../agentic-engineering/hooks.md). A generic `yes` in the
generated table means at least one hook event is supported.

## Custom Adapters

A custom `AgentAdapter` declares `kind`, `shortname`, and
`AdapterCapabilities`, then implements async/sync structured production and raw
invocation. Set an unsupported capability to false/`None`; never advertise a
surface and silently ignore it. See the [Capability Reference](../adapter-development/capabilities.md).
