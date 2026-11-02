# Agent backends, profiles, and tools

## Contents

- [Select a backend](#select-a-backend)
- [Capture and composition](#capture-and-composition)
- [Expose tools](#expose-tools)
- [MCP, hooks, and skills](#mcp-hooks-and-skills)
- [Decision models](#decision-models)

## Select a backend

| Surface | Selections | Read before use |
| --- | --- | --- |
| Framework `> adapter:` / Python `adapter=` | `pydantic`, `langchain`, `dspy` | [Pydantic](docs/agent-adapters/pydantic-ai.md), [LangChain](docs/agent-adapters/langchain.md), [DSPy](docs/agent-adapters/dspy.md) |
| Harness `> agent:` / Python `agent=` | `codex`, `claude`, `acp`, `a2a` | [Codex](docs/agent-adapters/codex.md), [Claude](docs/agent-adapters/claude.md), [ACP](docs/agent-adapters/acp.md), [A2A](docs/agent-adapters/a2a.md) |
| Codex-authenticated framework model | `codex_responses_model(...)` | [Codex models](docs/agent-adapters/codex-models.md); distinct from the Codex harness |

The CLI's historical `--adapter` flag accepts both kinds; that does not make `> adapter: codex` valid source. ACP needs an explicit command in `> agent: acp:` or `ACPAdapter(command=...)`. A2A needs an endpoint/auth reference; the remote service owns its model/tools. Generic A2A peers provide text; typed outputs need the negotiated Kedi extension.

Read the [capability matrix](docs/reference/capability-matrix.md) for the specific operation. Use `> requires:` for hard contracts such as `structured_output`, `tool_registration`, or `background_subagents`. Unsupported capability checks must fail, not silently turn into manual JSON prompting or nonexistent tools. Literal selections improve static diagnostics. Install/authenticate the chosen integration; an extra alone does not do that.

## Capture and composition

Define `> profile: name:` at top level, with `>`-prefixed members. Use only one backend kind within a profile. Definition/application configures later calls; neither itself performs a model request.

A procedure captures configuration at its **definition site**. Later caller directives do not retroactively supply its tools/instructions. Apply profiles and direct directives in source order; a later profile can replace an earlier direct setting. Read [profile merge rules](docs/agentic-engineering/profiles.md) before layering profiles:

- Specified scalar policy/model/instructions replace; instructions do not concatenate.
- Settings merge by key, not recursively.
- Tools/children merge by name; an empty tool list does not revoke earlier tools.
- MCP appends; hooks compose per event; artifacts overlay explicit policy fields.
- Omitted members retain the existing value.

Avoid tool/profile name collisions. Single `> use: name` resolves a Kedi procedure, then Python callable, then profile (and capability where supported). For portable stable code, the multiline form lists tools; consult the capability page for next-release composition changes.

Capabilities are marked **Unreleased** in the documentation. Read [capabilities](docs/agentic-engineering/capabilities.md), [Python capabilities](docs/python-api/capabilities.md), and the [version gate](pitfalls.md) before using them. They add owned instructions/tools/policies rather than replace host configuration. Pydantic supports eager/deferred composition; LangChain eager only. Do not invent `> capabilities:`. A capability cannot configure the host model/history/budget or act as a separate agent.

## Expose tools

Give a procedure a precise typed signature and a leading `###` docstring with argument meaning. Place one `> tool:` block after that docstring and before executable statements. Supported keys: `name`, `description`, `risk`, `retries`, `retry_on`. Register it with `> use:` in the model's scope. Directly calling a procedure is different from invoking it through an agent tool policy.

Python callables use `@kedi.tool(...)`, complete annotations, and a docstring. Make them visible via prelude/configured tools/env, then expose with `> use:`. Tools are invoked with keyword arguments; avoid weak/variadic signatures.

A live tool catalog can also include runtime-owned artifact tools and child-parent messaging. Assert that required application tools are present; do not assume `> use: one_tool` makes the complete catalog contain exactly one tool. Inventory assertions should distinguish application tools from enabled runtime capabilities.

Risks are `read_only`, `mutating`, `sensitive`; custom tools default to `mutating`. Mark a tool read-only only if it actually is. Python API defaults allow read-only and deny risky calls. CLI defaults ask interactively for risky calls. Use `> approval: allow`, `deny`, or a Python callback returning `ApprovalDecision`; `edit` is a callback result, not a directive value. Scope the policy to the intended tool use.

Only tool-body exceptions eligible under `retry_on` are retried, up to `retries + 1` attempts. Argument/result validation, approval, hooks, cancellation, `KeyboardInterrupt`, and `SystemExit` are outside those retries. Return concise serializable native data; raise precise exceptions for failures. Inspect actual tool results rather than assuming a model used a registered tool.

Use [tool_profile.kedi](../assets/examples/tool_profile.kedi) for a complete local lookup. Its fixture test checks that the real tool body executes before the model response.

## MCP, hooks, and skills

MCP configuration uses plain subkeys. `stdio` needs `command`, optional native list `args` and string mapping `env`; HTTP/streamable HTTP and SSE need `url`, optional string-mapping headers. Read [MCP](docs/agentic-engineering/mcp.md) for backend support and path handling. Do not invent a server or authenticate one from an example placeholder.

Read [hooks](docs/agentic-engineering/hooks.md) before implementing lifecycle handlers. Prompt/pre-tool handlers may edit/deny; post handlers observe and return `None`. Tool arguments are revalidated after edits. Preserve inherited enforcement and event-specific adapter support. `post_tool_result_failure` is distinct from tool-body failure; a completed side effect must not be blindly retried.

`tool_reason: enabled` requests optional untrusted justifications for risky tools; it is not user authorization. The experimental `helpers.llm_approval` requires reasons enabled. Use [approvals](docs/agentic-engineering/approvals.md) and [tool reasons](docs/agentic-engineering/tool-reasons.md) for exact contracts.

`> skills: enabled` exposes a bounded inventory and `read_skill`. It does not inject all bodies or grant capabilities. **Kedi's own installer copies only SKILL.md**; this multi-file authoring skill requires an agent host with filesystem/reference access. Do not install it with `kedi skills add` and assume references/scripts were copied. See [skills](docs/agentic-engineering/skills.md).

## Decision models

Jev/Typesafe and Laya classify decision fields using capture questions/criteria, unlike completion field filling. Use the exact provider/model setup from [Typesafe](docs/agent-adapters/typesafe.md), [Laya](docs/agent-adapters/laya.md), and the Jev guides in the [documentation index](docs.md). Import shared `Probability`, `Rubric`, `BooleanCriteria`, `ChoiceCriteria` through `> import: decisions` or `kedi.decisions`; those optional criteria do not select a model or load weights.

A nullable output field allows None but does not prevent a model from inventing a syntactically valid value. If an owner or identifier must come only from caller input, enforce that provenance deterministically before returning the result.

Decision evidence is provenance attached to a binding, not a reactive value. `decision_info(name)` can return `None` for a non-Jev binding and raises for an unknown name. Input changes do not rerun the model; reassigning output removes its current evidence. `matches_inputs` checks explicit tracked inputs, not changing files/tools/history. Read [reviewed evidence](docs/agentic-engineering/reviewed-evidence.md) before building a decision gate.
