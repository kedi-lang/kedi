> Offline source: `kedi-docs-main/docs/agent-adapters/index.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Agent Framework Adapters](#agent-framework-adapters)
- [Agent Harness Adapters](#agent-harness-adapters)
- [Shared Adapter Contract](#shared-adapter-contract)
- [Capability Negotiation](#capability-negotiation)
- [Choose a Backend](#choose-a-backend)
- [In This Section](#in-this-section)

# Models and Integrations { #agent-adapters }

Adapters translate Kedi's uniform prompt contract into a model framework or an
agent harness. Selection is explicit because those backends have materially
different capabilities.

## Agent Framework Adapters

Use `> adapter:` or Python `adapter=`:

| Shortname | Backend |
| --- | --- |
| `pydantic` | Pydantic AI |
| `dspy` | DSPy |
| `langchain` | LangChain agents |

Framework adapters accept Kedi-defined tools and structured output directly.

## Agent Harness Adapters

Use `> agent:` or Python `agent=`:

| Shortname | Backend |
| --- | --- |
| `claude` | Claude Agent SDK |
| `codex` | Codex App Server |
| `acp` | Generic stdio ACP agent |
| `a2a` | Remote Agent2Agent agent |

Harnesses are full agent processes with their own tools, sessions, sandboxing,
and permission models. Kedi maps what each protocol actually supports.

## Shared Adapter Contract

Every adapter provides async and sync paths for:

- `produce(...)`: a template plus structured output schema;
- `invoke(...)`: a raw prompt returning text.

It also declares `kind`, `shortname`, and `AdapterCapabilities`. Optional
protocols add profile overrides, tool registration, approvals, subagents,
compact artifacts, and stateful conversation handoff.

## Capability Negotiation

The [capability matrix](doc-reference-capability-matrix.md) is generated from
the built-in adapters' declared metadata and is the canonical support table.
Kedi checks required capabilities rather than silently emulating a missing
surface. For example, `>> ... [field]` requires structured output; a text-only
adapter must use raw capture instead:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
[answer] << Answer the question in plain text.
```

Transport modes, provider restrictions, native harness tools, and lifecycle
details are adapter-specific and remain documented on each adapter page.

## Choose a Backend

- Choose Pydantic for the broadest typed Python integration.
- Choose LangChain for LangChain models, middleware, and tool ecosystems.
- Choose DSPy for signatures, ReAct, and GEPA workflows.
- Choose Claude or Codex when a coding-agent harness and its native tools are
  part of the task.
- Choose ACP when driving an existing stdio ACP agent as a text harness.
- Choose A2A when orchestration stays local but an independently deployed agent
  owns execution, tools, and sandboxing.

## In This Section

**Selection**

- [Backend Selection](doc-agentic-engineering-backend-selection.md)
- [Models and Reasoning](doc-agentic-engineering-models-and-reasoning.md)
- [Capability Matrix](doc-reference-capability-matrix.md)
- [Codex Model Connections](doc-agent-adapters-codex-models.md)

**Framework Adapters**

- [Pydantic AI](doc-agent-adapters-pydantic-ai.md)
- [LangChain](doc-agent-adapters-langchain.md)
- [DSPy](doc-agent-adapters-dspy.md)

**Agent Harnesses**

- [Codex App Server](doc-agent-adapters-codex.md)
- [Claude Agent SDK](doc-agent-adapters-claude.md)
- [ACP Agents](doc-agent-adapters-acp.md)
- [A2A Cloud Agents](doc-agent-adapters-a2a.md)

**Decision Models**

- [TypeSafe Jev](doc-agent-adapters-typesafe.md)
- [Claims and Thresholds](doc-agent-adapters-jev-claims.md)
- [Criteria and Output Types](doc-agent-adapters-jev-criteria.md)
- [Finite Extraction](doc-agent-adapters-jev-extraction.md)
- [Decision Evidence](doc-agent-adapters-jev-evidence.md)
- [Tool Selection](doc-agent-adapters-jev-tools.md)
- [Generate, Evaluate, Route](doc-agent-adapters-jev-workflow.md)

**Extending Kedi**

- [Custom Adapters](doc-agent-adapters-custom-adapters.md)
- [Artifact Integration](doc-agent-adapters-artifact-contract.md)
