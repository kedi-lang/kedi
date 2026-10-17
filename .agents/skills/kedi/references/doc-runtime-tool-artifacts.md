> Offline source: `kedi-docs-main/docs/runtime/tool-artifacts.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Normative Model](#normative-model)
- [Runtime Architecture](#runtime-architecture)
- [End-to-End Dataflow](#end-to-end-dataflow)
- [Generated fields](#generated-fields)
- [Tool results](#tool-results)
- [Native reads](#native-reads)
- [Model substitutions](#model-substitutions)
- [Admission Algorithm](#admission-algorithm)
- [Configuration and Lexical Scope](#configuration-and-lexical-scope)
- [Policy reference](#policy-reference)
- [Reference Contract](#reference-contract)
- [Management Tools](#management-tools)
- [`search_artifacts`](#search_artifacts)
- [`read_artifact`](#read_artifact)
- [`query_artifact` (opt-in)](#query_artifact-opt-in)
- [`run_artifact_code`](#run_artifact_code)
- [`release_artifact`](#release_artifact)
- [Agent Routing Contract](#agent-routing-contract)
- [End-to-End Multi-Artifact Reduction](#end-to-end-multi-artifact-reduction)
- [Explicit Streaming](#explicit-streaming)
- [Serialization and Stores](#serialization-and-stores)
- [Canonical serialization](#canonical-serialization)
- [Memory store](#memory-store)
- [File store](#file-store)
- [Quotas, Expiry, and Concurrency](#quotas-expiry-and-concurrency)
- [History and Cache Stability](#history-and-cache-stability)
- [Failure Semantics](#failure-semantics)
- [Adapter Contract](#adapter-contract)
- [Security and Observability](#security-and-observability)
- [Operational Guidance](#operational-guidance)

# Artifacts { #tool-artifacts }

Tool Artifacts is Kedi's bounded large-value transport. It prevents large tool
results and generated fields from being copied into model context while
preserving the original typed value for Kedi and embedded Python.

Artifacts are enabled by default. They are part of the normal agent execution
contract, not an optional persistence API: values that remain small continue to
flow inline, while qualifying values are represented to the model by compact,
session-scoped references.


## Normative Model

Tool Artifacts separates one logical value into three representations:

1. **Native value**: the original `str`, `bytes`, list, Pydantic model,
   dataclass, or other Python object used by Kedi and Python.
2. **Stored payload**: the serialized or process-local value owned by one
   `ArtifactManager` and one artifact session.
3. **Model-visible reference**: a bounded `ArtifactRef` containing metadata,
   summary, preview, size, expiry, and instructions for retrieving content.

These representations have different consumers and must not be conflated:

| Consumer | Representation |
| --- | --- |
| Kedi return block, embedded Python, `run_main()` | Original native value |
| A later model call receiving an artifact-backed substitution | `ArtifactRef` |
| `read_artifact` and `run_artifact_code` | Bounded views of the stored payload |
| Portable conversation history | References and bounded read results |

The core invariant is:

> Artifact conversion may change how a value crosses the model boundary, but
> it must not change the value observed by Kedi or Python.

Consequently, `ArtifactRef[T]` is a model transport type. It does not replace
`T` in ordinary application code.


## Runtime Architecture

One artifact-enabled execution is composed of the following parts:

| Component | Responsibility |
| --- | --- |
| `ArtifactPolicy` | Lexically scoped admission, storage, quota, and expiry configuration |
| `ArtifactManager` | Session ownership, admission, deduplication, reads, leases, release, and cleanup |
| `ArtifactStore` | Memory or file-backed payload storage |
| `ArtifactHandle[T]` | Lazy internal binding that resolves to the native `T` |
| `ArtifactRef[T]` | Compact model-visible metadata |
| Artifact management tools | Search, bounded read, code-based reduction, and release |
| `ArtifactHistory` | Append-only lifecycle events for portable history and telemetry |

The manager belongs to an explicit artifact session. A normal stateless Kedi
run creates a session for that run. A Python `session()` keeps one manager and
conversation state across multiple calls.

The management tools and artifact instructions are installed as a stable
contract from the first artifact-enabled model call. They are not injected only
after the first large result. Keeping the prefix stable avoids invalidating
provider prompt caches when an artifact first appears later in a run.


## End-to-End Dataflow

### Generated fields

Each generated field is validated against its declared Kedi type before
artifact admission. Fields are measured independently; one large field can
become an artifact while sibling fields remain inline.

```text
model output
  -> structured-output validation
  -> serialize and measure each field
  -> inline value OR ArtifactHandle[T]
  -> bind into KediEnv
```

### Tool results

A tool result enters artifact admission only after the tool was approved,
executed, and return-validated:

```text
tool request
  -> approval
  -> tool execution
  -> return validation
  -> serialize and measure
  -> inline result OR ArtifactHandle[T]
  -> model receives inline result OR ArtifactRef[T]
```

An `ArtifactRef` returned from a tool means the source tool already completed.
The model must inspect the reference instead of repeating the source operation.

### Native reads

Kedi stores an `ArtifactHandle[T]` internally. When Kedi or embedded Python
reads that binding, the handle asks its session manager for the original value:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> artifacts:
    threshold: 1b

>> A detailed explanation of Kedi is [report].

# `report` resolves to the complete native string here.
= `report.upper()`
```

The same rule applies to `run_main()`, `@kedi.query`, and `@kedi.bind`: their
callers receive the native result, not artifact metadata.

### Model substitutions

The model boundary is deliberately different. A later template that receives
an artifact-backed value sees only its compact reference:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
>> A detailed release report is [report].
>> The conclusion in <report> is [conclusion].
```

The second call can use `read_artifact` or `run_artifact_code` to retrieve only
the evidence required to fill `conclusion`. The full report is not implicitly
copied into the second prompt.


## Admission Algorithm

See [Admission Algorithm](doc-runtime-artifact-policy.md).

## Configuration and Lexical Scope

See [Configuration and Lexical Scope](doc-runtime-artifact-policy.md).

### Policy reference

See [Configuration and Lexical Scope](doc-runtime-artifact-policy.md).

## Reference Contract

See [Reference Contract](doc-runtime-artifact-policy.md).

## Management Tools

See [Management Tools](doc-runtime-artifact-retrieval.md).

### `search_artifacts`

See [Management Tools](doc-runtime-artifact-retrieval.md).

### `read_artifact`

See [Management Tools](doc-runtime-artifact-retrieval.md).

### `query_artifact` (opt-in)

See [Management Tools](doc-runtime-artifact-retrieval.md).

### `run_artifact_code`

See [Management Tools](doc-runtime-artifact-retrieval.md).

### `release_artifact`

See [Management Tools](doc-runtime-artifact-retrieval.md).

## Agent Routing Contract

See [Agent Routing Contract](doc-runtime-artifact-retrieval.md).

## End-to-End Multi-Artifact Reduction

See [End-to-End Multi-Artifact Reduction](doc-runtime-artifact-reduction.md).

## Explicit Streaming

See [Explicit Streaming](doc-runtime-artifact-lifecycle.md).

## Serialization and Stores

See [Serialization and Stores](doc-runtime-artifact-lifecycle.md).

### Canonical serialization

See [Serialization and Stores](doc-runtime-artifact-lifecycle.md).

### Memory store

See [Serialization and Stores](doc-runtime-artifact-lifecycle.md).

### File store

See [Serialization and Stores](doc-runtime-artifact-lifecycle.md).

## Quotas, Expiry, and Concurrency

See [Quotas, Expiry, and Concurrency](doc-runtime-artifact-lifecycle.md).

## History and Cache Stability

See [History and Cache Stability](doc-runtime-artifact-lifecycle.md).

## Failure Semantics

See [Failure Semantics](doc-runtime-artifact-safety.md).

## Adapter Contract

Artifact support and conversation continuation are separate contracts. Consult
the [adapter extension contract](doc-agent-adapters-artifact-contract.md) and
the selected adapter's documentation instead of inferring history support from
artifact support. Pydantic AI and LangChain both support stateful conversations.


## Security and Observability

See [Security and Observability](doc-runtime-artifact-safety.md).

## Operational Guidance

See [Operational Guidance](doc-runtime-artifact-safety.md).
