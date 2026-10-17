> Offline source: `kedi-docs-main/docs/python-api/public-exports.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

# Public Export Index

History-specific exports live in `kedi.agent_adapter`: `HistoryProcessorContext`,
`HistoryProcessorState`, `HistoryProcessorCallback`, `HistoryProcessorCondition`,
`HistoryOrigin`, `HistoryProcessingError`, and
`HistoryProcessingValidationError`. `HistoryProcessingSettings` is exported from
`kedi.agent_profile`. See [History](doc-runtime-history.md#user-defined-history-processing).

This index covers the root package's `__all__`. A root export is not necessarily
a high-level constructor: event unions and protocol aliases describe contracts,
while runtime and artifact handles are advanced integration surfaces.

| Exports | Contract and detailed reference |
| --- | --- |
| `__version__` | Installed Kedi version string; not the provider or grammar revision |
| `query`, `bind` | Decorators preserving sync/async call shape; execute docstring/file Kedi, not the Python stub. [Query](doc-python-api-query.md), [Bind](doc-python-api-bind.md) |
| `explain`, `ExplainReport`, `ExplainedCall` | Read-only per-call-site adapter, model, tool and capability inspection; no user program execution. [Run and Parse](doc-tooling-run-and-parse.md) |
| `configure`, `context`, `reset_config`, `parallel`, `interactive` | Configuration replacement, scoped overlay, reset, concurrent override and incremental construction. [Parameters](doc-python-api-public-parameters.md) |
| `AdapterLike`, `AgentName`, `FrameworkAdapterName` | Selection aliases; framework and harness selectors are not interchangeable. [Selection](doc-python-api-configuration-and-context.md#framework-and-harness-selection) |
| `type`, `tool`, `Constraints` | Class registration, callable metadata/retries/risk, validation-only field metadata. [Types and Tools](doc-python-api-types-and-tools.md) |
| `Capability` | Reusable tools, guidance, and owned policies applied with `> use:`; requires supporting adapters. [Capability Definitions](doc-python-api-capabilities.md) |
| `approval`, `ApprovalPolicy`, `ApprovalMode`, `ApprovalHandler`, `ApprovalRequest`, `ApprovalDecision`, `ApprovalCancelled` | Policy registration, static/dynamic policy types, immutable request and allow/deny/edit results; cancellation is not denial. [Approvals](doc-python-api-approvals-mcp-and-skills.md) |
| `McpServerSpec`, `SkillsSettings` | Normalized server transport specification and scoped skill-discovery policy. [MCP](doc-python-api-mcp.md), [Skills](doc-python-api-skills.md) |
| `CodeModeSettings` | Activation, preloaded tool names, and limits for the scoped tool catalog. [CodeMode](doc-agentic-engineering-codemode.md) |
| `on`, `HookSettings`, `HookHandler`, `HookEvent`, `HookEventName`, `HookDecisionKind` | Registration and event/decision aliases, not arbitrary provider callbacks. [Hooks](doc-agentic-engineering-hooks.md) |
| `HookContext`, `ToolHookContext`, `ToolOrigin` | Run lineage and tool-origin metadata supplied by runtime; do not infer authorization from origin. [Hooks](doc-agentic-engineering-hooks.md) |
| `UserPromptSubmitRequest`, `UserPromptSubmitDecision`, `PreToolUseRequest`, `PreToolUseDecision`, `PostToolUseEvent`, `PostToolUseFailureEvent`, `PostToolResultFailureEvent`, `ToolCallRecord` | Typed hook payloads and body/result evidence; only pre-operation decisions can edit/deny, post events cannot undo effects. [Hook payloads](doc-agentic-engineering-hooks.md) |
| `HookDeniedError`, `HookExecutionError`, `HookTimeoutError` | Denial, handler failure, and timeout remain distinct failure paths. [Hooks](doc-agentic-engineering-hooks.md) |
| `AgentMessageEvent`, `AgentMessagePhase`, `AgentRunState`, `AgentRunStateEvent`, `AgentStreamEvent` | Semantic message/run-state events and discriminants, not token streams. [Stream Events](doc-agentic-engineering-stream-events.md) |
| `AsyncAgentEventQueue`, `observe_agent_events`, `bind_agent_event_dispatcher` | Async queue and scoped event observer/dispatcher; own and join producer tasks explicitly. [Stream Events](doc-agentic-engineering-stream-events.md) |
| `SubagentExecutionContext`, `SubagentUsageLimits`, `current_subagent_execution` | Current child lineage (or `None` outside a child) and request/tool/token ceilings. [Python Embedding](doc-agentic-engineering-subagent-python.md) |
| `SubagentScope`, `SubagentTask`, `SubagentResult` | Invocation-owned child scope, task handle with async `wait()`, and completed result with typed `output`. [Python Embedding](doc-agentic-engineering-subagent-python.md), [Native Tasks](doc-agentic-engineering-subagent-tasks.md) |
| `AgentInput`, `InputReceipt` | Runtime-owned input identity and routing, plus queued, interrupt-requested, or rejected admission receipts; admission does not confirm delivery. [Task Input and Redirection](doc-agentic-engineering-subagent-messaging.md) |
| `RunBudget`, `RunBudgetExceeded` | Immutable root/descendant request and managed tool-attempt ceilings; admission is atomic. [Run Budgets](doc-runtime-run-budgets.md) |
| `ConversationState`, `session` | Model history and artifact ownership, not incremental variable scope. [Sessions](doc-python-api-artifacts-and-sessions.md) |
| `ArtifactPolicy`, `ArtifactRef`, `ArtifactChunk`, `ArtifactSearchResult`, `ArtifactReleaseResult`, `ArtifactHandle`, `ArtifactStream` | Policy, bounded transport/results, internal native handle, explicit single-use stream. [Artifacts](doc-runtime-tool-artifacts.md), [Retrieval](doc-runtime-artifact-retrieval.md), [Lifetime](doc-runtime-artifact-lifecycle.md) |
| `DecisionCapture`, `DecisionInfo`, `DecisionSource`, `capture_decisions`, `decision_info` | Historical evaluation metadata, capture scope, and binding lookup; no extra model requests. [Decisions](doc-python-api-decisions.md) |
| `CodexModel`, `CodexModelAdapter`, `CodexResponsesConnection`, `CodexResponsesFallback`, `codex_responses_model` | Model factory/result aliases, framework choice, HTTP/WebSocket and explicit fallback policy. [Codex Models](doc-agent-adapters-codex-models.md) |
| `CacheInfo`, `cache_info`, `clear_cache` | Process-memory cache counts and invalidation; no provider cache control. [Cache Control](doc-python-api-cache-control.md) |
| `KediRuntime`, `AgentSurface`, `i`, `o`, `c` | Compiled container, bound agent lifecycle, and input/output/call expression constructors. [Embedding](doc-python-api-embedding.md), [Runtime](doc-python-api-caching-runtime-and-executors.md) |
| `KediPromise`, `KediPromiseLeak`, `force` | Deferred value, erroneous concrete use and explicit resolution. [Promises](doc-python-api-concurrency-and-promises.md) |
| `InteractiveSession`, `dump_session`, `load_session`, `SessionPersistenceError`, `SessionDumpError`, `SessionLoadError` | Incremental execution and strict snapshot serialization/errors. [Incremental Execution](doc-runtime-interactive-execution.md) |
| `LoopIterationLimitError` | Conditional-loop guard exceeded, not normal completion. [Loops](doc-core-language-loops-and-map.md) |
| `Executor`, `DefaultExecutor`, `ExecutorDebugEvent`, `ExecutorDebugExporter`, `MarkdownDebugExporter`, `default_debug_export_path` | Python execution protocol and diagnostic exports; default execution is unsandboxed and exports are not secret-redacted. [Executors](doc-python-api-executors.md) |

`kedi.decisions` additionally exports `Probability`, `Rubric`, `BooleanCriteria`,
and `ChoiceCriteria` from the optional `kedi-decisions` package. Importing these
types requires that package, but ordinary decision-evidence inspection does not.
The criteria are shared by Jev and Laya, without importing either provider SDK;
see [Decision Criteria](doc-agent-adapters-jev-criteria.md).
