# Telemetry

Kedi provides a dependency-free telemetry seam in the runtime. It is a no-op
until the separate `opentelemetry-instrumentation-kedi` package installs an
OpenTelemetry backend. Application code keeps ownership of the OpenTelemetry
SDK, resource, sampler, exporter, and shutdown lifecycle.

## Installation

Install the instrumentor beside Kedi:

```bash
pip install opentelemetry-instrumentation-kedi
```

Enable it once during application startup:

```python
from opentelemetry.instrumentation.kedi import KediInstrumentor

instrumentor = KediInstrumentor()
instrumentor.instrument()
```

Configure the tracer and meter providers with `service.name=kedi`. The
instrumentor uses three instrumentation scope names:

| Scope | Responsibility |
| --- | --- |
| `kedi.runtime` | parsing, compilation, execution, procedures, and embedded Python |
| `kedi.agent` | agents, models, tools, MCP, approvals, subagents, and workflows |
| `kedi.artifacts` | artifact admission, storage, reads, release, expiry, and cleanup |

Importing Kedi or the instrumentor does not configure an exporter and does not
start telemetry by itself.

## Logfire

For Pydantic AI tracing from a Kedi program, the registry's
[Logfire package](https://registry.kedi-lang.org/package/logfire) installs its
Python dependencies alongside Kedi:

```bash
kedi add logfire
```

```kedi
> import: logfire

`configure_logfire("my-service")`
```

This explicitly configures Logfire and Pydantic AI model/tool instrumentation;
model content capture is disabled by default. Pass `True` as the third argument
to enable HTTPX spans. Configure before making model calls and use
`flush_logfire()` before exiting a short-lived program. `LOGFIRE_TOKEN` selects
the destination project; without a token, the default does not send to Logfire.

For full Kedi runtime, agent and artifact instrumentation, use the Kedi
instrumentor below instead of calling `configure_logfire` for the same process.

Logfire accepts the native OpenTelemetry data emitted by the instrumentor:

```python
import logfire
from opentelemetry.instrumentation.kedi import KediInstrumentor

logfire.configure(service_name="kedi")
KediInstrumentor().instrument()
```

Pydantic AI instrumentation is enabled by default, so Pydantic AI model and
tool spans appear at their native detail beneath the Kedi operation that owns
them. Kedi reclassifies those native spans with Kedi agent semantics instead of
emitting duplicate agent, chat, and tool spans.

HTTPX instrumentation is disabled by default. Install the extra and enable it
when outbound request spans are useful:

```bash
pip install "opentelemetry-instrumentation-kedi[httpx]"
```

```python
KediInstrumentor().instrument(instrument_httpx=True)
```

Do not separately call `logfire.instrument_pydantic_ai()` when the Kedi
instrumentor owns that integration. HTTPX setup is process-wide and one-way;
Kedi does not uninstrument HTTPX during teardown because doing so could remove
instrumentation installed by another library.

## Configuration

`instrument()` accepts strict, validated options:

```python
KediInstrumentor().instrument(
    runtime_detail="lifecycle",  # "off", "lifecycle", or "detailed"
    agent_enabled=True,
    artifacts_enabled=True,
    capture_content=False,
    capture_binary_content=False,
    capture_source_paths=False,
    capture_source_snippets=False,
    capture_model_request_parameters=False,
    capture_tool_definitions=False,
    capture_exception_messages=False,
    capture_exception_stacktraces=False,
    max_metric_instruments=128,
    instrument_pydantic_ai=True,
    instrument_httpx=False,
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)
```

Unknown option names, non-boolean values for boolean options, and invalid
`runtime_detail` values fail immediately. The three scope switches behave as
follows:

- `runtime_detail="off"` disables all runtime spans and metrics.
- `runtime_detail="lifecycle"` records bounded parse, compile, and run
  lifecycle operations while omitting detailed procedure and Python spans.
- `runtime_detail="detailed"` also records procedure calls and embedded Python
  execution.
- `agent_enabled=False` disables the complete agent surface, including model
  and tool calls, MCP, approvals, subagents, and dynamic workflows.
- `artifacts_enabled=False` disables artifact spans, events, and metrics.

`max_metric_instruments` bounds the number of metric instruments the backend
may create. This prevents unbounded instrument growth if an extension supplies
unexpected metric names.

## Span Hierarchy

Span names are concise operation labels intended for trace UIs. Machine-facing
semantics remain in `kedi.operation.name` and standard `gen_ai.*` attributes.
A representative agent run is shaped like this:

```text
kedi run program.kedi
`-- agent researcher
    |-- chat openai:gpt-6-luna
    |-- call web_search
    |-- await approval write_file
    |-- planner run
    `-- researcher workflow
```

The principal spans are:

| Span name | Operation | Notes |
| --- | --- | --- |
| `kedi run <program>` | `run_program` | complete program execution |
| `parse <program>` | `parse_program` | source parsing and diagnostics |
| `compile <program>` | `compile_program` | declarations, profiles, and runtime construction |
| `compile module <name>` | `compile_module` | imported-module compilation |
| `call <procedure>` | `call_procedure` | detailed procedure execution |
| `run python` | `run_python` | detailed inline, fenced, prelude, or type Python execution |
| `agent <profile>` | `run_agent` | one semantic Kedi agent call |
| `chat <model>` | `chat` | adapter/model boundary when no native owner exists |
| `call <tool>` | `call_tool` | approved tool execution and output admission |
| `initialize mcp <server>` | `initialize_mcp` | MCP connection and tool discovery |
| `await approval <tool>` | `await_approval` | blocking CLI, Python, or native approval decision |
| `run <event> hooks` | `run_hooks` | one nonempty lifecycle-handler chain |
| `<profile> run` | `run_agent` | a subagent run; replaces the delegation tool wrapper |
| `<profile> workflow` | `run_workflow` | a dynamic workflow run |
| `search tools` | `search_tools` | bounded CodeMode catalog discovery |
| `get tool schema` | `get_tool_schema` | exact CodeMode schema hydration |
| `execute code` | `execute_code` | one bounded Monty notebook cell and its nested calls |
| `process history` | `process_history` | deterministic Kedi history selection after the reduction threshold is reached |
| `compact history` | `compact_history` | one actual native or Kedi-owned compaction attempt |
| `store artifact` | `store_artifact` | material artifact storage |
| `cleanup artifacts` | `cleanup_artifacts` | scheduled or explicit cleanup |

Lightweight artifact searches, reads, releases, expirations, and context
preflight decisions are events on the active operation rather than extra nested
spans. This keeps traces readable during artifact-heavy runs.

## Attributes and Events

Kedi uses OpenTelemetry GenAI semantic attributes where they apply:

- `gen_ai.agent.name`, `gen_ai.request.model`, and `gen_ai.output.type` describe
  agent and model calls;
- `gen_ai.tool.name` identifies the invoked tool;
- `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` report observed
  usage;
- `kedi.adapter.name`, `kedi.adapter.kind`, and `kedi.agent.call.kind` identify
  the Kedi boundary;
- `kedi.tool.source`, `kedi.tool.risk`, and
  `kedi.tool.result.artifactized` describe tool execution;
- `kedi.approval.mechanism`, `kedi.approval.risk`, and
  `kedi.approval.decision` describe approval without recording arguments;
- `kedi.hook.event`, `kedi.hook.handler_count`, `kedi.tool.origin`, and
  `kedi.hook.outcome` describe a lifecycle chain without recording payloads;
- `kedi.subagent.*` and `kedi.workflow.*` carry run identity, depth, budget,
  concurrency, outcome, and child-call counts;
- `kedi.codemode.*` carries catalog, hydration, code-size, restart, nested-call,
  nested-result-byte, duration, and outcome observations without code or tool
  payloads;
- `kedi.history.compaction.*` carries mode, trigger, input/output message and
  token counts, reduction ratio, retained-prefix validation, checkpoint state,
  and whether the cache epoch changed;
- `kedi.artifact.*` carries sizes, offsets, lifecycle outcome, and context bytes
  avoided without exposing the payload.

Events mark meaningful transitions such as `agent request prepared`, `tool
execution started`, `tool output admitted`, `approval requested`, `approval
pending`, `mcp connected`, `artifact released`, and `context preflight`.
Compaction uses `compaction triggered`, `checkpoint validated`, `checkpoint
committed`, `cache epoch advanced`, and `compaction rejected`. A
`compact history` span means a backend was actually invoked or a validated Kedi
checkpoint commit was attempted; applying provider request settings alone does
not emit it. Below-threshold history checks do not create spans.

## Metrics

Runtime metrics include:

- `kedi.run.count`, `kedi.run.duration`, and `kedi.run.active`;
- `kedi.parse.count` and `kedi.parse.duration`;
- `kedi.compile.count` and `kedi.compile.duration`;
- `kedi.procedure.calls` and `kedi.procedure.duration`;
- `kedi.python.execution.count` and `kedi.python.execution.duration`.

Agent metrics include:

- `kedi.agent.invocations` and `kedi.agent.duration`;
- `gen_ai.client.operation.duration` and `gen_ai.client.token.usage`;
- `kedi.tool.calls` and `kedi.tool.duration`;
- `kedi.approval.requests` and `kedi.approval.wait.duration`;
- `kedi.hook.calls`, `kedi.hook.duration`, `kedi.hook.edits`,
  `kedi.hook.denials`, `kedi.hook.failures`, and `kedi.hook.timeouts`;
- `kedi.mcp.initializations` and `kedi.mcp.initialization.duration`;
- `kedi.codemode.operations` and `kedi.codemode.duration`;
- adapter, subagent, and workflow duration/count observations carried by their
  respective operations.

History compaction metrics include:

- `kedi.history.compactions` and `kedi.history.compaction.duration`;
- `kedi.history.compaction.input_tokens` and
  `kedi.history.compaction.output_tokens`;
- `kedi.history.compaction.reduction_ratio`;
- `kedi.history.compaction.failures`.

Compaction metric dimensions are limited to mode, trigger, and result. Provider
cache-read and cache-write token usage is already reported by agent/model
telemetry and is deliberately not duplicated by compaction metrics.

Artifact metrics include:

- `kedi.artifact.created`, `kedi.artifact.released`, and
  `kedi.artifact.expired`;
- `kedi.artifact.active` and `kedi.artifact.active_bytes`;
- `kedi.artifact.reads` and `kedi.artifact.read_bytes`;
- `kedi.artifact.context_bytes_avoided`;
- `kedi.artifact.quota_rejections` and
  `kedi.artifact.cleanup_duration`;
- `kedi.tool.output_admissions` and `kedi.tool.output_size`.

Outcome, adapter, call kind, tool source/risk, transport, and token type are
bounded metric dimensions. Prompt text, paths, arbitrary tool names, and
artifact IDs are not metric dimensions.

## Privacy and Content Capture

The default configuration does **not** capture:

- prompt or model output content;
- binary content;
- source paths or source snippets;
- model request parameters;
- tool definitions;
- exception messages or stack traces.

Each category has an independent opt-in. Exception type remains observable
without exposing the message. Captured exception messages and stack traces are
bounded. Content options should be enabled only after reviewing provider data,
tool arguments, source code, and credentials that may enter telemetry.

## Lifecycle and Ownership

Instrumentation setup and teardown are serialized and idempotent.
`instrumentor.uninstrument()` restores the previous Kedi telemetry backend only
if the active backend is still the one installed by that instrumentor. It also
restores Pydantic AI instrumentation only while Kedi still owns the exact
setting. Newer application-owned configuration is never overwritten.

If one teardown step fails, Kedi attempts the remaining cleanup steps and then
raises `KediInstrumentationCleanupError` with every cause. If setup fails, the
setup exception remains primary and rollback failure is chained beneath it.

Kedi deliberately keeps tracing optional: without an installed backend, span,
event, and metric calls remain no-ops and do not require OpenTelemetry at
runtime.
