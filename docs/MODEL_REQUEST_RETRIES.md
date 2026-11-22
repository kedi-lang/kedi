# Model Request Retries

Pydantic and LangChain retry transient model-request failures without restarting
the agent, template, procedure, or a previously completed tool. Both adapters
default to two additional attempts for HTTP 408, 429, and 5xx errors.

Configure the active profile's policy in Kedi:

```kedi
> settings:
    request_retries: 2
```

Use `request_retries: 0` to disable Kedi's same-model retries; an explicitly
configured fallback can still run. The setting is a non-negative integer,
accepts computed backtick expressions, and follows normal
profile override rules. It is runtime-owned and is not forwarded as a provider
model parameter. Other adapters do not advertise this setting.

Python integrations can set the adapter default:

```python
adapter = PydanticAdapter(model, request_retries=2)
adapter = LangChainAdapter(model, request_retries=2)
```

An active profile setting overrides that constructor default. This policy is
independent of tool-body retries, malformed-tool-argument corrections, and
structured-output corrections.

## Failure and Budget Boundaries

- Only the failed model request is retried. Earlier tool results and history
  remain available; successful tool bodies and approvals are not replayed.
- Child requests use the same policy. Retrying a request inside a child does
  not start another child or consume another `max_agents` slot.
- `max_agents` continues to count child starts, including children that fail
  after request retries are exhausted. There is no automatic budget refund.
- Request limits and strict `RunBudget` request ceilings still apply. Failed
  HTTP attempts are included in invocation request usage, without inventing
  token usage for a rejected request.
- Native fallback attempts are included once, including nested fallback chains.
  Measured token usage remains the SDK's reported usage; an unsuccessful HTTP
  request does not manufacture token usage.
- A LangChain history processor's continuation reset is retained across retries.
  Retrying cannot reconnect discarded history through a stale response ID or
  server conversation. Subsequent successful turns resume normal continuation.
- Waiting for retry backoff is cancellable and remains inside the enclosing
  run/child deadline. Authentication, validation and ordinary application
  failures are not treated as transient HTTP errors.
- Pydantic AI's native request-attempt hooks handle stream-opening failures;
  a failure after streamed output has begun is not replayed. Plain network
  stream disconnections are not classified as transient HTTP statuses.

Without a valid `Retry-After`, delays are 0.5 seconds, then 1 second, increasing
exponentially up to 30 seconds for explicitly larger retry counts. Numeric and
HTTP-date `Retry-After` headers are honored. If the provider requires more than
30 seconds, Kedi propagates the original error instead of retrying too early
or keeping the run waiting indefinitely. An eligible failure can still advance
to a different configured fallback candidate without retrying that provider early.

## Ordered Model Fallback

Both framework adapters support one alternative or an ordered block:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
> fallback:
    openai:gpt-6.1-sol
    openai:gpt-6-luna
> settings:
    request_retries: 1

>> The capital of Turkey is [capital: str].
= <capital>
```

The same declaration works with `> adapter: langchain`. Fallback is an
independent lexical setting, not a requirement to follow `> model:` immediately.
It uses the effective inherited primary, including Python/environment selection;
an unresolved primary is an error, not a reason to promote the first backup.
Profiles, conditional profile members, procedures, and subagent profiles use
normal capture and override rules. Each list replaces the previous one, a later
model directive preserves it, and local overrides do not mutate callers.
``> fallback: `[]` `` clears the list; `none` is not a generic reset literal.
Computed values must be model IDs or lists/tuples of IDs, validated before I/O.
Harness adapters reject the setting; capabilities cannot own model selection.

Try the primary first, applying its bounded `request_retries` policy to HTTP
408, 429, and 5xx failures. After eligible exhaustion, advance in declaration
order and apply that policy to each candidate. No whole-run retry is added.
Authentication errors, refusals, argument/output validation failures, tool-body
failures, budget exhaustion, cancellation, and deadline expiry do not advance
the chain. Exhausting all candidates raises a terminal failure with attempt
context.

Each actual provider attempt, including a retry or dynamically selected backup,
must pass admission against the same request budget. Failed attempts count once;
refused admissions never invoke the provider or increment attempt usage. Measured
tokens are reported once, without inventing usage for HTTP failures. Usage limits,
tool limits, child-start limits, and deadlines are not reset by model selection.
For example, with `request_retries: 0`, `RunBudget(request_limit=1)` admits only
the primary; a limit of `2` can admit one backup after an eligible primary failure.

Only the failed model request changes models. Existing history, completed tool
results, approvals, and delegation state remain; no template, tool, child, agent,
or program is restarted. Streaming recovery is limited to opening failures before
response content reaches the caller. A partially delivered stream is never
replayed or spliced to another model's output.

Portable history survives a model change, but incompatible response IDs,
server-owned conversation state, and cache settings must not cross providers.
History-processor continuation resets remain effective. Incompatible required
server state, output/tool/media contracts, or explicit settings fail rather than
silently losing context or weakening the call. Selection is invocation-local;
concurrent calls retain their own primary models.

Python adapter constructors accept `fallback_models=[...]` containing model IDs
or native models from their framework. `AgentProfile.fallback_models` is the
shared `tuple[str, ...] | None` field: `None` inherits, a tuple replaces, and `()`
clears the chain. A profile override does not mutate the constructor's clients.

## Provider SDK Retries

Kedi does not silently mutate a user-supplied SDK client's retry configuration.
Its recovery happens after that client raises, so SDK retries and Kedi retries
can compound. Set SDK `max_retries=0` when Kedi should own retry attempts, for
example on `AsyncOpenAI` or `ChatOpenAI`.

A strict Kedi request budget continues to reject opaque or enabled SDK retries:
otherwise attempts hidden inside the SDK cannot be admitted or counted at the
model boundary. `request_retries: 0` disables Kedi retries only; it does not
disable an SDK's own retry policy.

Both native `pydantic_ai.capabilities.ModelFallback` selections and ordinary
`pydantic_ai.models.fallback.FallbackModel` chains apply Kedi's admission guard
to every selected model, including dynamically selected backups. Admission does
not depend on enabling Kedi request retries: a native `Agent` using Kedi's
`budget_model` wrapper also guards capability-selected backups without the retry
capability. With `RunBudget(request_limit=1)`, a failed primary spends the only
slot and `RunBudgetExceeded` prevents the backup from starting. Refused admission
adds no provider attempt or token usage; admitted attempts are accounted once.

The opt-in `JobManagerEngine` retry policy retries whole jobs and is a separate
contract. Do not use whole-job retries as a substitute for request recovery
when a job may have already performed external side effects.

## Child Failure Feedback

A child whose execution ends unsuccessfully reports a terminal tool failure
to its calling model, without spending the tool-argument correction budget.
Malformed delegation arguments still use the bounded correction path. Run
budgets and cancellation remain control signals, not ordinary failure results.

Waiting on a known failed child returns repeatable terminal status and
sanitized error information. A caller wait timeout while the child is running
still returns a running handle. Direct coordinator and native Python task
waits keep their exception semantics.

The default fail-closed delegation ledger remains active: reporting a failure
does not permit the parent to silently claim successful completion. A matching
successful replacement is still required unless the caller explicitly selects
the recovery policy.

## Verification

`tests/test_model_request_retry.py` covers both adapters, transient and
permanent statuses, finite attempts, profile overrides, usage accounting,
request ceilings, deadlines, retry headers, and a single-start child recovery.
Its HTTP transport tests use the real OpenAI SDK and adapter models with
`httpx.MockTransport`; no live provider or API key is required. They verify
identical retried history and an exactly-once tool side effect.

`tests/test_langchain_request_retry_history.py` checks rewritten Responses API
history, continuation reset retention, per-attempt admission and concurrent
scope cleanup. `tests/test_pydantic_request_retry_fallback.py` checks fallback
accounting, including identical errors, nested models, native fallback
capabilities, streaming and concurrent runs.

`tests/test_subagent_tool_failures.py` covers terminal child feedback, malformed
arguments, repeatable failed waits, fail-closed behavior, and control signals.
