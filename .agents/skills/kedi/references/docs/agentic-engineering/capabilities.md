# Capabilities

!!! note "Unreleased"
    This page describes the next Kedi release. Eager composition is supported by
    Pydantic AI and LangChain; deferred loading is supported by Pydantic AI only.

A **profile** selects agent configuration. A **capability** contributes reusable
instructions, tools, and tool-local policies without replacing the host model or
base instructions. Both are applied with `> use:`. Neither starts a model call.

Use a [subagent](subagents.md) when the contribution needs a separate model,
conversation, or execution loop.

## Compose Instructions

```kedi
> capability: evidence:
    ###
    Separate observed facts from assumptions when assessing readiness.
    ###
    > instructions: Use the supplied evidence and identify unfinished checks.

> capability: concise:
    > instructions: Keep the explanation to one sentence.

> profile: release_reviewer:
    > instructions: Assess whether the release can proceed.
    > use: evidence
    > use: concise

> use: release_reviewer
>> Unit tests passed, but staging verification is still running.
The release readiness is [readiness: Literal["ready", "pending"]],
because [reason: str].
= `{"readiness": readiness, "reason": reason}`
```

The request receives the profile's release-assessment instruction followed by
the evidence and concise contributions. The docstring describes the capability
in the editor and, when deferred, in its discovery catalog. The example needs
a configured model, just like any other template program.

Use `> use:` inside a profile to compose capabilities. There is no separate
`> capabilities:` directive. A capability may itself use another capability or
a tool, but cannot use a profile to import host-wide configuration indirectly.

## Reset, Then Reuse

Using the definitions above:

```kedi
> use: release_reviewer
> instructions: Report only the observed deployment state.
>> Staging verification is still running, so its status is [status: str].

> use: concise
>> The current status of <status> means [explanation: str].
```

The first request has only the new user instruction. The second also has the
concise contribution. An explicit instruction assignment clears prior user
guidance even when the new base text equals the old text.

Resetting instructions does **not** remove tools, capability approval, hooks,
history, or runtime instructions such as the template-filling contract. Reusing
a capability restores its guidance once; repeated use does not duplicate it.
Deduplication uses capability identity, not matching instruction text.

Profiles still override. A profile with explicit instructions replaces the
current user instruction composition; a model-only profile does not. Profile
members follow source order: instructions after capability use clear earlier
guidance; capability use after instructions appends to the new base.

## Contribute Tools

This exportable review module is a small building block for a harness. It reads
a caller-supplied source file and returns a typed finding, not a complete coding
agent or a sandbox.

````kedi
```
from pathlib import Path
```

@read_source(path: str) -> str:
    ###
    Read a UTF-8 source file for inspection.
    ###
    = ```
    source = Path(path)
    return source.read_text(encoding="utf-8")
    ```

> capability: source_reader:
    > instructions: Read the named file before describing its implementation.
    > approval: allow
    > use: read_source

> instructions: Review code without changing files.
> use: source_reader

@review_file(path: str) -> str:
    >> The most significant concrete issue in <path> is [finding: str].
    = <finding>

> export:
    source_reader
    review_file
````

The local allow policy belongs only to `read_source`; it does not allow unrelated
host tools. An explicit policy on the exposed tool takes precedence, then the
capability policy, then the calling agent's default. Independent mandatory
enforcement still applies. A procedure's internal `> approval:` governs its
internal agent calls, not whether the outer caller may invoke that procedure.

Tools retain their concrete definition-site bindings when imported. Exporting
the capability does not export its private procedures separately. Incompatible
ownership or policy for the same tool name is an error, not last-writer-wins.

The example reads paths allowed by the process. Instruction text and capability
names do not impose filesystem restrictions; enforce those in the tool or its
execution boundary when needed.

## Local Procedure Instructions

Procedures capture the configuration visible at their definition site. A local
`> instructions:` replaces that invocation's inherited user guidance, while
captured tools remain available. A later local capability use appends there.
Returning, failing, or cancelling the procedure does not alter its caller.

This is not a fresh conversation and does not erase history. A procedure declared
before a later base instruction does not acquire that later instruction, even
when it had no base instruction at its definition site. The same rule applies
across notebook and IDLE cells.

A procedure with no model binding still falls back to the invoking agent's
model, as before. That inference fallback does not import the caller's later
instructions or tools; an explicitly captured model continues to win.

## Deferred Loading

Using the source tool defined above:

```kedi
> capability: source_reader_on_demand:
    ###
    Read source files when implementation evidence is required.
    ###
    > defer: true
    > instructions: Read the requested source file before answering.
    > use: read_source

> use: source_reader_on_demand
```

Without `defer`, use is eager: tools and guidance are available for the next
request without a discovery call. With `defer: true`, the native Pydantic AI
loader exposes a catalog entry first. Tools cannot execute until activation,
even if the model guesses their names. Repeated loads are idempotent.
This means registrations and activation are not duplicated; the native loader
may return a retry response for an already-active capability.

`defer` also accepts an inline Python expression returning an actual boolean.
Kedi evaluates it once per use context, not on every model retry. Strings and
numbers are not accepted as truthy substitutes.

The initial adapter contract is Pydantic AI eager/deferred and LangChain eager
only. Unsupported adapter combinations fail before inference. Nested deferred
capabilities and deferred subagent targets are supported on Pydantic AI.

Deferred instructions can remain in historical load results after a reset.
Reset suppresses their automatic reinjection; it does not make the model forget
history. Explicit reuse restores guidance only after native activation. Native
MCP setup/catalog discovery may occur before loading, so deferral is not a
promise of lazy connections or zero resource cost.

### Nested Capabilities

Using `read_source` from the earlier example:

```kedi
> capability: implementation_evidence:
    > defer: true
    > instructions: Read the named source before making implementation claims.
    > use: read_source

> capability: code_review:
    > defer: true
    > instructions: Distinguish observed defects from suggestions.
    > use: implementation_evidence

> use: code_review
```

Initially only `code_review` is discoverable. After the native loader confirms
its activation, `implementation_evidence` becomes discoverable. Loading that
second capability makes `read_source` callable. Loading the parent does not
automatically load its deferred child; parent and child load calls in one model
response do not skip that boundary. Eager child contributions become available
with their deferred parent.

A shared child has one stable registration even when multiple parents include
it. Any activated parent makes it discoverable. Explicit direct use also makes
that child discoverable independently of its parents. These are contribution
and loading relationships, not a sequence of tasks or separate agents.

### Deferred Subagent Targets

```kedi
> profile: evidence_reviewer:
    > instructions: Review supplied evidence and identify unsupported conclusions.

> capability: specialist_review:
    > defer: true
    > subagent: evidence_reviewer

> use: specialist_review
```

The child profile is captured when the capability is used. Its target becomes
available to the shared delegation or dynamic-workflow tools only after loading
`specialist_review`. Guessing the target before activation does not start a child
or invoke its capability-owned hooks. If the host already has other subagents,
their shared controls remain available, with only those active targets exposed.

Background work, waiting, continuation, and messaging keep the captured binding
and ownership checks. The subagent's own model and instructions remain those of
its profile. CodeMode catalogs and schema results obey the same target visibility
and activation rules as native calls.

Reset suppresses guidance throughout the nested contribution graph while retaining
registrations. Reuse restores guidance once, still gated on native activation.
If two deferred parents share a child and only one parent is reused after a
reset, the child's guidance returns only after that parent and the child are
loaded. Loading the other parent alone does not restore the child's guidance;
the order of the parent declarations does not change this behavior.
Each run owns its activation; sharing an adapter or descriptor does not share
loaded state. Revealing a nested catalog changes its instruction prefix, so this
path does not guarantee prefix-cache preservation.

## Conditional Contributions

Use deterministic guards to choose an instruction/tool bundle or its loading
policy. The capability still contributes resources, not agent-wide overrides:

```kedi
[needs_citations: bool] = `True`

> capability: evidence_policy:
    > if: `needs_citations`:
        > instructions: Support factual conclusions with supplied sources.
    > else:
        > instructions: Clearly distinguish suggestions from observed facts.

> use: evidence_policy
```

The Python expression must return an exact `bool` and is evaluated once per
application. Only the selected branch's computed values and uses run. Nested
guards and optional `> else:` follow the same rules as
[conditional profile configuration](profiles.md#conditional-configuration).
Selected instructions remain additive to the host's base instructions. Selected
approval and hooks remain limited to this capability's owned tools. A selected
`> defer: true` makes the resulting capability load on demand; it does not defer
the guard itself. Unselected tools are not registered.

All branches must respect the configuration boundary below, including branches
whose guard is false. A condition is not a way to add model selection or other
caller-owned settings to a capability.

## Configuration Boundary

| Capability member | Effect |
| --- | --- |
| `instructions` | Owned instruction contribution; local replacements affect only its bundle |
| `use`, `mcp`, `subagent` | Contribute tools, MCP sources, or child-profile bindings |
| `approval` | Override the default for owned concrete tools only |
| `hooks` | Owned tool lifecycle events, retaining mandatory enforcement |
| `requires` | Declare adapter requirements, not permissions |
| `defer` | Choose immediate contribution or native on-demand loading |

Host `model`, `effort`, `settings`, `agent`, `adapter`, `output`, `history`,
budgets, skills discovery, CodeMode policy, and artifacts are rejected at the
capability root. `user_prompt_submit` hooks and `tool_reason` remain agent-wide.

Use the calling agent's artifact `exclude_tools` for exact-name exceptions.
Exclusion does not disable approval, reason, output limits, or tracing. See
[Artifact Policy](../runtime/artifact-policy.md) and [Hooks](hooks.md).

## Inspect Behavior

Telemetry records application/reuse, instruction replacement, request snapshots,
native load attempts and activation, tool ownership and approval source, and
artifact admission decisions. CodeMode retains the actual inner tool span.
Capability IDs are stable hashes; source locations require source-path capture.
Instruction content follows the existing content capture policy.

Deferred loading can change request prefixes. Cache reuse depends on the
provider and transport; it is not guaranteed. Capability loading does not require
schema search or CodeMode, and is not automatically more efficient than a small
eager tool set.

See [Python Capability Definitions](../python-api/capabilities.md) for the same
composition contract through `kedi.Capability`, and the
[adapter capability matrix](../reference/capability-matrix.md) for support checks.
