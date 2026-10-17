> Offline source: `kedi-docs-main/docs/agentic-engineering/profiles.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Define a Profile](#define-a-profile)
- [Profile Documentation](#profile-documentation)
- [Apply a Profile](#apply-a-profile)
- [Merge Rules](#merge-rules)
- [Default Child Result Type](#default-child-result-type)
- [Export and Import Profiles](#export-and-import-profiles)
- [Forward References and Validation](#forward-references-and-validation)
- [Child Profiles](#child-profiles)
- [Choosing Profile Boundaries](#choosing-profile-boundaries)

# Profiles

Profiles can also select a lexical native history processor through
`> history:` with `enabled: true` and ``processor: `callback` ``. This overrides
the adapter constructor default only for the scoped invocation. See
[History Processing](doc-runtime-history.md#native-configuration) for callback
types, explicit disable, safe group retention, and lineage.

Profiles are named, reusable agent configuration. A profile describes how a
model call should run: its backend, instructions, tools, permissions, and optional
children. Defining or applying a profile does not start a model call, create a
conversation, or run its tools.

A procedure describes executable steps; a profile supplies configuration to
those steps. A child profile describes a separate agent invocation that a parent
may request. These are different roles even when they share the same tools.

[Capabilities](doc-agentic-engineering-capabilities.md) are the separate additive surface for reusable
tool and instruction bundles. A profile may use them with `> use:` without
changing its own override semantics.

## Define a Profile

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> profile: reviewer:
    ###
    Reviews one change against repository evidence.
    ###
    > adapter: pydantic
    > model: openai:gpt-5.6-luna
    > effort: medium
    > instructions: Cite the evidence used for every finding.
    > settings:
        timeout: 120
```

A profile body contains directives, not an executable procedure body:

| Concern | Members | Meaning |
| --- | --- | --- |
| Backend | `adapter`, `agent`, `model`, `effort` | Select execution backend and model behavior. |
| Instructions | `instructions`, `settings` | Supply instructions and supported runtime/model settings. |
| Tools | `use`, `mcp`, `skills` | Expose procedures, external tools, and skill discovery. |
| Permissions and lifecycle | `approval`, `hooks` | Mediate sensitive calls and lifecycle boundaries. |
| Context | `history`, `artifacts` | Configure conversation retention and artifact handling. |
| Tool execution | `codemode` | Enable the code-based tool interface. |
| Child agents | `subagent`, `max_agents`, `workflow`, `output` | Define delegation, its budget, orchestration mode, and a default child result type. |

Choose either `adapter` or `agent` within a profile, not both. Profiles are
top-level declarations. Tools referenced by the profile must exist; declaring
their names does not implement them.

## Profile Documentation

When a `###` block is the first profile member, it becomes the profile
docstring. The LSP shows it in hover and subagent tool descriptions. A later
block remains an ordinary comment.

Document what task the profile owns, what evidence it should return, and any
important limitation. Child-agent descriptions depend on this text.

## Apply a Profile

Single-line `> use:` applies a profile when no procedure with that name exists:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> use: reviewer

>> A review of the current change is [review: str].
= <review>
```

If a procedure and profile share a name, the procedure wins and is registered
as a tool. Avoid such collisions.

## Merge Rules

Applying a profile merges it into the state already active at that location.
It is not a reset to an empty configuration:

| Member | Composition rule |
| --- | --- |
| Backend, model, effort, instructions, approval, output type, maximum starts, workflow mode | A specified value replaces the earlier value. Instructions are replaced, not concatenated. |
| Settings | Merge by key; later values replace the same key. Nested dictionaries are not recursively merged. |
| Tools and children | Merge by name. Later bindings win, and reintroduced names move to the end of the ordered list. |
| MCP servers | Append in declaration/application order. |
| Hooks | Compose per event; a supplied empty handler list clears that event's local handlers. Inherited enforcement is a separate boundary. |
| Skills, CodeMode, history/compaction | Supplied configuration replaces the corresponding earlier configuration. |
| Artifacts | Overlay the explicitly supplied policy fields. |
| Omitted members | Keep their earlier values. |

These are ordinary profile merge rules. Capability-owned resources are stricter:
conflicting concrete bindings or approval policies fail instead of transferring
ownership to a later same-named tool. An explicit instruction replacement also
clears preceding capability guidance, without removing its tools or policies.

An empty tools list is not a request to revoke tools inherited through profile
composition. For a separate restricted agent, define a child with its own tool
surface and use [delegation](doc-agentic-engineering-subagents.md). See [Hooks](doc-agentic-engineering-hooks.md) for inherited
handlers that child code cannot disable.

Direct directives after profile application can override profile members:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> use: reviewer
> effort: high
```

The result retains the reviewer's configuration and changes effort to `high`.
The selected provider must support that effort value.

Profile application and direct directives follow source order. A direct directive
does not have unconditional priority: a profile applied after it can replace it.
Procedures capture the configuration at their definition site; see
[Scoping and Capabilities](doc-agentic-engineering-scoping-and-capabilities.md).

## Default Child Result Type

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
~Finding(path: str, explanation: str)

> profile: reviewer:
    ###
    Reviews supplied evidence and returns actionable findings.
    ###
    > adapter: pydantic
    > model: openai:gpt-5.6-luna
    > instructions: Base findings only on the evidence supplied in the task.
    > output: list[Finding]
```

When a parent delegates to `reviewer` without supplying `final_schema`, this
output type supplies the child's result schema. The returned envelope contains
`run_id`, `subagent`, `task_summary`, and the validated `final_result`.
An explicit `final_schema` on the child call takes precedence over `> output:`.
Without either schema, the child returns its text in `task_summary` and has no
structured result.

This is a child-result contract, not a way to change the types of unrelated
template captures. A template's `[finding: Finding]` still declares its own
output type. The profile type must be representable as JSON Schema; declaring
two `output` members is an error.

## Export and Import Profiles

Profiles are module values and can be explicitly exported:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> profile: reviewer:
    > adapter: pydantic
    > model: openai:gpt-5.6-luna

> export:
  reviewer
```

Imported profiles retain private tool and child-profile bindings needed by
their contract without flattening those dependencies into the importer's
procedure namespace. This lets a package expose a profile facade while keeping
helpers private.

## Forward References and Validation

A profile can name a child profile declared later. Kedi resolves the complete
profile graph and rejects unknown children and cycles.

Within one profile, framework and harness selection remain mutually exclusive.
Invalid backend kinds, unknown members, duplicate output/workflow declarations, invalid settings,
nonpositive `max_agents`, missing tools, and unsupported required capabilities
produce diagnostics or compile-time errors.

## Child Profiles

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> profile: coordinator:
    > adapter: pydantic
    > subagent: researcher
    > max_agents: 3
    > workflow: dynamic

> profile: researcher:
    > adapter: pydantic
    > instructions: Investigate one self-contained question.
```

Only directly listed children can be delegated to. `max_agents` bounds the
descendant work started by one parent invocation. Workflow mode defaults to
`delegate`; `dynamic` exposes one sandboxed Python orchestration tool. See
[Subagents](doc-agentic-engineering-subagents.md) for lifecycle and safety rules.

## Choosing Profile Boundaries

Create a profile when a task needs a stable combination of behavior and
capabilities. Do not create one merely to alias a model string. Profiles should
be narrow enough that their tools, approvals, and child relationships can be
reviewed as a coherent security boundary.
