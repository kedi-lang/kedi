# Directive Index

## Agent Configuration

| Directive | Scope and contract |
| --- | --- |
| `> adapter: pydantic` | Select framework adapter: `pydantic`, `dspy`, or `langchain` |
| `> agent: codex` | Select harness: `claude`, `codex`, `acp`, or `a2a` |
| `> agent: acp:` | Select ACP and bind its explicit stdio `command` |
| `> agent: a2a:` | Select a remote A2A endpoint, auth reference, and transport timeouts |
| `> model: value` | Set plain or Python-evaluated model identifier |
| `> requires: capability` | Require one adapter capability before model I/O |
| `> requires:` | Require an indented list of adapter capabilities |
| `> effort: level` | Set `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` |
| `> instructions: text` | Replace active instructions; block form joins lines |
| `> settings:` | Merge adapter settings by key |
| `> history: enabled\|disabled` | Enable or disable scoped conversation continuity |
| `> history:` | Configure history ownership and native compaction settings |
| `> artifacts:` | Configure scoped large-value storage and compact references |
| `> codemode: enabled\|disabled` | Enable or disable scoped tool discovery and sandboxed tool execution |
| `> codemode:` | Configure CodeMode discovery, execution and resource limits |
| `> approval: allow`, `deny`, or handler | Set lexical tool approval policy |
| `> hooks: enabled\|disabled` | Enable or disable inherited lexical lifecycle handlers |
| `> hooks:` | Register handlers for named agent lifecycle events |
| `> mcp:` | Append one MCP server specification |
| `> use: name` | Register a tool or apply a profile |
| `> skills: enabled\|disabled` | Configure scoped skill discovery |
| `> use:` | Register an indented list of procedure/Python tools |

Framework and harness selection are mutually exclusive in one state. Literal
names are statically validated; backtick expressions defer validation to
runtime. Adapter settings unsupported by the selected backend are filtered or
rejected according to that adapter's contract.

A directive block has one body shape. Configuration directives contain
unprefixed `name: value` subsettings; composite directives such as `> profile:`
contain `>`-prefixed subdirectives. Kedi does not mix both forms in one body.

`> system:` has been removed and produces a parse error. Use `> instructions:`
instead; its body, substitutions, lexical scope, and precedence stay the same.
The Python `system=` configuration parameter is unchanged.

`> artifacts:` fields are `enabled`, `query_artifacts`, `store`, `path`, `threshold`, `ttl`,
`idle_ttl`, `preview_chars`, `read_max_chars`, `session_quota`,
`max_artifacts`, and `cleanup_interval`. The policy is lexical and enabled by
default. Artifact querying is separately disabled by default and accepts
`query_artifacts: enabled|disabled`. See [Tool Artifacts](../runtime/tool-artifacts.md).

The expanded `> history:` form contains only subsettings. It requires
`enabled: true|false` and accepts `compaction_mode: native|disabled` plus an
optional positive `compaction_threshold`. Pydantic and LangChain also accept
`processor` as a sync/async callable reference in inline Python. Omission inherits;
an explicit value must evaluate to a callable. Boolean `enabled` fields may
instead use a backtick expression returning `bool`. The short activation form
also accepts a Boolean expression. Artifact `query_artifacts` accepts
`enabled|disabled` or a backtick expression returning `bool`.
`processor_condition` accepts a sync/async predicate returning exactly `bool`.
It receives read-only history metadata before each logical request; without a
condition, processing is unconditional. A condition alone can inherit a processor;
a new processor does not inherit an old condition. See
[History Processing](../runtime/history.md#native-configuration) for group-safe
retention and provenance. Compaction is history lifecycle
policy, not a model setting. See
[Caching and Conversation History](../runtime/caching.md).

`> codemode:` accepts `enabled`, `preload_tools`, `default_search_limit`,
`max_search_limit`, `max_hydrated_tools`, `max_discovery_result_bytes`,
`max_code_chars`, `max_nested_calls`, `max_concurrent_calls`,
`max_tool_result_bytes`, `max_total_tool_result_bytes`, `max_print_bytes`, and
`request_timeout`. `preload_tools` may be one exact exposed name or an indented
list; preloading resolves schemas without executing tools. See
[CodeMode](../agentic-engineering/codemode.md#preload-known-tools).

`> hooks:` fields are `user_prompt_submit`, `pre_tool_use`, `post_tool_use`,
and `post_tool_use_failure`. Each value is a Python callable or a sequence of
callables. See [Agent Lifecycle Hooks](../agentic-engineering/hooks.md).

`> mcp:` fields:

```kedi
> mcp:
    name: project_docs
    transport: stdio
    command: uv
    args: `["run", "docs-server"]`
    env: `{"LOG_LEVEL": "warning"}`
    cwd: .
```

Network transports use `url` instead of `command`/`args`. Valid normalized
transports are documented in [MCP Servers](../agentic-engineering/mcp.md).

## Procedure Tool Metadata

| Directive | Scope and contract |
| --- | --- |
| `> tool:` | One declaration inside a procedure, after an optional leading docstring and before executable statements |

The fields are `name`, `description`, `risk`, `retries`, and `retry_on`.
`retry_on` is an indented list of visible `Exception` class names. Metadata
changes the exposed tool contract, not the procedure's source identity. Retries
apply only to body failures; validation, hooks, approval, cancellation, and
return validation are not retried. See
[Tools and `> use:`](../agentic-engineering/tools-and-use.md#native-tool-metadata).

## Profiles and Delegation

| Directive | Contract |
| --- | --- |
| `> profile: name:` | Define a reusable agent state |
| `> subagent: child` | Permit one direct child profile |
| `> max_agents: N` | Bound descendant starts for one invocation |
| `> workflow: delegate\|blocking\|dynamic` | Select full delegation, synchronous-only delegation, or sandboxed dynamic child orchestration |
| `> workflow:` block with `> mode:` and `> tools:` | Set the mode and, in `delegate` mode, which subagent control tools the model sees |
| `> output: Type` | Declare a profile's default structured child result; an explicit child-call `final_schema` overrides it |

A profile body may contain adapter or agent selection, model, effort, instructions,
settings, explicit requirements, approval, hooks, history, skills, CodeMode,
artifact policy, MCP, tools, child profiles, descendant budget, workflow mode,
and output type. See
[Profiles and Composition](../agentic-engineering/profiles.md#merge-rules) for
the member-specific merge rules. Defining or applying a profile is configuration,
not a model invocation.

`> subagent:`, `> max_agents:`, and `> workflow:` are profile members rather
than arbitrary runtime spawn commands. `delegate` is the default. `dynamic`
requires at least one direct child and exposes `run_workflow` instead of the
delegation lifecycle tools. Forward child references are valid; unknown
children, cycles, nonpositive budgets, duplicate workflow modes, and
unsupported adapter capabilities fail.

## Modules and Packages

| Directive | Contract |
| --- | --- |
| `> import: module/path` | Import all explicitly exported names |
| `> import: module/path:` | Import only listed exported names |
| `> export:` | Export listed top-level procedures, types, values, or profiles |
| `> export: *` | Export public names, excluding `_`-prefixed names and runtime-owned `args` |
| `> package: name:` | Declare metadata in `package.kedi` only |

Imports are relative to their source file before bundled and installed-package
fallbacks. Selective imports bind names directly, not under a namespace.
Modules without an export directive expose nothing.

`> package:` supports `author`, `contact`, `version`, `source`, `python`,
`license`, and one `python_dependencies:` list. The manifest must contain no
executable statements.

## Tests and Evaluation

| Directive | Parent | Contract |
| --- | --- | --- |
| `> case: name:` | `@test:` | One Python assertion block |
| `> data: name:` | `@eval:` | Training/fallback-eval iterable |
| `> test_data: name:` | `@eval:` | Matching held-out iterable |
| `> metric: name(dataset):` | `@eval:` | One score function for the suite |

Cases run in source order and failures do not stop later cases. Eval suites
allow one metric. `--eval` prefers same-named test data, while optimization
always trains on data and validates with matching test data.

## Optimization and Generation

```kedi
@classify(text: str) -> str:
  > optimize: classification_prompt:
    >> The text <text> belongs to the [label] category.
  = `label`

@generated_slug(value: str) -> str:
  > auto:
    Implement deterministic slug generation for <value>.
```

`> optimize: span_name:` owns one model template whose optimized artifact key is
the containing procedure plus span name. Every optimized procedure needs a
same-named eval suite with training data and a metric.

`> auto:` requests generated procedure implementation and tests. Codegen
uses a separate agent/model configuration and stores generated source in
`program.cache.kedi`.

An optimize body must start its template with `>>`. Its continuation lines form
one prompt, not separate calls. A bare optimize prompt is a parse error.
The `> auto:` body remains an indented code-generation specification.

## Placement and Capture

Agent directives at top level are captured by procedures defined afterward.
Directives in a procedure affect subsequent calls in that invocation and are
restored on exit. Profile application follows the same lexical rule.

Unknown directives, invalid placement, duplicate singleton members, malformed
blocks, or wrong directive kinds are errors rather than ignored configuration.
