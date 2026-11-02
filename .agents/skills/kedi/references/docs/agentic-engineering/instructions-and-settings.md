# Instructions and Settings

Use `> instructions:` for scoped agent instructions. The old spelling
`> system:` is no longer accepted and produces a parse error. Replace the
name without changing the body; lexical scope and override rules stay the
same. Python's `system=` configuration parameter is unchanged.

Instructions define agent behavior; settings define adapter/model mechanics.
Both are lexical and apply to following calls.

Instructions express desired behavior; they are not permissions or enforced
invariants. Use typed captures for output validation, tool schemas for argument
contracts, and [approvals](approvals.md) for effectful operations. Asking a model
to "never write outside reports" does not replace a path check.

## Single-Line Instructions

```kedi
> instructions: Answer with evidence from available tools.
```

Use a single line for one stable rule. It is rendered when active state is
materialized.

## Multiline Instructions

```kedi
[audience] = maintainers

> instructions:
    Act as a release engineer for <audience>.
    Inspect evidence before reaching a conclusion.
    Report uncertainty explicitly.
```

Continuation lines are newline-joined. Multiline instruction bodies are read-only
templates: literal text, `<name>` substitutions, and inline Python
substitutions are allowed. Output fields and procedure calls are not.

A later `> instructions:` replaces the previous instruction value; it does not append
another paragraph. Compute one combined instruction explicitly when both parts
must survive. [Profile composition](profiles.md#merge-rules) uses the same rule.

For named reusable instruction contributions, use
[capabilities](capabilities.md). They append without changing profile override
semantics. A later explicit `> instructions:` clears those contributions as well,
including when the replacement text is identical. Runtime-owned instruction
sections are not removed.

```kedi
> instructions:
    Current audience: <audience>.
```

An instruction cannot declare `[output]` or call `<helper()>`; compute that
value before the directive and substitute the resulting variable. Use `<``>`
when the instruction itself must mention a literal triple-backtick fence.

## Runtime Instruction Sections

Kedi keeps scoped user instructions separate from its runtime rules. For an LLM
template with output captures, the runtime adds a `<TemplateInstructions>` section
to the request's instructions. The user message contains the field list and the
rendered template, without repeating the completion rules in its prefix or in
conversation history.

Profiles with tools, MCP, or enabled skills also receive a separate
`<ToolCallInstructions>` section. It covers evidence, explicit tool requirements,
permissions, and untrusted retrieved content. It does not prescribe a tool-use
strategy or require a lookup for every answer. The two sections are siblings,
not nested, and neither replaces the active `> instructions:` value.

When enabled, artifact and skill guidance use sibling `<ArtifactInstructions>`
and `<SkillInstructions>` sections. Built-in instruction XML tags, including
their internal headings, consistently use PascalCase. These headings do not
rename tool parameters, output fields, or serialized protocol data.

Raw invokes and claims do not receive template-completion rules. Jev and Laya
keep their decision-specific prompt semantics rather than sentence-completion
instructions. Runtime sections are assembled for the current call, not appended
to the stored user conversation on each turn.

These section names are internal prompt formatting, not Kedi syntax or security
boundaries. Typed validation and approval policies remain responsible for enforcing
contracts; provider caching still depends on the model and transport.

## Settings Blocks

```kedi
> settings:
    temperature: 0.2
    max_tokens: 1024
    parallel_tool_calls: false
```

Plain values parse as booleans, `null`/`none`, integers, floats, Python literals
when valid, or otherwise strings.

Use backticks for explicit Python objects:

```kedi
> settings:
    stop_sequences: `["END", "DONE"]`
    extra_body: `{"mode": "json"}`
    parallel_tool_calls: `False`
```

Backticks are necessary for computed values and remove ambiguity about list,
dict, or object identity.

## Merge and Filtering

Settings merge by key. A later value replaces the same earlier key while
unmentioned keys remain:

```kedi
> settings:
    temperature: 0.2
    timeout: 30

@patient_call() -> str:
  > settings:
      timeout: 120
  >> The completed task's result is [answer: str].
  = <answer>
```

The inner call retains `temperature: 0.2` and uses `timeout: 120`.

Kedi validates setting names against its known public setting inventory, then
filters the merged mapping at the adapter boundary. Pydantic receives supported
`ModelSettings`; DSPy receives supported `dspy.LM` kwargs; LangChain receives
its chat-model fields; harnesses receive their own process/runtime settings.

## Common Framework Settings

Common fields include `temperature`, `max_tokens`, `timeout`, `top_p`,
`parallel_tool_calls`, `tool_choice`, retry controls, stop sequences, and
provider extension bodies. Exact support still depends on the active adapter
and model provider.

Do not copy a provider-specific setting into every profile. Keep it in the
profile that owns that backend.

## Harness Settings

ACP accepts `cwd`, `env`, and `timeout`. Codex accepts fields including `cwd`,
`sandbox`, `approval_policy`, `config`, `model_provider`, `service_tier`, and
`timeout`. Claude accepts fields including `cwd`, `env`, `tools`,
`allowed_tools`, `disallowed_tools`, `permission_mode`, `max_turns`, and
`max_budget_usd`.

```kedi
> agent: acp:
    command: npx @zed-industries/codex-acp
> settings:
    cwd: /workspace/project
    env: `{"LOG_LEVEL": "INFO"}`
    timeout: 120
```

`cwd` is passed to ACP, Codex, and Claude where supported. Relative child-agent
working directories are constrained by their parent safety boundary.

Claude appends `> instructions:` content to its Claude Code preset, preserving
built-in file behavior. Non-interactive Claude defaults to
`permission_mode: acceptEdits`; set tool and permission fields explicitly to
narrow it. Codex defaults to `workspace-write` unless settings narrow or replace
its sandbox.

## Unknown and Inapplicable Settings

Unknown setting names are parser/LSP errors. A known union-level key can still
be irrelevant to a selected backend and is filtered rather than passed
blindly. Adapter docs list the exact mapping.

Prefer a minimal settings block. Every additional provider knob is part of the
profile's operational contract and should have a tested reason.

Conversation history and compaction are intentionally not model settings. Use
`> history: enabled` for stateful calls or the expanded history policy described
in [Caching and Conversation History](../runtime/caching.md).
