# Skills

Skills are scoped instruction documents loaded on demand through two read-only
tools. Kedi does not inject every skill into every prompt.

## Layout

```text
.agents/
  skills/
    release-review/
      SKILL.md
    incident-triage/
      SKILL.md
```

Each skill is exactly one UTF-8 `SKILL.md` under one skill-name directory.
Skill names start with an alphanumeric character and may contain letters,
digits, `_`, and `-`.

For example, `.agents/skills/release-review/SKILL.md` can contain:

```markdown
# Release Review

Compare the supplied change list against the supplied test results.
Report observed failures separately from tests that were not run.
Do not claim a deployment occurred without a tool result proving it.
```

The document provides guidance. It is not executable Kedi, does not create a
profile, and does not grant permission to read repositories or deploy releases.

## Enable Skills

```kedi
> skills: enabled

>> Use a relevant project skill if one applies. The result is [answer: str].
= <answer>
```

The compact directive registers `list_skills` and `read_skill`. Disable an
inherited policy with `> skills: disabled`.

Profiles can carry the same setting:

```kedi
> profile: maintainer:
    > adapter: pydantic
    > skills: enabled
```

The expanded form configures discovery:

```kedi
> skills:
    enabled: true
    cwd: workspace
    max_skills: 40
    include_registry: true
    include_all: false
    exclude_paths: `["~/.agents/skills"]`
```

`enabled` is required in the expanded form. The remaining settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `cwd` | program directory | Base directory used to locate the project-local `.agents/skills` source |
| `max_skills` | `20` | Maximum number of discoverable skills; accepts 1 through 100 |
| `include_registry` | `true` | Include `$KEDI_HOME/registry/skills` in discovery |
| `include_all` | `false` | Merge all source roots instead of stopping at the first root containing valid skills |
| `exclude_paths` | empty | Exclude source roots or individual skill directories after path expansion |

Relative `cwd` values resolve from the Kedi program, not the shell's incidental
working directory. `exclude_paths` is an inline Python list; `~` is expanded
before source comparison.

The policy is lexical. A top-level directive affects following calls and
procedures defined after it. A directive inside a procedure or profile applies
only in that scope. `> skills: disabled` explicitly turns off an inherited
policy.

## Source Priority

Kedi checks sources in this order:

1. `$KEDI_HOME/registry/skills`, normally `~/.kedi/registry/skills`;
2. `<cwd>/.agents/skills`;
3. `~/.agents/skills`.

Merged names are deterministic. If sources define the same skill name,
`read_skill` uses the highest-priority copy. Set `include_registry: false` to
skip only the registry source.

## Discovery Tools

`list_skills(all: bool = false, limit: int = 20) -> list[str]` returns valid
skill identifiers in deterministic sorted order. `limit` must be a positive
integer no greater than 100 and is additionally bounded by `max_skills`.
Source merging is configured by `include_all`; the `all` tool argument remains
accepted for API compatibility.

`read_skill(skill_name: str) -> str` returns the exact UTF-8 file content for
one valid listed skill.

Both tools are `read_only`, so the default approval policy allows them.

## On-Demand Loading

Enabling skills adds a sorted, bounded inventory of available names, not their
contents, and compact instructions telling the agent to:

1. use the inventory to identify a relevant skill; call `list_skills` only to refresh discovery;
2. call `read_skill` before answering a project-specific task that skill covers;
3. follow it only for the current task;
4. never claim to have read a skill whose contents were not returned.

The model does not know a skill's body until it reads that skill. This avoids
filling every context with unrelated instructions and makes skill usage visible
in the tool trace. Unrelated questions do not require a discovery or read call.

For tools and skills, Kedi's shared grounding instruction gives task-specific
evidence precedence over a familiar answer from model memory. It does not demand
tools for stable general knowledge or self-contained calculations. This is a
behavioral instruction, not a proof that the model used evidence correctly;
validate important decisions and retain the tool trace.

## Install Into the Kedi Registry

Install a one-file skill from a local directory or GitHub repository:

```bash
kedi skills add --path ~/my-skill
kedi skills add --repo owner/skill-repository
```

The source must have `SKILL.md` at its root. Kedi copies only that file into
`~/.kedi/registry/skills/<name>/SKILL.md`. GitHub input uses the
`OWNER/REPOSITORY` form, performs a credential-free shallow checkout, and
records the checked-out revision. Installation is user-scoped and never writes
into the project.

`--path` accepts either a skill directory containing `SKILL.md` or a path to
that file. Reinstalling the same skill name replaces its registry copy only
after the new source has passed validation.

## Resolution and Security Checks

Kedi rejects invalid names, absolute/traversal attempts, symlink escapes,
missing/non-file targets, non-UTF-8 content, and files larger than 256 KiB.

`list_skills` omits entries that fail validation rather than advertising
unreadable content. `read_skill` reports the specific failure when explicitly
requested.

These checks protect skill-file resolution. Skill content is instruction text
and may attempt to influence agent behavior; file validation does not establish
that its instructions are trustworthy. Review project
and installed skills, and keep approvals/tool boundaries active.

## When to Use Skills

Use a skill for repeatable operational guidance that should be selected at task
time: release procedures, code-review policy, migration checklists, or domain
workflows. Put unconditional behavior in `> instructions:` and executable typed
operations in tools instead.

Skills do not grant capabilities. A skill may instruct use of a tool, but the
profile must separately expose that tool and its approval policy.
