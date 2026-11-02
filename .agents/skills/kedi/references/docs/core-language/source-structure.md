# Source Structure

## Statements and Source Order

Kedi preserves source order for declarations and executable statements. Imports
publish their exported bindings at the position of the import, and later writes
replace earlier bindings:

```kedi
[label] = draft
> import: labels
[label] = final

= <label>
```

The final same-scope initialization wins. The same rule applies when a
procedure, type, value, or import reuses a name. Prefer unique public names;
relying on replacement is mainly useful for deliberate overrides.

## Indentation and Lexical Blocks

Indentation opens procedure, profile, directive, test, eval, and Python-owned
blocks. Tabs count as width four when indentation is compared, but spaces are
recommended so editor rendering cannot alter perceived scope.

```kedi
@greet(name: str) -> str:
  [prefix] = Hello
  = <prefix>, <name>
```

The body must be indented relative to `@greet`. A dedent ends the block.

## Top-Level and Procedure Scope

Top-level values are visible to following statements and procedures compiled in
that environment. Procedure parameters and local initializations live in the
procedure frame. Inner scopes may shadow outer values without permanently
changing the caller's frame; `:=` explicitly assigns through that lexical
chain.

Agent configuration has capture semantics in addition to normal lexical scope:
top-level directives are captured by following procedure definitions, while
directives inside a procedure affect following model calls in that procedure.

## Blank Lines

Blank lines do not close a block. Dedentation determines block boundaries.
Within a multiline `>>` template, continuation rows are part of one prompt only
while they remain adjacent continuation syntax at the same Kedi indentation.

## Line Comments

`#` starts a line comment outside Python blocks:

```kedi
[limit: int] = `10`  # Runtime limit
```

Escape a literal hash in Kedi text as `\#`:

```kedi
= Ticket \#<`42`>
```

Python blocks use Python's own comment rules because their contents are parsed
as Python rather than Kedi.

## Block Comments

Matching lines containing only `###` open and close a block comment:

```kedi
###
This workflow is intentionally deterministic after classification.
The block can span multiple lines.
###
```

The delimiters may be indented with their surrounding Kedi block. They must be
paired, must occupy their own lines, and `###` cannot appear inside the comment
body as ordinary text.

## Procedure Docstrings

When the first statement in a procedure body is a block comment, that comment
becomes the procedure's docstring:

```kedi
@slugify(value: str) -> str:
  ###
  Convert a display label into a stable lowercase slug.
  ###
  = `value.strip().lower().replace(" ", "-")`
```

Procedure docstrings feed Python introspection, editor hovers, virtual stubs,
and tool descriptions when the procedure is exposed as an agent tool. A block
comment after another body statement remains a normal comment.

## Profile Docstrings

Profiles also support a docstring when the first member is a block comment:

```kedi
> profile: reviewer:
    ###
    Review a change against repository evidence.
    ###
    > instructions: Cite evidence for each finding.
```

This documents the profile; it is not a system instruction. A later block
comment remains an ordinary comment. See [Profile Documentation](../agentic-engineering/profiles.md#profile-documentation)
for its use in editor hovers and subagent descriptions.

## Reserved Names

`args` is the default runtime-owned command-line argument binding and cannot be
assigned or removed. Embedders can change the identifier through the compiler
environment before compilation, but ordinary programs should treat it as
reserved.

Directive names and output identifiers are also validated. Unknown `>`
directives fail instead of being treated as prompt text.
