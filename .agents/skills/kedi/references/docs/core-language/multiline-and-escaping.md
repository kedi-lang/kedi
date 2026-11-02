# Multiline Syntax and Escaping

Kedi has separate multiline rules for model prompts, rendered returns, Python,
comments, and directives. Do not transfer one construct's continuation syntax
to another.

## Multiline Templates

Same-indentation lines following `>>` form one model request:

```kedi
>> The deployment risk is [risk: Literal["low", "medium", "high"]].
The reason for that decision is [reason: str].
= <risk>: <reason>
```

The two prompt lines are newline-joined and share one structured schema.
Another `>>` starts another model call:

```kedi
>> The affected service is [service: str].
>> The recommended owner for <service> is [owner: str].
```

Use one block when outputs belong to one judgement. Use separate blocks when a
later prompt depends on an earlier output or represents an independent call.

A plain `>>` block without output fields still runs the model but discards the
response. Use `[answer] << ...` to keep unstructured response text.

## Multiline Instructions and Directives

Block directives such as `> instructions:` use indented continuation text:

```kedi
> instructions:
    Act as a release engineer.
    Prefer evidence from tools over assumptions.
    Keep the final answer concise.
```

Directive-specific substitution rules are documented with each directive.
System instruction text can include variable and inline Python substitutions;
output fields and procedure calls are not declarations there.

## Multiline Python

Fenced Python opening and closing markers must be alone on their lines and align
with the surrounding Kedi scope. The Python source aligns with those fences:

````kedi
@summarize(values: list[int]) -> dict[str, float]:
  = ```
  total = sum(values)
  return {
      "total": total,
      "average": total / len(values),
  }
  ```
````

Kedi dedents the block relative to its Kedi indentation before Python executes.
Do not indent the Python an extra level merely because it is inside a fence.

Use fenced blocks for statements. A single backtick line is a side-effect
statement; backticks inside an initialization, assignment, or return are expressions.

## Return Continuations

A rendered return uses a trailing backslash:

```kedi
@message(name: str) -> str:
  = Hello <name>, \
    the deployment completed.
```

This joins physical lines into one return expression. It is not model-template
continuation. Use `\\` when the output must contain a literal backslash.

## Escapes

Backslash escapes syntax characters in text:

| Escape | Result |
| --- | --- |
| `\<` / `\>` | Literal `<` / `>` |
| `\[` / `\]` | Literal `[` / `]` |
| `\=` | Literal `=` |
| `\@` | Literal `@` |
| `\,` | Literal comma, especially inside a call argument |
| `\\` | Literal backslash |
| `\#` | Literal `#` rather than a comment |
| `\~` or `~~` | Literal `~` |
| ``\` `` | Literal backtick |
| `\(` / `\)` | Literal parentheses |
| `\t` / `\n` / `\s` | Tab, newline, or one preserved space |

```kedi
@literal() -> str:
  = Use \<tag\>, \[field\], and contact\@example.com.
```

The same delimiter escapes work inside substitutions, output expressions, and
call-argument text where applicable. A backslash before an unsupported
character is an error rather than an implicit literal.

## Comments and Literal Hashes

An unescaped `#` begins a line or inline comment outside Python:

```kedi
[mode] = strict  # Used by the reviewer
= mode\=<mode>
```

Use `\#` for visible hash text:

```kedi
= Issue \#42
```

Inside Python fences, Python's own comment rules apply.

## Block Comments and Docstrings

Triple hashes delimit multiline comments:

```kedi
###
This block is ignored by execution.
Kedi syntax inside it is not parsed as program statements.
###
```

When this is the first statement inside a procedure or profile, its contents
become that declaration's docstring. In any later position it is only a comment.

## Code Fences in Markdown and Generated Text

Kedi's Python fence is exactly three backticks on a line. When documenting it in
Markdown, wrap the example in four backticks as this reference does. Within a
Kedi string, escape a literal backtick with ``\` ``. Do not place language labels
such as `python` after a Kedi runtime fence.

## Whitespace Preservation

Kedi trims ordinary whitespace at text boundaries and preserves internal
spacing. Escaped whitespace from `\s`, `\t`, and `\n` remains significant even
at a boundary:

```kedi
= \tIndented\n
```

Indentation itself defines lexical scope, so tabs and inconsistent widths should
not be used for layout. Keep one project-wide indentation style; the examples
use two spaces for Kedi blocks and four spaces inside nested Python syntax after
dedenting.
