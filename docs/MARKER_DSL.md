## Marker DSL: Language Grammar and Semantics

### Overview
Marker is a lightweight DSL to orchestrate LLM interactions using indentation-scoped blocks, inline substitutions, outputs, and returns. It compiles to a `MarkerRuntime` that executes prompts and threads values across steps.

### Building Blocks
- Procedures: `@name(param1, param2, ...):` followed by an indented block.
- Template lines: free text with substitutions and outputs.
- Substitutions (R-values): `<var>` or `<call(args...)>`.
- Outputs (L-values): `[name]` within a template line; signals the LLM must produce this value.
- Return lines: `= expression` returns from the current procedure or top-level.

### Indentation and Preprocessing
- Indentation defines block scope; the preprocessor inserts virtual BEGIN/END markers on indentation changes for parsing.
- Tabs count as width 4 for indentation comparison.

### Substitutions and Calls
- `<name>` reads a variable by name.
- `<name(args...)>` calls a procedure. Calls may be nested: `<outer(<inner(...)>)>`.
- Inside call arg lists:
  - Bare tokens are string literals (no quotes). Example: `<combine(x, <prep(y)>)>` → `combine("x", prep("y"))`.
  - Variables or nested calls must be wrapped in `<...>`, e.g. `<combine(<x>, <prep(<y>)>)>` → `combine(x, prep(y))`.
  - Multiple literal args separated by commas.
  - Commas inside a single literal must be escaped by doubling: `,,`.

### Outputs
- `[name]` is an output placeholder that must be filled by the LLM on that line.
- Only identifiers are allowed in `[]`: `^[A-Za-z_][A-Za-z0-9_]*$`.
- Procedure calls are not allowed in `[]`.
- Multiple outputs per line are allowed; all will be requested and made available in scope.

### Returns
- A line starting with `=` returns the value of the expression that follows.
- The expression can interleave text with substitutions and nested calls.
- Whitespace handling: trim only at the ends; do not collapse internal spaces. That is, spaces immediately after `=` and before end-of-line are trimmed; the rest are preserved.

### Escaping Special Characters
Double the character to treat it as a literal:
- `<<` → `<`, `>>` → `>`, `[[` → `[`, `]]` → `]`, `==` → `=`, `@@` → `@`, `,,` → `,`.

### Examples
Procedure and call:
```
@get_country(city):
  <city> is located in [country].
  = <country>

= <get_country(Paris)>
```

Literal vs variable args:
```
@make_thing(x, y):
  a fine [out] object from <combine(x, <prep(y)>)>
  = <out>
```
Semantics: `combine("x", prep("y"))`. To pass variables instead, write `<combine(<x>, <prep(<y>)>)>`.

Escaped commas and nesting:
```
= <foo(alpha,, beta,, gamma, <bar(<x>)>)>    done   
```
Semantics: `{foo("alpha, beta, gamma", bar(x))}    done` (ends trimmed only).

Interleaving without internal space collapse:
```
=    Leading     <x>    and    <y>     combined    
```
Result: `Leading     {x}    and    {y}     combined`.

### Error Conditions
Implementations should raise descriptive errors for:
- Unmatched `<` ... `>` or `[` ... `]` delimiters.
- Output name not a valid identifier (e.g., `[42x]`, `[]`, `[foo(bar)]`).
- Malformed procedure header (missing `(`, missing `)`, missing trailing `:`).
- Unterminated call arg list.
- Using an output expression inside a call (not supported).

### Notes
- Top-level may contain template lines and a final return that acts as `main`.
- Outputs defined by template lines become variables for subsequent lines in the same scope.


