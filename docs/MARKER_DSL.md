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
- Inline Python expression: `<`code`>` executes a single-line Python expression and substitutes its string value.
  - Example: `<x> + <y> is <`int(x)+int(y)`>`.
  - Scope injection: All marker variables in scope and procedures are available as Python names and callables.
  - Only allowed inside input context `<...>`; not permitted in `[...]` outputs.
  - Full Python builtins are available; imports are permitted (e.g., `<`__import__('math').sqrt(16)`>`).

### Multiline Python Blocks
- Use fenced blocks with triple backticks on their own lines:
```
```
python code...
```
```
- The opening and closing fences must be the only content on their lines (no inline ```pythoncode```).
- The code inherits the indentation level of its block; indentation is normalized (dedented) before execution.
- Variables in scope and procedures are injected into the Python environment; changes to variables inside the block are reflected back.
- Executed with full Python builtins, same as inline expressions.
- Value-returning Python blocks (NEW):
  - Assignment with return value:
    ```
    [x] = ```
    return 2 + 3
    ```
    ```
    Optional type annotation coerces the returned value: `[x: int] = ``` ... ```.
  - Return with value:
    ```
    = ```
    return compute()
    ```
    ```
    The value of the last executed return in the block is converted to string for final output.
  - Semantics: the block is wrapped into a function and executed; the Python `return` value becomes the assigned value or the final return respectively. If no value is returned, it is treated as empty string for `=` return and `None` for assignment.
- Inside call arg lists:
  - Bare tokens are string literals (no quotes). Example: `<combine(x, <prep(y)>)>` → `combine("x", prep("y"))`.
  - Variables or nested calls must be wrapped in `<...>`, e.g. `<combine(<x>, <prep(<y>)>)>` → `combine(x, prep(y))`.
  - Multiple literal args separated by commas.
  - Commas inside a single literal must be escaped by doubling: `,,`.

### Outputs
- `[name]` is an output placeholder that must be filled by the LLM on that line.
- Typed outputs are supported via `[name: type]`. If omitted, type defaults to `str`.
  - Examples:
    - `Most visited country is [country: str]`
    - `Top 3 cities are [cities: list[str]]`
- Only identifiers are allowed for names in `[]`: `^[A-Za-z_][A-Za-z0-9_]*$`.
- Procedure calls are not allowed in `[]`.
- Multiple outputs per line are allowed; all will be requested and made available in scope.
- Implementation note: When any typed outputs are present in a prompt, a Python `TypedDict` is constructed for the model output type; otherwise a generic `StructuredDict` schema of strings is used. The prompt `description` is preserved in both cases.

### Returns
- A line starting with `=` returns the value of the expression that follows.
- The expression can interleave text with substitutions and nested calls.
- Whitespace handling: trim only at the ends; do not collapse internal spaces. That is, spaces immediately after `=` and before end-of-line are trimmed; the rest are preserved.

### Variable Initialization / Assignment
- Assign a variable in the current scope using:
  - `[name] = expression`
  - `[name: type] = expression`
  - `[name(: type)] = ``` ... ```  # value-returning Python block`
  - Backtick inline Python for native value: `[name(: type)] = `expr``
    - Must be a single Python expression (no other text). Example:
      - `[x] = `1+1`` sets `x` as int 2
      - `[x] = <`1+1`>` sets `x` to string "2"
      - `[x] = `1+1` and more` → treated as plain string (no native eval)
- Example: `[prev] = <word>` assigns the current `word` into `prev`.
- Typed examples:
  - `[count: int] = <`1 + 1`>`
  - `[items: list[str]] = <`["a", "b", "c"]`>`
- The right-hand side is a full expression like a return: free text, `<var>`, `<call(...)>`, and inline Python `<`code`>` allowed.
- The assigned variable is immediately usable in subsequent lines, LLM prompts, calls, and Python.
- Semantics: if the RHS is a pure inline Python expression, its native Python value is assigned (e.g., a real `int` or `list[str]`). Otherwise, the rendered string is assigned.

### Escaping Special Characters
Double the character to treat it as a literal:
- `<<` → `<`, `>>` → `>`, `[[` → `[`, `]]` → `]`, `==` → `=`, `@@` → `@`, `,,` → `,`.
Backticks are only meaningful inside `<...>` for inline Python; no escaping is needed otherwise.

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

### Type Annotations
Basic typed output in a template line:
```
Most visited country is [country: str]
```

Collect a list of strings from the LLM and use it:
```
Top cities: [cities: list[str]]
= Count: <`len(cities)`>
```

Typed assignment capturing native Python values via inline Python:
```
[nums: list[int]] = <`[1, 2, 3, 4]`>
[total: int] = <`sum(nums)`>
= Total: <total>
```

Value-returning Python blocks:
```
[x: int] = ```
return 5
```
= ```
return x * 2
```
```

Mix typed outputs and inline Python:
```
<country> neighbors are [neigh: list[str]]
= First: <`neigh[0] if neigh else ""`>
```

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


