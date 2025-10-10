## Marker DSL: Language Grammar and Semantics

### Overview
Marker is a lightweight DSL to orchestrate LLM interactions using indentation-scoped blocks, inline substitutions, outputs, and returns. It compiles to a `MarkerRuntime` that executes prompts and threads values across steps.

### Building Blocks
- Procedures: `@name(param1, param2, ...):` followed by an indented block.
- Template lines: free text with substitutions and outputs.
- Substitutions (R-values): `<var>` or `<call(args...)>`.
- Outputs (L-values): `[name]` within a template line; signals the LLM must produce this value.
- Return lines: `= expression` returns from the current procedure or top-level.
- Comments:
  - Inline: everything after a single `#` on a line is ignored. Use `##` to escape a literal `#`.
  - Block: lines containing only `###` (after trimming whitespace) start and end block comments. `###` is not allowed anywhere else and must appear in matching pairs. Block comment delimiters may not appear inside python fences.

### Indentation and Preprocessing
- Indentation defines block scope; the preprocessor inserts virtual BEGIN/END markers on indentation changes for parsing.
- Tabs count as width 4 for indentation comparison.

### Substitutions and Calls
- Multiline marker strings:
  - Any template or return line can be continued by ending the line with a single unescaped backslash `\`. Whitespace between the backslash and newline is ignored, and the following line's content is concatenated as if it were the same line.
  - To render a literal backslash, escape it as `\\`.
  - Example:
    ````
    = Start \
      middle \
      end
    ````
    renders as `Start middle end`.
- `<name>` reads a variable by name.
- `<name(args...)>` calls a procedure. Calls may be nested: `<outer(<inner(...)>)>`.
- Inline Python expression: `<`code`>` executes a single-line Python expression.
  - In template/return expressions, its value is converted to string and substituted into the output: `<x> + <y> is <`int(x)+int(y)`>`.
  - In call argument lists, a single backtick expression used as the entire argument (see below) passes the native value as the argument.
  - Scope injection: All marker variables in scope and procedures are available as Python names and callables.
  - Only allowed inside input context `<...>`; not permitted in `[...]` outputs.
  - Full Python builtins are available; imports are permitted (e.g., `<`__import__('math').sqrt(16)`>`).

### Multiline Python Blocks
- Use fenced blocks with triple backticks on their own lines:
````
```
python code...
```
````
- The opening and closing fences must be the only content on their lines (no inline ```pythoncode```).
- The code must be indented to the same level as the surrounding Marker block. The code inherits the block's indentation and is normalized (dedented) relative to that level before execution.
- Correct vs incorrect indentation inside a procedure:
  ````
  @foo():
    = ```
    print("correct")
    ```

  @bar():
    = ```
  print("incorrect")
    ```
  ````
- Variables in scope and procedures are injected into the Python environment; changes to variables inside the block are reflected back.
- Executed with full Python builtins, same as inline expressions.
- **Prelude block (file-level python bootstrap):** If the very first meaningful content in the file is a multiline Python block, it is treated as a *prelude*.
  - The prelude executes before any Marker code and its resulting Python globals (imports, functions, variables, etc.) are injected into every inline python expression and multiline python block that follows.
  - Subsequent top-level python blocks execute normally and do not replace the prelude; only the first block gets this special treatment.
  - Prelude code runs exactly once when the program is compiled; failures raise `RuntimeError("Python prelude error: ...")`.
- **Lexical closures for nested procedures:** When you define a procedure inside another procedure, the inner procedure captures the outer procedure’s local environment (including variables assigned via `[name] = ...`, outputs, python blocks, and parameters).
  - Captured values are shared by reference; if the outer scope mutates a variable before or after the nested definition, the inner procedure observes the latest value.
  - Shadowing still applies: re-assigning the same name inside inner scopes hides the outer binding for that scope and deeper nested scopes.
- Value-returning Python blocks:
  - Assignment with return value:
    ````
    [x : int] = ```
    return 2 + 3
    ```
    ````
    If the returned value is non-string, an explicit compatible type annotation on `[name: type]` is required; string returns may optionally annotate `str` and must not annotate non-`str` types.
  - Return with value:
    ````
    = ```
    return compute()
    ```
    ````
    The value of the last executed return in the block is converted to string for final output.

### Procedure Call Arguments
- Each argument is parsed as one of two forms:
  1) Single backtick Python expression: `` `expr` ``
     - The Python expression is evaluated and the native value is passed to the callee.
     - Examples: `` `2` `` (int 2), `` `[1,2,3]` `` (list[int]), `` `int(x)+1` `` (int), `` `add(3, 4)` `` (uses a procedure callable in Python).
  2) Rendered mini‑template string: any other text segment (may include `<var>`, nested `<call(...)>`, inline Python `<`...`>`, and escaped commas `,,`). The result is rendered to a string and passed as a `str`.
     - Examples: `alpha`, `alpha <x> beta`, `<outer(<inner(y)>)>`, `count=<`len(items)`>`.

- Commas separate arguments; escape commas within a single argument by doubling: `,,`.
- Consequences:
  - To pass native typed values into typed parameters, use the single‑backtick form per argument.
  - Mixing text and substitutions in an argument creates a string argument.

Examples:
```
@foo(nums: list[int], label: str):
  = <`str(sum(nums))`>-<label>

= <foo(` [1,2,3] `, total:<`3`>)>     # passes list[int] and string "total:3"
```

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
      - `[x:int] = `1+1`` sets `x` as int 2
      - `[x] = <`1+1`>` sets `x` to string "2"
      - `[x] = `1+1` and more` → treated as plain string (no native eval)
    - Type rules:
      - If the evaluated value is non-string, an explicit compatible type annotation is required (e.g., `[x: int] = `1+1``).
      - String values must not be annotated with non-`str` types.
- Example: `[prev] = <word>` assigns the current `word` into `prev`.
- The right-hand side is a full expression like a return: free text, `<var>`, `<call(...)>`, and inline Python `<`code`>` allowed.
- The assigned variable is immediately usable in subsequent lines, LLM prompts, calls, and Python.
- Angle-inline pure `<`...`>` on RHS is always treated as string during assignment; annotating it as non-`str` is illegal.

### Escaping Special Characters
Double the character to treat it as a literal:
- `<<` → `<`, `>>` → `>`, `[[` → `[`, `]]` → `]`, `==` → `=`, `@@` → `@`, `,,` → `,`, `\\` → `\`.
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


### Type Annotations
#### Procedure parameter types
- Syntax: in `@name(params):`, each parameter may annotate its type: `param: type`. Unannotated parameters default to `str`.
- Supported types: `str`, `int`, `float`, `bool`, and `list[T]` (e.g., `list[int]`, `list[str]`). Invalid type names are rejected.
- Enforcement at call time (no implicit conversions):
  - The value supplied for each parameter must already be of the expected type.
  - Use the single‑backtick argument form to pass native values (e.g., `` `2` ``, `` `[1,2]` ``). Any argument that is not exactly a single backtick expression is rendered to a string and thus only satisfies `str` parameters.
  - Nested procedure calls used inside a backtick Python expression (e.g., `` `add(3, 4)` ``) pass the callee’s native return value if it has an annotated return type; otherwise the callee returns `str`.

Examples:
```
@add(x: int, y: int):
  = <`str(x + y)`>

= <add(`2`, `3`)>             # OK → "5"
= <add(2, 3)>                 # ERROR: unannotated literals are string args
```

```
@wrap(xs: list[int]) -> list[int]:
  = ` [v*2 for v in xs] `

= <`str(len(wrap([1,2,3])))`>  # OK: backtick arg supplies native list to wrap
```
#### Procedure return types
- Syntax: `@name(params) -> type:` to declare a non-string return type.
- If omitted, return type defaults to `str`.
- Returning non-string values requires an annotation and must use native Python returns:
```
@bar() -> list[str]:
  = `[]`
```
The above returns an empty list object.

```
@bar():
  = `[]`
```
This is invalid and will raise a TypeError because the default return type is `str`.
Basic typed output in a template line:
```
Most visited country is [country: str]
```

Collect a list of strings from the LLM and use it:
```
Top cities: [cities: list[str]]
= Count: <`len(cities)`>
```

Typed assignments:
````
[nums: list[int]] = `[1, 2, 3, 4]`
[total: int] = ```
return sum(nums)
```
= Total: <`str(total)`>
````

Value-returning Python blocks:
````
[x: int] = ```
return 5
```
= ```
return x * 2
```
````

Mix typed outputs and inline Python:
```
<country> neighbors are [neigh: list[str]]
= First: <`neigh[0] if neigh else ""`>
```


