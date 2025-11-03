# Kedi 
## A typed reasoning layer for LLM orchestration

Kedi is a compact DSL that lets you compose multi‑step LLM workflows with strong typing, Python interop, and clear dataflow. You write indentation‑scoped procedures, mix natural language prompts with typed outputs, and weave in Python for logic—all compiled into a runtime you can execute locally or against your preferred model.

### A Quick Taste

````markdown
~City(name, history, economy, cuisine)

@get_city_info(city) -> City:
    Paragraphs [history], [economy], and [cuisine] describe <city> in different aspects.
    = `City(city, history, economy, cuisine)`

@get_cities_in(country) -> list[str]:
    All cities in <country> are [cities: list[str]].
    = `cities`

# AI-generated procedure !
@consolidate_city_info_as_markdown(cities: list[City], country) -> str:
    > This is a procedure that takes a list of City objects of cities in \
    the country (given in the country parameter) and \
    with the information about their history, economy and cuisine. \
    The procedure will consolidate the information into a markdown \
    string and return it for an informative report.  
    
[country] = Singapore
[cities_info : list[City]] = `[get_city_info(city) for city in get_cities_in(country)]`
[markdown] = `consolidate_city_info_as_markdown(cities_info, country)`

= <markdown>
````

### Run it
- Local install:
`pip install -e .`
- After install:
`kedi <source> [OPTIONS]`
- Execute a program: 
`kedi your_program.kedi`
- Override model (optional): 
`kedi your_program.kedi --adapter-model groq:qwen/qwen3-32b`
- Run tests:
`kedi your_program.kedi --test`
- Run evaluations:
`kedi your_program.kedi --eval`
- Run with optimizer:
`kedi your_program.kedi --eval --optimize --optimizer gepa`

The runner loads the source, compiles it, optionally applies model/adapter settings, and executes the top‑level program via `run_main()`.

### Core syntax in 30 seconds
- **Procedures**: `@name(param[: type], ...) -> ret_type:` then an indented block.
  - Types: `str`, `int`, `float`, `bool`, `list[T]`. Omitted types default to `str`.
- **Template lines and outputs**: prompt the model and capture typed fields.
  - Example: `Top cities are [cities: list[str]]`
- **Substitutions**: use values and calls inline.
  - `<var>`, `<call(arg1, arg2)>`, nested: `<outer(<inner(x)>)>`
  - Inline Python: `<`len(xs)`>` renders as string in templates/returns.
- **Arguments and native types**: pass native values with single backticks per arg.
  - Example: `<add(`2`, `3`)>` where `add(x: int, y: int)`.
- **Multiline Python blocks**: fence with triple backticks on their own lines.
  - Assign value: 
    ````py
    [x: int] = ```
    return 40 + 2
    ```
    ````
  - Return native values from procedures by annotating the return type and using Python expressions/blocks: `@wrap() -> list[int]: = `[1,2,3]``
- **Variables and assignment**: `[name(: type)] = expression` then reuse anywhere.
- **Custom types**: `~TypeName(field[: type], ...)` creates a Pydantic model in scope.
- **Returns**: a line starting with `=` returns a value (string by default).
- **Line continuation**: end a line with `\` to continue on the next line.
- **Escaping specials**: use backslash before special chars. Escapable: `<`, `>`, `[`, `]`, `=`, `@`, `,`, `\\`, `#`, `~`, `` ` ``, `(`, `)`.
  - Examples: `\<`, `\>`, `\[`, `\]`, `\\`, `\,`, `\=`, `\@`, `\#`, `\~`, `` \` ``, `\(`, `\)`.

See `docs/KEDI_DSL.md` for the full grammar, typing rules, multiline Python prelude, closures for nested procedures, and structured output behavior.


