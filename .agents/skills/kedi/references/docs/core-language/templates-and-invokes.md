# Templates and Invokes

## Template Lines with `>>`

`>>` opens a model template at top level or inside a procedure:

```kedi
@extract_owner(issue: str) -> str:
  >> Issue: <issue>
  The [owner: str] team owns the next action.
  = <owner>
```

Literal text is sent as written after substitutions are rendered. Output fields
describe values the model must produce.

## Multiline Template Blocks

Continuation rows at the same indentation are newline-joined into one model
request:

```kedi
>> Incident: <incident>.
The affected service is [service: str].
The severity is [severity: Literal["low", "medium", "high"]].
```

Do not prefix every continuation row with `>>` unless separate model calls are
intended.

## One Model Call per Block

The previous example starts one logical adapter operation with two output
fields. Tools, validation retries, or transport recovery can require more than
one provider request inside that operation. Starting a second block starts a
dependency-aware second operation:

```kedi
>> Incident <incident> affects [service: str].
>> To restore <service>, follow [steps: list[str]].
```

The second call can substitute `service` because the first call completed and
captured it. In parallel mode Kedi derives dependencies from these reads and
writes; no special concurrency syntax is needed.

## Raw Model Invokes with `<<`

Use a capture target before `<<` when the entire response should be retained as
text:

```kedi
[explanation] << Explain <concept> for an experienced Python developer.
= <explanation>
```

The result is always a string. `[explanation: str]` is accepted but redundant;
any non-string capture type is invalid. A raw prompt cannot contain output
fields such as `[count: int]`.

## Structured and Text Results

Choose `>>` with output fields when:

- downstream code needs named or typed values;
- multiple fields must be captured from one call;
- schema descriptions improve model behavior;
- invalid output should trigger adapter validation or retry behavior.

Prefer `>>` with `[name: str]` even for one piece of prose when the result is a
declared part of program dataflow. Choose `[name] << ...` only when deliberately
retaining an opaque, provider-native text response without an output schema. Do
not parse a predictable structured response from raw text when a typed output
can express the contract directly.

## Procedure-Local Backend State

Templates use the active adapter or harness, model, effort, instructions,
settings, tools, MCP servers, skills, and approval policy at their source
location. Procedures capture applicable outer configuration when defined, and
inner directives can override following calls.

## Template Failure Behavior

A plain `>>` prompt with no output fields is still sent to the model, but its
response is discarded:

```kedi
>> Record a concise acknowledgement of <event>.
```

Use this only when discarding the response is intentional. To keep it, use a raw
capture. Structured calls can fail before contacting the model when types or
adapter schema capabilities are unsupported; provider, validation, retry, and
approval failures surface as execution errors rather than empty outputs.
