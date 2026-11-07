# Evolve Groups and Forms

An `> evolve:` group names a region of a program whose body can change without
editing the source file. The body in the source is the group's default form. A
validated form can run in its place, supplied from a forms file, by an optimizer, or
proposed while the program is loaded. Every run records which form ran, so changes
can be compared over time.

## Mark a Group

```kedi
@solve(problem: str) -> str:
  > evolve: planner:
    >> Break <problem> into steps: [steps: list[str]]
    [plan] << Turn <steps> into a short numbered plan.

  = <plan>
```

If no form is supplied, the program runs as if the directive were absent and sends
the same model requests, byte for byte.

- A group does not open a scope. Names its body binds, like `plan`, stay visible
  after it.
- The body may contain templates, raw invokes, bindings, control flow, tasks, and
  Python. It cannot return, define procedures or types, contain directives, or
  contain another group, including `> optimize:`.
- A group's qualified name is `<procedure>.<name>`, or `<name>` at the top level. It
  must be unique in its module, and `> optimize:` span names count too.

## Identity

A group's identity does not depend on its source lines:
`kedi:group:<module>:<qualified name>`. `<module>` is the module's import name as the
root program sees it. That is the root file's name without `.kedi`, a path such as
`services/profiles` for a module below it, or a package name. An `> optimize:` span is
also a group, `kedi:group:<module>:<procedure>.<span>`. So is a procedure that an
`> auto:` cache implements, `kedi:group:<module>:<procedure>`.

When telemetry is enabled, each run records the active form as a `form` asset. Its
content is the body's canonical text: comments removed, indentation normalized by
nesting depth, and template text kept exactly. The canonical text's fingerprint
identifies the form, so an edit that only changes comments or indentation is the same
form. Each model call in the form is recorded as the template
`kedi:template:<module>:<qualified name>#<ordinal>` and links to the form.

## Supply Forms

A forms file maps group names to forms. Name a root-module group by its qualified
name, and a group in an imported module by `<module>:<qualified name>`:

```json
{
  "solve.planner": {"body": "forms/planner-3.kedi", "parent": "3fa2c19b...", "note": "one call"},
  "lib/helpers:helper.draft": {"text": "[answer] << Answer <x> in one word."}
}
```

| Field | Meaning |
| --- | --- |
| `body` | A file of unindented body statements, relative to the forms file |
| `text` | The same statements inline; give exactly one of `body` and `text` |
| `parent` | Optional 64-character fingerprint of the form this one derives from |
| `note` | Optional reason for the form, written by whoever produced it |

```bash
kedi solve.kedi --forms forms.json
kedi solve.kedi --test --forms forms.json
```

A form for a group or module that the program does not have is an error. From
Python, pass the same mapping to `compile_program(program, forms=...)`, either loaded
with `kedi.forms.load_forms(path)` or built from `kedi.forms.Form(text, parent=...,
note=...)` values. `run_tests` and `run_evals` also accept `forms=`, so you can check
a form against the program's tests before using it.

## The Gate

Every form passes a gate before the program runs. If a form breaks a rule, Kedi
rejects it, names the rule, and runs nothing:

1. **Parse**: the form parses as the group's body under the group body rules. A form
   for an `> optimize:` span must be exactly one template block. A span that already
   runs a `.optimized.json` result does not take a form.
2. **Python**: the form contains no Python, even when the source body does. That rules
   out Python blocks, backtick statements, expressions, and types, Python conditions,
   and loops, whose iterables are Python. Kedi values are Python globals, so Python
   could bypass the interface rules. Move any computation that needs Python outside
   the group.
3. **Interface**: the form binds every name that the source body binds and later code
   reads, with the same declared type. It reads only names that are visible at the
   group. It does not bind a name visible before the group, or reassign one with
   `:=`, unless the source body does too.

```text
Error: Form for group 'solve.planner' rejected (interface): the form must bind 'plan', which code after the group reads
```

## Provenance

The `form` asset records who produced the active form in its `producer` metadata:
`source`, `supplied`, `optimizer`, `codegen`, or `runtime`. It also records the
producer's `note`, and the declared parent as the asset's `parent_fingerprint`. None
of these is part of the form's identity: the same text is the same form, whoever
produced it. An optimizer result runs as the optimizer's form, with its prompt prefix
ahead of the span's text and the span's source form as its parent.

## Propose Forms at Run Time

An agent or a search loop can propose a form while the program is loaded:

```python
runtime = compile_program(parse_program(source))
fingerprint = runtime.propose_form(
    "solve.planner",
    "[plan] << Write a numbered plan for <problem>.",
    parent=previous_fingerprint,
    note="one call instead of two",
)
runtime.run_main()
```

If the form passes the gate, it runs from the group's next execution in this process,
and the call returns its fingerprint. An execution already inside the group finishes
with the form it started with. If the form is rejected, `FormError` names the broken
rule and the active form keeps running. Saving an accepted form, for example in a
forms file, is up to the caller.

## Compare Forms

[kedi-autobench](https://github.com/kedi-lang/kedi-autobench) records forms as
Autobench assets. A variant selects forms with `kedi.form.<group>` factors, and
`kedi_autobench.variant_forms(ctx.variant)` turns them into the `forms=` mapping.
A form that names its parent is recorded as that version's child, even when the
parent ran in an earlier record. You can pool one record per generation into a single
lineage, then compare the first and last generations with `root` and `leaf`
selectors.
