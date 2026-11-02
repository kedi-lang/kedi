# Imports and Exports

Imports publish selected module bindings directly into the current source
environment. Exports are explicit so implementation helpers remain private.

## Import a Module

```kedi
> import: profiles

= `get_profile("Ada")`
```

The imported module must explicitly export `get_profile`. Imports do not bind a
`profiles` namespace and do not expose unexported names. The backtick expression
preserves the native `Profile`. A sole angle procedure call on a return also
preserves its native result; surrounding text renders it. An ordinary
`<name>` value substitution renders to text.

## Explicit Exports

In `profiles.kedi`:

```kedi
~Profile(name: str, id: int)

@get_profile(name: str) -> Profile:
  = `Profile(name=name, id=1)`

@normalize_internal(name: str) -> str:
  = `name.strip()`

[profile_id: int] = `1`

> export:
  Profile
  get_profile
  profile_id
```

`normalize_internal` remains usable inside the module but is unavailable to
importers. An export name must resolve to a top-level declaration or value in
that module.

## Selective Imports

An importer can request only part of the exported surface:

```kedi
> import: services/profiles:
  Profile
  get_profile
```

Every selected name must be exported by the target and must appear once in the
list. A missing, private, or duplicate name is an error. Selective import is
appropriate for documenting dependencies and avoiding collisions in large
programs.

## Star Exports

`> export: *` exports every public top-level name except the runtime-owned `args`
binding. Imported procedures still read the caller's CLI arguments; those
arguments are not re-exported or rebound by the import. Explicitly importing
an exported `args` binding is rejected because the name is reserved.

```kedi
@public_name() -> str:
  = visible

@_internal_name() -> str:
  = hidden

> export: *
```

Names beginning with `_` remain private. Star export is convenient for small
facade modules, but explicit export lists make API reviews and compatibility
changes clearer.

## Re-Exports

Because imported names enter the module environment, an importing module can
export them again:

```kedi
> import: services/profiles:
  Profile

@load_default() -> Profile:
  = `Profile(name="default", id=0)`

> export:
  Profile
  load_default
```

Use re-exports to create a stable facade. Avoid chains of accidental star
re-exports, which make ownership and compatibility difficult to see.

## Source-Order Binding

Imports and declarations bind names where they appear. When several statements
provide the same name, the last binding wins:

```kedi
[label] = before
> import: labels
[label] = after

= <label>
```

The result is `"after"`. This behavior supports deliberate overrides but can
hide collisions. Prefer selective imports or distinct names when both values
matter.

An initialized module's exports are republished at every import position, so a
later repeated import can overwrite a binding changed between imports without
re-running module initialization.

## Private and Non-Exported Names

No export directive means the module publishes nothing. A leading underscore
only matters to star export; an explicit export list should still be used for
the intended public API.

Module Python-prelude helpers are implementation details unless surfaced through
a Kedi value, procedure, type, or profile and explicitly exported.

## Errors

Kedi rejects invalid paths, missing module files, unknown selective names,
duplicate names in one selective list, unknown exports, and malformed export
directives. Import errors include the resolution candidates so local,
built-in, and registry problems can be distinguished.
