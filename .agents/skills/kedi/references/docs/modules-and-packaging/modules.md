# Modules

A module is a `.kedi` source file compiled as part of a root program. Its
top-level code initializes module state; its export directive selects which
bindings are published to importers.

## Module Files and Paths

Given:

```text
project/
  main.kedi
  services/
    users.kedi
    audit.kedi
```

`main.kedi` imports the nested files without suffixes:

```kedi
> import: services/users
> import: services/audit

= <user_count()> users
```

Each slash-separated segment must be a valid identifier. Absolute paths,
`.`/`..`, backslashes, empty segments, and a written `.kedi` suffix are not
module names.

## Relative Resolution

The first candidate is relative to the file containing the import, not the
process working directory and not always the root program:

```text
project/
  main.kedi
  features/
    report.kedi
    formatting.kedi
```

An import of `formatting` inside `features/report.kedi` resolves to
`features/formatting.kedi`. Use this property to keep feature-local module
graphs relocatable.

A source compiled only from an in-memory string has no relative directory.
Without a source path it can import registered bundled modules, but unresolved
file modules produce an error explaining that a source path is required.

## Resolution Order

For each import Kedi checks:

1. the importer-relative `.kedi` path;
2. a bundled internal module with that name;
3. the installed package registry.

The first existing file wins. Registry validation is deferred until local and
built-in candidates miss, so a corrupt installed package cannot break a valid
higher-priority local import.

## Package Modules

For an installed package `kedi_http`:

```kedi
> import: kedi_http
> import: kedi_http/client
```

The root import resolves the package source's `main.kedi`; the second resolves
`client.kedi`. Installed paths are checked for containment, symlink escapes,
manifest validity, package-name agreement, and source layout before loading.

## Initialization Once per Compilation

A resolved module executes at most once for one root compilation. Later imports
reuse its initialized exports and publish those bindings at each import
statement's source position.

This prevents duplicate prelude and top-level side effects in a diamond:

```text
main -> reports -> shared
     -> alerts  -> shared
```

`shared.kedi` initializes once, even though both branches import it. A separate
root compilation initializes it again; module state is not a process-global
singleton.

## Module Prelude and Side Effects

A module may begin with a Python prelude and may execute top-level statements:

````kedi
```
import re
PATTERN = re.compile(r"^[a-z0-9-]+$")
```

@valid_slug(value: str) -> bool:
  = `PATTERN.fullmatch(value) is not None`

> export:
  valid_slug
````

Prelude failures fail module initialization and therefore the importing
program. Keep initialization lightweight and deterministic. Network calls,
filesystem writes, and mutable global setup are usually better placed in
explicit procedures.

## Cycles and Missing Modules

Do not design cyclic module imports. A module graph should flow from higher-level
features toward lower-level shared modules. Missing modules report the searched
candidate paths; invalid module segments fail before filesystem lookup.

Use the LSP's definition and diagnostics support to inspect resolution rather
than compensating with working-directory-dependent paths.
