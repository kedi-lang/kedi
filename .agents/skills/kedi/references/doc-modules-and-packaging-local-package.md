> Offline source: `kedi-docs-main/docs/modules-and-packaging/local-package.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

# Build a Local Package

This deterministic package has no model or third-party Python dependency. It
separates a public module entry point from an implementation module.

## Files

```text
release_labels/
  package.kedi
  src/
    release_labels/
      main.kedi
      formatting.kedi
consumer.kedi
```

`release_labels/package.kedi`:

<!-- Manifest snippet: parse-checked as package.kedi; save under that exact filename. -->
```kedi
> package: release_labels:
  version: 0.1.0
  source: src/release_labels
  python: python@3.10-3.14
```

`release_labels/src/release_labels/formatting.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@release_label(name: str, version: str) -> str:
  = <name> v<version>

> export:
  release_label
```

`release_labels/src/release_labels/main.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> import: formatting:
  release_label

> export:
  release_label
```

`consumer.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> import: release_labels:
  release_label

= <release_label(Kedi, 0.4.0)>
```

## Install and Run

From the directory containing `release_labels/` and `consumer.kedi`:

```sh
export KEDI_HOME="$PWD/.kedi-demo"
kedi install release_labels/package.kedi
kedi consumer.kedi
```

The consumer returns `Kedi v0.4.0`. The explicit absolute Kedi home keeps this
exercise separate from the normal user registry. Installation copies the
declared source; edits to the original need reinstallation. The receipt records
provenance but does not make the package trusted.

## What This Demonstrates

The internal `formatting` import resolves relative to `main.kedi`. Importers
receive exported bindings, not a namespace object. The package's Python version
range checks the installer interpreter; it does not download Python. See the
[manifest reference](doc-reference-package-manifest.md) for optional metadata
and dependency fields.
