# Modules and Packages { #modules-and-packaging }

Kedi modules split a program into explicit public surfaces. Packages distribute
one module tree through the local Kedi registry; they do not change the module
language.

## File Modules

Every `.kedi` file can be a module. Imports use slash-separated module names
without the `.kedi` suffix:

```kedi
> import: services/users

= <find_user(`7`)>
```

The import above first looks for `services/users.kedi` relative to the importing
file. A module is initialized once per root compilation, even when several
import paths form a diamond.

## Public Surfaces

Modules are private by default. Only names listed by `> export:` enter an
importer's environment:

```kedi
> export:
  User
  find_user
```

Imports do not create namespace objects. Exported procedures, types, profiles,
and values are bound directly at the import statement's source position.

## Packages

A package has a declarative `package.kedi` and a source directory containing
`main.kedi`. Installation copies the validated manifest and source tree to the
local registry. Importing the package name loads `main.kedi`; importing
`package/submodule` loads `submodule.kedi`.

Package metadata can declare Python compatibility and PEP 508 dependencies, but
the installer does not create an environment or install those Python packages.

## Resolution and Security

Module resolution order is:

1. a file relative to the importer;
2. a bundled internal module;
3. a package installed under the local Kedi registry.

This lets a project deliberately provide a local module without a registry
package shadowing it. It also means a local file named `filesystem.kedi` takes
precedence over the bundled `filesystem` module.

Imported modules may contain embedded Python. Treat third-party Kedi packages as
executable code with the same host permissions as the Kedi process. Registry
identity and digest checks are integrity controls, not a sandbox.

## Section Map

- [Modules](modules.md) covers file resolution and initialization.
- [Imports and Exports](imports-and-exports.md) defines public binding behavior.
- [Package Manifests](package-manifests.md) defines `package.kedi`.
- [Installation and Registries](installation-and-registries.md) covers `install`
  and `add`.
- [Built-In Modules](built-in-modules.md) documents the bundled public modules.

## In This Section

**Module System**

- [Modules and Resolution](modules.md)
- [Imports and Exports](imports-and-exports.md)

**Standard Modules**

- [Built-In Modules](built-in-modules.md)
- [Filesystem](filesystem.md)
- [Example Modules](example-modules.md)

**Distribution**

- [Package Manifests](package-manifests.md)
- [Build a Local Package](local-package.md)
- [Installation and Registries](installation-and-registries.md)
- [Manifest Reference](../reference/package-manifest.md)
