# Modules, packages, and filesystem work

## Resolve and export explicitly

`> import: utilities` loads a sibling `utilities.kedi` first, then bundled modules, then installed packages. A nested `services/profiles` path follows the importing file, not its caller. Selective imports use an indented name list and bind those names directly: there is no implicit `utilities.name` namespace.

Only `> export:` names become visible; a module without exports exposes nothing. `> export: *` exposes public names except `_` names and reserved `args`. Do not explicitly import/rebind `args`. Duplicate names in one selective list fail. Modules initialize at most once per root compilation; imports publish bindings at their source position. Later declarations/imports win names.

Export a facade of typed procedures/models/profiles. Imported profiles retain their private tool/child dependencies; do not export all helpers just to make that facade work. Use [module example](../assets/examples/modules/main.kedi) and its [pricing module](../assets/examples/modules/pricing.kedi).

## Packages

A manifest is **exactly `package.kedi`**, with one `> package: name:` and metadata only. No executable statements/imports. Names use lowercase letters/digits/underscores and start with a letter. `source` is a literal relative directory contained in the package root and must contain `main.kedi`. No parent traversal, symlink escape, or sparse patterns.

Supported metadata: author, contact, license, version, source, python, python_dependencies. Version follows PEP 440. Python requirements use `python@3.11` or inclusive `python@3.11-3.14`, not general pip version syntax. Dependencies are PEP 508 strings; `kedi install` and `kedi add` install them into the interpreter running Kedi using its pip or `uv pip --python`. Imports and manifest inspection never install dependencies. A dependency failure preserves the previous Kedi package, but Python environment changes are not rolled back by Kedi.

```bash
kedi parse path/to/package.kedi
kedi install path/to/package.kedi
```

Installation copies the manifest/source under `$KEDI_HOME/registry/<name>`; the default home is `~/.kedi`. An override must be absolute. To test installation, set a temporary absolute `KEDI_HOME` rather than overwriting a user's registry. Package root imports map to `main.kedi`.

`kedi add NAME` uses a verified exact registry commit; explicit credential-free GitHub `git+https` is a separate unverified source path. `kedi audit` checks receipts/status, not all code or sandbox permissions. Imported package Python executes with process authority. Read [installation and registries](doc-modules-and-packaging-installation-and-registries.md) before publishing/installing third-party packages.

## Bundled modules

| Module | Use and boundary |
| --- | --- |
| `filesystem` | Filesystem tools and profiles; consult exact signatures/export list |
| `require` | Check Python module import availability; does not install or check versions |
| `errors` | `ModuleNotInstalledError` |
| `sandbox` | `execute_code` and sandbox profile, requires Monty; `fail_fast=True` raises execution failures |
| `helpers` | Experimental `llm_approval`, needs enabled tool reasons |
| `decisions` | Shared optional decision criteria; import does not select a model |
| `this`, `wordle` | Demonstrations; `this` imports with output, wordle needs optional GUI dependencies |

Do not shadow a bundled module accidentally with a sibling file. Treat [filesystem](doc-modules-and-packaging-filesystem.md) as the API reference rather than inventing signatures. Bundled filesystem tools restrict paths to process CWD by default and protect secret files; relative Python `Path(...)` uses CWD too. Source-relative resource reads should derive from `__file__`.

Read [imports/exports](doc-modules-and-packaging-imports-and-exports.md), [manifests](doc-modules-and-packaging-package-manifests.md), and [built-ins](doc-modules-and-packaging-built-in-modules.md) for the detailed contracts. Ordinary embedded Python is not protected by the filesystem tool policy.
