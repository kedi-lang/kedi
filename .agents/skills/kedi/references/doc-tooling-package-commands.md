> Offline source: `kedi-docs-main/docs/tooling/package-commands.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

# Package Commands

## `kedi install`

From a package root:

```bash
kedi install
```

Or provide the manifest:

```bash
kedi install path/to/package.kedi
```

The command validates the manifest and copies its declared source tree.

## Manifest Paths

The default manifest is `./package.kedi`. The `source` path is relative to that
manifest and must contain `main.kedi`.

## `kedi add`

Install one named or Git source:

```bash
kedi add package_name
kedi add git+https://github.com/owner/repository.git
```

Exactly one argument is required.

## Named Registry Packages

Named installs use the future `registry.kedi-lang.org` contract. Until the
service is available, `KEDI_REGISTRY_MOCK_ROOT` can point at package source
directories for local registry testing.

## Explicit GitHub Sources

Git sources must use credential-free `git+https`, host `github.com`, and have
no query/fragment/userinfo. Kedi performs a shallow filtered clone and sparse
checkout of only the manifest's literal source directory.

## Installation Locations

Packages install under:

```text
${KEDI_HOME:-$HOME/.kedi}/registry/<package-name>/
```

`KEDI_HOME` must be absolute.

## Receipts and Reinstallation

`.kedi-install.json` records source kind/path, manifest digest, and Git URL/
commit when applicable. Reinstalling replaces the installed package from the
new validated source.

## Package Command Errors

Missing manifests, invalid names/versions/Python ranges, unsafe paths, source
limits, malformed Git URLs, and registry failures exit nonzero. Package
dependencies are recorded but not installed into the active Python environment.

## Skills Are Not Packages

```bash
kedi skills add --path ./release-review
kedi skills add --repo owner/release-review
```

Choose one source per command. It must contain a root `SKILL.md`; installation
copies that skill into the user skill registry, not the module import namespace.
Installing a skill does not activate it or grant tool permissions. Consult
[Skills](doc-agentic-engineering-skills.md) for explicit loading.
