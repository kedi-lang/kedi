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

The command validates the manifest, stages its declared source tree, installs
`python_dependencies` into the interpreter running Kedi, then commits the package.

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

Named installs resolve exact verified commits from `registry.kedi-lang.org`.
`KEDI_REGISTRY_MOCK_ROOT` can point at package source directories for local
registry testing.

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

Python dependencies install into `sys.executable`, independently of the Kedi
source registry location. Activating a venv and using its Kedi command targets
that venv. The command uses that interpreter's pip or `uv pip --python` when
pip is unavailable; it does not select a different Python from PATH.

## Receipts and Reinstallation

`.kedi-install.json` records source kind/path, manifest digest, and Git URL/
commit when applicable. Reinstalling replaces the installed package from the
new validated source.

## Package Command Errors

Missing manifests, invalid names/versions/Python ranges, unsafe paths, source
limits, malformed Git URLs, registry failures, and Python dependency installation
failures exit nonzero. A dependency failure preserves the previous Kedi package;
changes already made by the Python installer are not rolled back.

## Skills Are Not Packages

```bash
kedi skills add --path ./release-review
kedi skills add --repo owner/release-review
```

Choose one source per command. It must contain a root `SKILL.md`; installation
copies that skill into the user skill registry, not the module import namespace.
Installing a skill does not activate it or grant tool permissions. Consult
[Skills](../agentic-engineering/skills.md) for explicit loading.
