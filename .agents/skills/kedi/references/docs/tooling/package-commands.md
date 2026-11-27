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
`python_dependencies` into the interpreter running Kedi and records the distributions that
were added, then commits the package.

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

`.kedi-install.json` records:
- the source kind and path;
- the manifest digest;
- the Git URL and commit, when there is one;
- a `lifecycle` record: what the installation created, and how and when each thing is removed.

```json
"lifecycle": {
  "installed_as": "explicit",
  "created": [
    {"kind": "package_directory", "path": ".", "removed": "with the package"},
    {"kind": "runtime_directory", "path": ".kedi-runtime", "removed": "with the package"},
    {"kind": "python_distribution", "name": "httpx", "version": "0.28.1",
     "target": "/path/to/python", "removed": "by kedi remove --with-python-dependencies, ..."}
  ],
  "changed": []
}
```

- **What is recorded:** the distributions that `python_dependencies` added to Kedi's
  interpreter. A distribution that was already installed but got upgraded is listed under
  `changed`, and Kedi never reverts it.
- **Reinstalling:** it replaces the installed package from the new validated source. The
  record keeps the distributions the first install added.

## `kedi remove`

```bash
kedi remove package_name
kedi remove package_name --with-python-dependencies
kedi remove package_name --dry-run
```

It deletes the installed package directory, with its `.kedi-runtime/`, together with an
interrupted installation's staging and backup directories and the package's lock file.

- **A needed package stays.** A package another installed package lists in
  `kedi_dependencies` is not removed, and the error names its dependents.
- **Orphans go with it.** The dependencies it was the last to need, and that were
  installed only as dependencies, are removed too. `--dry-run` shows them.

- **Python by default:** untouched. The distributions the package added stay, because
  something else may use them, and `kedi remove` names them.
- **`--with-python-dependencies`:** uninstalls them too, but keeps any distribution that:
  - another installed package needs;
  - another installed distribution requires (extras included, whatever their condition);
  - has changed version since the package installed it.

  Kedi prints each one it keeps, with the reason.
- **`--dry-run`:** prints the plan without removing anything.

A package installed before Kedi kept this record is removed the same way. Its Python
distributions are unknown, so they stay.

Package bridges keep their run-time files inside the package:
- `KEDI_PACKAGE_RUNTIME_DIR` names its `.kedi-runtime/`;
- `KEDI_BRIDGE_CACHE` points below it for the bridge SDKs' preflight cache.

## Kedi Dependencies

The Kedi packages a manifest lists in `kedi_dependencies` install with it:

```console
$ kedi install
Installed dependency kedi_http at /home/me/.kedi/registry/kedi_http
Installed release_tools at /home/me/.kedi/registry/release_tools
```

- **Resolution comes first.** Kedi reads the whole graph before it changes anything:
  - the registry's active version of each dependency, checked out at its verified commit;
  - the specifiers;
  - cycles;
  - the installed packages a change would break.

  A cycle, or a conflict that no version meets, is refused with the requirements that
  clash, and nothing changes.
- **What is reused.** An installed package that satisfies every requirement on it is kept.
  Otherwise the registry's active version replaces it, but only if it also satisfies every
  installed package that needs it.
- **One transaction.** Dependencies install first, then the package. If a step fails, the
  packages the installation added are removed and the ones it replaced come back. The error
  says so, and names any Python distributions they added, which stay installed.
- **Explicit or dependency.** The install record says how each package came:
  - `installed_as: explicit` for the package asked for, and `dependency` for the others;
  - an explicit package stays explicit when a dependent upgrades it;
  - `kedi add` of a dependency makes it explicit.

  `kedi remove` takes along only orphaned dependencies.
- **One version per Kedi home.** A package is installed once; there is no side-by-side
  versioning.

`kedi audit` reports an installed package whose Kedi dependency is missing or no longer
satisfies it.

## Package Environments

A package adapter with `environment: uv` runs its `command:` in a Python environment of the
package's own. The package's `source` holds a uv project (`pyproject.toml` and `uv.lock`) at its
root.

- **What gets installed.** `kedi install` and `kedi add` run `uv sync --frozen --no-dev
  --no-install-project` into the installed package's `.kedi-env/`:
  ```text
  ${KEDI_HOME:-$HOME/.kedi}/registry/<package-name>/.kedi-env/
    venv/                  the environment
    python/, python-bin/   its Python, one uv manages
  ```
- **The environment's Python.** uv downloads a Python it manages into the package, so
  upgrading or removing a system Python cannot break the environment. Set
  `UV_PYTHON_PREFERENCE` (for example `system`) to let uv use an installed Python instead.
- **The cache.** uv's cache lives in `.kedi-env/cache/` while the sync runs and is removed after
  it.
- **When it fails.** If the sync fails, the new package is deleted and the previous version
  comes back.
- **uv.** It must be installed: `kedi python install uv`. Kedi checks for it, and for the
  project files, before it changes anything.
- **Running.** The command's program comes from the environment when the environment has it,
  so `command: python bridge.py` runs the environment's interpreter. Nothing is installed into
  the interpreter running Kedi.
- **Removal.** The install record lists the environment, and `kedi remove` takes it with the
  package.

## `kedi python install`

```bash
kedi python install uv
kedi python install "httpx>=0.27"
```

It installs into the interpreter running Kedi, with its pip, or with uv when pip is missing.
These are your own installs: no package records them, so `kedi remove` never removes them.
Kedi looks for uv on `PATH`, then beside its interpreter, then in the `uv` package, so
`kedi python install uv` is enough. An environment with neither pip nor uv needs one of them
first: run `python -m ensurepip`, or install uv itself.

## Package Command Errors

These exit nonzero:
- missing manifests;
- invalid names, versions or Python ranges;
- unsafe paths and source limits;
- malformed Git URLs;
- registry failures;
- failed Python dependency installation, which preserves the previous Kedi package;
- removing a package that is not installed.

## Skills Are Not Packages

```bash
kedi skills add --path ./release-review
kedi skills add --repo owner/release-review
```

Choose one source per command. It must contain a root `SKILL.md`; installation
copies that skill into the user skill registry, not the module import namespace.
Installing a skill does not activate it or grant tool permissions. Consult
[Skills](../agentic-engineering/skills.md) for explicit loading.
