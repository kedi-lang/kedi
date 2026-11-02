# Installation and Registries

Kedi installs package source into a user-local registry. Installation validates
the manifest, Python compatibility, source containment, file kinds, and bounded
tree size before replacing an existing installed copy.

## Install a Local Package

From a package root:

```console
$ kedi install
Installed kedi_http ...
```

Or identify the manifest:

```console
$ kedi install path/to/package.kedi
```

The command validates and stages `package.kedi` and its declared source directory,
installs `python_dependencies` into the interpreter running Kedi, then commits
the staged package. Dependency errors leave the previous Kedi package intact.
Python environment changes are not part of the registry's atomic transaction.

Installing the same package name replaces the previous installation as one
validated transaction. Do not edit installed registry files manually; reinstall
from a source package.

## Add a Named Package

```console
$ kedi add textkit
$ kedi audit
```

Named add uses [the public Kedi registry](https://registry.kedi-lang.org).
It reads `v1/package/<name>.json`, rejects yanked or revoked packages, and
fetches the exact verified Git commit in that record rather than repository
HEAD. The manifest must match the registered package name and version.

`textkit` provides deterministic whitespace normalization, word counting, and
slug creation without a model or API key:

```kedi
> import: textkit

> show: <slugify(`"Release Notes"`)>
```

The output is `release-notes`. Named and Git installs automatically install
requirements listed in `python_dependencies` into the same interpreter running
the command. Dependency-free packages such as `textkit` do not run pip or uv.

### Development Registries

Set `KEDI_REGISTRY_URL` to use another HTTPS registry or a local HTTP server:

```console
$ KEDI_REGISTRY_URL=http://127.0.0.1:8767 kedi add textkit
$ KEDI_REGISTRY_URL=http://127.0.0.1:8767 kedi audit
```

Plain HTTP is accepted only for loopback addresses. Use the same registry
override for installation and subsequent audits.

For local registry-contract testing, set `KEDI_REGISTRY_MOCK_ROOT` to a directory
whose children are package source directories:

```console
$ export KEDI_REGISTRY_MOCK_ROOT=/absolute/path/to/mock-registry
$ kedi add kedi_http
```

The mock path is development infrastructure, not a production trust mechanism.

## Add an Explicit GitHub Package

```console
$ kedi add git+https://github.com/user/project.git
```

Kedi accepts credential-free `https` URLs hosted on `github.com`. It performs a
shallow, no-checkout clone with blob filtering, reads the root `package.kedi`,
and sparse-checks out only the declared source tree. The checked-out commit is
printed and recorded.

Arbitrary hosts, embedded credentials, unsafe source patterns, manifest
symlinks, and files escaping the checkout are rejected. A Git URL is an explicit
source install; it is separate from verified registry resolution and is reported
as `unverified` by `kedi audit`.

## Registry Location

The default home is:

```text
~/.kedi/
  registry/
    kedi_http/
      package.kedi
      .kedi-install.json
      src-or-copied-source...
```

Set `KEDI_HOME` to select the Kedi home used by package installation/resolution:

```console
$ export KEDI_HOME=/absolute/path/to/kedi-home
```

The override must be absolute. Relative values are rejected so changing the
working directory cannot silently switch registries.

## Installation Receipts

Each installed package has `.kedi-install.json` containing Kedi-owned provenance
such as source kind, source path, manifest digest, and, for Git, normalized URL
and commit. The receipt supports diagnosis and integrity checks; it is not a
signature or security audit.

Do not publish a source-owned `.kedi-install.json` and do not treat receipt
fields as package-controlled metadata.

## Audit Installed Packages

`kedi audit` reads receipts and checks the registry's `v1/audit.json` index once.
It does not execute imported packages, fetch repository HEAD, or update code.
Findings match the package name, repository, and exact verified commit; an
unchanged version string does not conceal a revoked commit.

| Status | Meaning |
| --- | --- |
| `active` | The installed commit remains active in the registry. |
| `superseded` | A historical commit, not withdrawn. |
| `yanked` / `revoked` | A withdrawn commit; review the registry's reason and replace it. |
| `unknown` | The verified commit is absent from the audit index. |
| `malformed` | The receipt or installed manifest fails validation. |
| `unverified` | A local or explicit Git source without verified registry provenance. |

Exit code `0` means no actionable findings, `1` means a yanked, revoked,
unknown, or malformed installation, and `2` means the online audit could not
complete. `unverified` is not a security endorsement even though it does not
make the command fail. The audit checks manifest receipts and registry status,
not the contents of every installed source file.

## Source Safety and Limits

Installation accepts regular files and directories within the declared source
tree. It rejects path traversal, symlink escapes, special files, sparse-checkout
patterns, missing `main.kedi`, and trees exceeding configured count or size
bounds.

Installed-package resolution repeats boundary checks. It rejects a symlinked
package root or manifest, a directory name that disagrees with the manifest,
missing metadata, and source files outside the package.

These checks protect the registry layout and installation transaction. They do
not make package code safe to execute.

## Executable-Code Security

Importing a third-party Kedi package can execute its prelude, Python blocks, and
top-level statements with the Kedi process's host permissions. Review the exact
source and commit, install in an isolated Python environment, and restrict host
credentials and filesystem access as you would for any Python dependency.

A registry-verified commit pins identity and source provenance. It does not
sandbox behavior, prove correctness, or approve capabilities.
