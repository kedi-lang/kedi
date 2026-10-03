# Local Package Registry

Kedi uses `https://registry.kedi-lang.org` by default. A loopback static server
supports local registry development without changing the public registry or
requiring production DNS.

Generated registry files use this layout:

```text
generated/
  v1/
    audit.json
    index.json
    package/
      package_name.json
```

Start the server from the Kedi repository:

```console
python scripts/serve_package_registry.py path/to/generated
```

The local server defaults to `http://127.0.0.1:8767`; explicitly point the client
at it with `KEDI_REGISTRY_URL=http://127.0.0.1:8767`. To use another port:

```console
python scripts/serve_package_registry.py path/to/generated --port 8877
KEDI_REGISTRY_URL=http://127.0.0.1:8877 kedi add package_name
KEDI_REGISTRY_URL=http://127.0.0.1:8877 kedi audit
```

Plain HTTP is accepted only for loopback registry servers. Non-loopback
registries must use HTTPS. Local registry records may reference an absolute
`file://` Git repository for integration testing; hosted registry records may
not.

`kedi audit` reads installed `.kedi-install.json` receipts and fetches
`v1/audit.json` once. It does not import package code, update packages, or
contact repository HEAD. Exit code `0` means the check completed without an
actionable finding, `1` means a yanked, revoked, unknown, or malformed verified
installation was found, and `2` means the online audit could not complete.
