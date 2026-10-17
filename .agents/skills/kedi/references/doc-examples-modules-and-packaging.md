> Offline source: `kedi-docs-main/docs/examples/modules-and-packaging.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Directory Layout](#directory-layout)
- [Package Entry Module](#package-entry-module)
- [Nested Module](#nested-module)
- [Manifest](#manifest)
- [Install and Consume](#install-and-consume)
- [Local Development Without Installation](#local-development-without-installation)
- [Expected Result and Isolation](#expected-result-and-isolation)

# Modules and Packaging

This example creates a package whose public surface contains a type and two
procedures while its parsing helper remains private.

## Directory Layout

```text
catalog/
  package.kedi
  src/
    catalog/
      main.kedi
      formatting.kedi
consumer/
  report.kedi
```

## Package Entry Module

`catalog/src/catalog/main.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
~Product(name: str, price: float)

@_normalize_name(name: str) -> str:
  = `name.strip().title()`

@make_product(name: str, price: float) -> Product:
  = `Product(name=_normalize_name(name), price=price)`

@inventory_value(products: list[Product]) -> float:
  = `sum(product.price for product in products)`

> export:
  Product
  make_product
  inventory_value
```

Only exported names enter an importer. `_normalize_name` remains private even
though exported procedures can use it.

## Nested Module

`catalog/src/catalog/formatting.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@format_currency(value: float) -> str:
  = `f"${value:,.2f}"`

> export:
  format_currency
```

## Manifest

`catalog/package.kedi`:

<!-- Manifest snippet: parse-checked as package.kedi; save under that exact filename. -->
```kedi
> package: catalog:
  version: 1.0.0
  source: src/catalog
  python: python@3.11-3.14
  python_dependencies:
    pydantic>=2
```

The manifest is declarative only. It records PEP 508 dependencies but does not
install them into the active Python environment.

## Install and Consume

From `catalog/`:

```bash
kedi install
```

`consumer/report.kedi`:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
````kedi
> import: catalog:
  Product
  make_product
  inventory_value

> import: catalog/formatting:
  format_currency

[products: list[Product]] = ```
return [
    make_product("keyboard", 120.0),
    make_product("mouse", 45.5),
]
```

[total: float] = `inventory_value(products)`

= `format_currency(total)`
````

Selective imports place the listed names directly in scope; they do not create
a `catalog` namespace object. Imports resolve at their source position, and a
module initializes at most once per root compilation.

## Local Development Without Installation

Sibling modules resolve relative to the importing file:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> import: formatting:
  format_currency
```

Use relative modules while developing one project. Install a package when its
root import must be available to unrelated projects. Third-party packages can
execute embedded Python with the importing process's permissions; package
integrity is not a sandbox.

## Expected Result and Isolation

The consumer prints `$165.50` without any model calls. The sibling-import
fragment belongs next to `formatting.kedi`, not in `consumer/` as drawn above.
An importer cannot selectively import `_normalize_name` because it is not exported.

For repeatable installation checks, set `KEDI_HOME` to a temporary directory
before both installing and running the consumer. This isolates the package
registry from your normal installation; it is not a Python execution sandbox.
The [Local Package Tutorial](doc-modules-and-packaging-local-package.md) walks
through the receipt and import resolution rules. This catalogue example is also
executed from its published files in the documentation tests.
