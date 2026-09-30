# A reusable order-pricing module

This example needs no model or API key. It shows how ordinary typed code and
tests fit alongside model-backed procedures without inventing a second arithmetic
language. Prices are integer cents, so binary floating-point rounding is avoided.

## Run

Activate the checkout environment, then from the repository root:

```bash
cd examples/module_imports
kedi main.kedi
kedi pricing.kedi --test
```

The quote has `subtotal_cents=49500`, `discount_cents=4950`, and
`total_cents=44550`. Both colocated test cases should pass.

## Module boundary

[pricing.kedi](pricing.kedi) exports `Item`, `Quote`, and `quote`. The entrypoint
imports only `Item` and `quote` and calls the procedure with a defaultable discount.
Module resolution is relative to the importing `.kedi` file.

The Python block does deterministic arithmetic and validation. Discounts round
down to whole cents; the fractional-cent test makes that policy explicit.
Negative prices/quantities and discount percentages outside 0-100 are rejected.
This deliberately omits tax, shipping, and currency conversion rather than
pretending to be a complete checkout service.
