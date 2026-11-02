# Jev Decision Evidence

Inspect the evaluation that produced a binding without changing the binding's native type or making another provider call. For a field-by-field API reference, see [Decision Evidence](../python-api/decisions.md).

## Decision Evidence in Kedi

Jev-backed output bindings retain their decision evidence without changing the
normal output type:

````kedi
```
from kedi import decision_info
```
> adapter: pydantic
> model: typesafe/jev-latest

[ticket] = I was charged twice for the same invoice.
>> It is [duplicate_charge: bool] that <ticket> describes a duplicate charge.

`print(decision_info("duplicate_charge").probability)`
`print(decision_info("duplicate_charge").threshold)`

[ticket] := I was charged only once.
`print(decision_info("duplicate_charge").matches_inputs(ticket=ticket))`
````

The helper resolves the nearest lexical binding. For an interactive session,
`session.decision_info("duplicate_charge")` inspects a top-level binding without
executing another fragment. Unknown names raise `NameError`; values without Jev
evidence return `None`.

Evidence includes the provider/model, source location, Noul probability, the
threshold used for boolean conversion, and Choice/Score confidence and
distributions when the provider supplied them. Noul does not gain an invented
confidence value. Multiple fields produced by one template share one call record
while retaining field-specific evidence. For a nested or multilabel output,
`answers` contains each matching child path separately; Kedi never merges several
Noul answers into one probability.

`state_fingerprint`, `criteria_fingerprint`, `config_fingerprint`, and
`request_fingerprint` identify the evaluated provider state, resolved criteria,
decision configuration, and their combined request. These are one-way identities,
not recoverable prompt copies.

`matches_inputs(**values)` performs a local fingerprint comparison over the
explicit `<name>` inputs. It makes no model or tool call and does not claim that
history or external state is unchanged. Reassigning an input leaves the historical
record intact, so this check can report `False`. Reassigning the output removes the
record from that current binding. Kedi does not automatically re-evaluate a
decision or apply a new threshold.

Decision records participate in strict interactive-session dump/load. They contain
versioned fingerprints, not recoverable copies of the original input text. A
fingerprint is an identity check, not anonymization; do not publish snapshots that
may contain sensitive provider evidence.


## Capture Decisions from Python Calls

`kedi.capture_decisions()` collects the completed template decisions produced by
`@kedi.query` and `@kedi.bind` calls without changing their return types:

```python
import kedi

@kedi.query(adapter="pydantic", model="typesafe/jev-latest")
def review(ticket: str) -> bool:
    """kedi
    >> It is [duplicate: bool] that <ticket> describes a duplicate charge.
    = `duplicate`
    """

with kedi.capture_decisions() as run:
    duplicate = review("I was charged twice for the same invoice.")

for decision in run.decisions:
    print(decision.field_path, decision.probability, decision.threshold)
```

`decisions` is a tuple of immutable `DecisionInfo` views, one per output binding,
in call completion order. Multiple fields share their `call_id`. These historical
snapshots remain accessible after the function returns. Nested capture scopes
also contribute to their enclosing scope; separate concurrent scopes are isolated.
Use regular `with` around `await` calls and await desired work before exiting.
Exit does not wait for pending work; the capture stops accepting records.

Capturing makes no additional model calls. Response-cache hits, failed calls, and
outputs without provider evidence do not create decision records. Successfully
completed decisions remain available if a later call raises. This captures typed
template outputs, not implicit control-flow claims or arbitrary SDK calls.
