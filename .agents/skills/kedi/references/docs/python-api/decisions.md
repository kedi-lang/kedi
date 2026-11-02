# Decision Evidence

Decision evidence describes a completed evaluation without replacing its typed
result. It is available only when the selected model integration supplies it;
Kedi does not fabricate probability metadata for ordinary generated booleans.

## Capture Completed Calls

```python
import kedi


@kedi.query(adapter="pydantic", model="typesafe/jev-latest")
def review(ticket: str) -> bool:
    """kedi
    >> It is [duplicate: bool] that <ticket> describes a duplicate charge.
    = `duplicate`
    """
    ...


with kedi.capture_decisions() as capture:
    duplicate = review("The same invoice was charged twice.")

for info in capture.decisions:
    print(info.field_path, info.probability, info.threshold)
```

This live example requires `kedi[typesafe]` and `TYPESAFE_API_KEY`.
`capture_decisions() -> ContextManager[DecisionCapture]` makes no additional
request. `DecisionCapture.decisions` is a tuple of immutable `DecisionInfo`
views in completion order, one per output binding. Sibling fields share a
`call_id`. Nested captures also contribute to the enclosing capture.

Use regular `with` around awaited calls and await all intended work before
exit. Exiting stops collection; it neither starts nor joins pending work.
Response-cache hits and outputs without evidence add no records. A later
failure does not remove earlier successful records. Implicit control-flow
claims and arbitrary direct SDK calls are not captured template bindings.

## Inspect a Binding

`decision_info(name: str) -> DecisionInfo | None` is for active embedded Kedi
Python execution. Outside a fragment, use
`InteractiveSession.decision_info(name)` for a top-level session binding.
Unknown names raise `NameError`; known values without evidence return `None`.
Calling the root helper outside Kedi execution raises `RuntimeError`.

## Information Fields

| Fields | Meaning |
| --- | --- |
| `call_id`, `field_path` | Shared evaluation identity and this output's path |
| `provider`, `model` | Evidence source, with model optional |
| `source` | `DecisionSource(path, line, column)`, each location component optional |
| `prompt_fingerprint`, `inputs` | Rendered prompt identity and named input fingerprints |
| `request_fingerprint`, `state_fingerprint`, `criteria_fingerprint`, `config_fingerprint` | Optional identities reported by the integration |
| `answers` | Immutable matching provider answers, including nested output paths |
| `probability`, `threshold`, `comparator` | A single compatible boolean/probability answer's evidence, otherwise `None` |
| `confidence`, `distribution`, `rubric` | Optional single-answer evidence; not inferred when absent |

## Historical Identity, Not Automatic Re-Evaluation

`info.matches_inputs(**inputs) -> bool` compares the explicit named inputs
locally. It makes no model request. Supply exactly the recorded input names.
Computed substitutions, unsupported values, or incomplete input coverage raise
`ValueError` rather than pretending to match.

Changing an input leaves historical evidence intact. Reassigning its output
clears evidence from the current binding. For supported fingerprintable output
values, a later in-place mutation also makes binding lookup return `None`.
An already captured `DecisionInfo` remains a historical snapshot. No operation
here applies a new threshold or certifies that external state is unchanged.

Fingerprints are versioned one-way identities, not copies of original inputs,
encryption, or anonymization. Unknown object graphs cannot be assumed comparable.
Session persistence remains subject to the full snapshot restrictions in
[Incremental Execution](../runtime/interactive-execution.md).
