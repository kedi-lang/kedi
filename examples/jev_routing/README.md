# Jev-backed ticket routing

Use Jev for finite decisions, and let Kedi turn those decisions into program
control flow. This example does not ask Jev to generate a free-form reply.

## Run

From the repository root, install the optional integration:

```bash
uv sync --dev --extra typesafe
source .venv/bin/activate
```

Set `TYPESAFE_API_KEY` in the environment or the checkout's `.env`, then:

```bash
cd examples/jev_routing
kedi main.kedi --adapter pydantic
```

`> model: typesafe/jev-latest` intentionally overrides the default generative
model for this program. The supplied ticket describes ongoing duplicate charges;
the expected action is `Escalate to billing.` The program also displays the
urgency probability and the threshold used for the decision.

## Criteria are not a text-generation prompt

`> import: decisions` exposes the optional decision primitives directly in Kedi.
This example imports only `BooleanCriteria` and `ChoiceCriteria`. The import
does not select a model, configure credentials, or make a provider request.
Python callers still use explicit imports from `kedi.decisions`.

`Team` is an `Annotated` finite choice with explicit `ChoiceCriteria`. `Urgent`
is a Boolean with positive and negative `BooleanCriteria`. Both are requested
in one template; the result is a string literal and a Python bool, respectively.

`typesafe_threshold: 0.85` controls the Boolean decision, not the category choice.
Displaying `<urgent>` first resolves the capture. `decision_info("urgent")`
then exposes the provider's probability and the effective threshold without
another inference. Metadata may not exist while a capture is still pending.
A low score means the escalation criterion
was not established at this threshold, not that the ticket is certainly harmless.

The native `> if:` consumes the already resolved bool. It does not ask a second
model to reinterpret the same question. The regression test checks both sides
of the threshold and that both fields share one Jev request. The probability
shown by a live run is not a guarantee of calibration or real-world correctness.
