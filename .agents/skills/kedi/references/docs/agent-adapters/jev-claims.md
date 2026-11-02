# Jev Claims and Thresholds

A claim asks for a decision, not generated prose. Its acceptance threshold is application policy, not a proof that the claim is true.

## Claims and Thresholds

```kedi
> adapter: pydantic
> model: typesafe/jev-latest
> settings:
  typesafe_threshold: 0.9

> if: Ankara is the capital of Turkey
  = established
> else:
  = not established
```

Replace `pydantic` with `langchain` to use the other adapter. The default decision
rule is **probability > 0.85**; equality is false. Explicit request/profile settings
override the model constructor threshold without changing the shared model.
Thresholds must be finite numbers in [0, 1], not strings or booleans.

This threshold applies to boolean outputs and individual multi-label memberships.
It does not threshold raw probabilities, categorical choices, or rubric scores.
A provider probability is not a guarantee of empirical accuracy.


## Migration from 0.1

The extended integration defaults to strict `> 0.85`, rather than the old 0.5
threshold. Set an explicit threshold when migrating an existing decision policy.
Do not assume an unwrapped upstream model enforces this policy: Kedi rejects that
model for claims when the policy cannot be enforced.

The package pins the upstream integration versions because it extends schema and
decision behavior at tested framework boundaries. Upgrade those pins together with
the compatibility tests, not independently.
