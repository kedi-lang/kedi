# TypeSafe integration programs

Frozen copies of the `kedi-typesafe` 0.2 example programs used by Kedi's
integration tests. Keeping these fixtures in Kedi means the tests only require
the installed package, not an ignored development checkout of its repository.
The risk-review example is similarly frozen for the hook contract test.

Run `pytest tests/test_typesafe_capabilities.py tests/test_typesafe_routing.py`.
Both adapters use deterministic transports here. Live evidence belongs in the
TypeSafe package's separate validation records, not in unit-test expectations.
`08_decision_evidence.kedi` also verifies Kedi's binding-level evidence API
without issuing a second provider request.
`09_decision_lifecycle.kedi` covers shared multi-output evidence, historical
input matching, explicit overwrite invalidation, and lexical shadowing.
