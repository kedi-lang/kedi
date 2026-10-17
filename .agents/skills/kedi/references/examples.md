# Complete examples and use cases

## Run the example you need

Use the environment described in [environment](environment.md). Paths below are relative to this skill. Keep resource/module files beside their program when copying. Model examples deliberately omit a hard-coded provider so the host/CLI can select an available backend. The validation script supplies a controlled Pydantic model.

```bash
kedi "$KEDI_SKILL/assets/examples/hello.kedi"
python "$KEDI_SKILL/scripts/verify_examples.py"
```

| Use case | Complete source | What verification asserts |
| --- | --- | --- |
| First program | [hello.kedi](../assets/examples/hello.kedi) | `Hello, Kedi.` |
| Typed deterministic invoice | [native_values.kedi](../assets/examples/native_values.kedi) | Native Invoice, total 36, labels book/pen |
| Branch shadowing, assignment, bounded loops | [control_flow.kedi](../assets/examples/control_flow.kedi) | Outer status pending, count 3, total 6 |
| Deterministic batch transform | [map_batch.kedi](../assets/examples/map_batch.kedi) | Ordered indexed result `[4, 9, 25]` |
| Python write-back and source-relative resources | [python_scope.kedi](../assets/examples/python_scope.kedi), [note.txt](../assets/examples/note.txt) | Count 2; fixture loaded using `__file__` |
| Constraints and isolated mutable defaults | [constraints.kedi](../assets/examples/constraints.kedi) | Typed scores; mutating first tags does not alter second |
| CLI inputs | [cli_args.kedi](../assets/examples/cli_args.kedi) | Missing-option defaults and explicit numeric conversion |
| Handwritten tests and held-out eval | [public_api.kedi](../assets/examples/public_api.kedi) | Two passing tests and eval score 1.0 |
| Reusable pricing module | [main.kedi](../assets/examples/modules/main.kedi), [pricing.kedi](../assets/examples/modules/pricing.kedi) | Explicit imports, native Quote total 900, two module tests |
| Local package | [package.kedi](../assets/examples/package/package.kedi), [module](../assets/examples/package/src/skill_demo/main.kedi), [client](../assets/examples/package_client.kedi) | Install into isolated temporary registry and import |
| Python integration | [python_api.py](../assets/examples/python_api.py), [bound.kedi](../assets/examples/bound.kedi) | Query returns 2; bind returns native 10 |
| Meeting action extraction | [extraction.kedi](../assets/examples/extraction.kedi) | Typed list of Action; missing owner remains None |
| Dependent prompt pipeline | [pipeline.kedi](../assets/examples/pipeline.kedi) | Second request receives first request's access category |
| Stateful raw chat | [raw_history.kedi](../assets/examples/raw_history.kedi) | Second turn has prior project identifier and returns atlas |
| Semantic branch | [claim_branch.kedi](../assets/examples/claim_branch.kedi) | Controlled false claim takes the hold branch |
| Model batch processing | [model_map.kedi](../assets/examples/model_map.kedi) | Per-record captures remain associated with their index |
| Tool-grounded release summary | [tool_profile.kedi](../assets/examples/tool_profile.kedi) | Registered real lookup body executes and returns evidence before summary |
| Two specialist reviews | [parallel_review.kedi](../assets/examples/parallel_review.kedi) | Two children receive explicit context and return separate typed results |
| Additive instructions | [capability.kedi](../assets/examples/capability.kedi) | Version-gated eager capability guidance reaches the model |
| Bound model work | [budget.kedi](../assets/examples/budget.kedi) | **Expected failure:** second generation is rejected after exactly one request |
| Optimizable extraction | [optimize.kedi](../assets/examples/optimize.kedi) | Original typed prompt executes; syntax/eval wiring present; no live GEPA run |
| Generated deterministic helper | [auto.kedi](../assets/examples/auto.kedi) | **Parse only**; includes independent acceptance cases; no generated implementation is claimed |

## Model runs

For an available model configured in `KEDI_ADAPTER_MODEL`:

```bash
kedi "$KEDI_SKILL/assets/examples/extraction.kedi" --adapter pydantic --adapter-model "$KEDI_ADAPTER_MODEL"
kedi "$KEDI_SKILL/assets/examples/tool_profile.kedi" --adapter pydantic --adapter-model "$KEDI_ADAPTER_MODEL"
```

Install the corresponding provider support and supply credentials first. The fixture suite validates Kedi's schema/dataflow/tool integration, not the semantic accuracy of a real provider. No live model, ACP/A2A service, external MCP server, Jev/Laya model, editor, or notebook was invoked during this skill's creation.

The budget example is a negative fixture: use its controlled host test or an actual provider client with observable retries. Its failure is the intended result. The auto example is a generation specification, not a completed algorithm; choosing the mock codegen agent does not make it a valid implementation.

## Adapt a recipe

1. Copy the complete file and any adjacent resources/imports it uses.
2. Define exact types and external inputs. Replace fixture data with explicit caller input or a registered read tool.
3. Preserve native values through procedure boundaries; render only for prompts/messages.
4. Select backend/profile before procedure definition. Add only required tool capabilities.
5. Add handwritten assertions for boundary/failure cases, then parse and run.
6. Inspect real provider/tool evidence when moving from fixtures to an actual integration.

For broader patterns, consult the original [complete program](doc-examples-complete-program.md), [structured extraction](doc-examples-structured-extraction.md), [agent delegation](doc-examples-agent-delegation.md), and [evaluation](doc-examples-evaluation-and-optimization.md). Those manual snippets retain their original context requirements.
