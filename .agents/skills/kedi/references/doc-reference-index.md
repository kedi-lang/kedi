> Offline source: `kedi-docs-main/docs/reference/index.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

# Find a Reference { #reference }

This section is the compact, searchable inventory of Kedi's public language,
runtime, Python, adapter, and tooling contracts. The conceptual sections linked
from each table provide examples and rationale.

## Find the Rule You Need

| Question | Contract |
| --- | --- |
| Is this initialization, assignment, or model output? | [Outputs and Bindings](doc-core-language-outputs-and-assignments.md) |
| Why is a name unavailable outside a branch or loop? | [Scopes and Binding Lifetime](doc-core-language-scopes-and-bindings.md) |
| Does this condition contact a model? | [Branches and Claims](doc-core-language-control-flow.md) |
| Does this call return an object or rendered text? | [Substitutions and Calls](doc-core-language-substitutions-and-calls.md) |
| Which model/profile applies here? | [Profiles and Composition](doc-agentic-engineering-profiles.md) |
| Which arguments can Python callers configure? | [Public Parameters](doc-python-api-public-parameters.md) |
| Why was a tool rejected? | [Approvals](doc-agentic-engineering-approvals.md) and [Filesystem Policy](doc-modules-and-packaging-filesystem.md) |
| What enters the next model request? | [History](doc-runtime-history.md) and [Artifact Policy](doc-runtime-artifact-policy.md) |

For a sequential introduction, return to [Learn Kedi](doc-getting-started-index.md).
For a complete composition, use the [Cookbook](doc-examples-index.md).

## Language

- [Syntax Index](doc-reference-syntax.md) lists every statement form and delimiter.
- [Directive Index](doc-reference-directives.md) lists every top-level, procedure, profile,
  test, eval, optimization, and package directive.
- [Package Manifest](doc-reference-package-manifest.md) defines `package.kedi`.

## Runtime and Integration

- [CLI Reference](doc-reference-cli.md) covers invocation, options, program arguments, and
  exit behavior.
- [Environment Variables](doc-reference-environment-variables.md) records Kedi-owned and
  provider-owned configuration.
- [Python API](doc-reference-python-api.md) inventories decorators, contexts, promises,
  runtime helpers, approvals, MCP, and debugging.
- [Capability Matrix](doc-reference-capability-matrix.md) distinguishes framework and harness
  feature support.

## Failure and Coverage

- [Diagnostics and Troubleshooting](doc-reference-diagnostics-and-troubleshooting.md) maps
  symptoms to violated contracts.
- [Semantic Coverage Map](doc-reference-semantic-coverage.md) is the completeness checklist
  for the documentation set.

The reference describes public behavior. Internal modules not exported from
`kedi` may change without preserving compatibility.

## Terminology

- **Template call** is the Kedi-level `>>` operation. It may request structured
  output fields or discard a raw response when no fields are present.
- **Raw invoke** is `[name] << prompt`, which retains the model's complete text
  response as a string.
- **Adapter invocation** is the runtime call through Kedi's `AgentAdapter`
  protocol.
- **Provider request** is the outbound model or harness request made by an
  adapter. One Kedi operation may require multiple provider requests when tools
  or agent loops are involved.
- **Variable initialization** (`=` after a binding) introduces a value in its
  owning scope; **assignment** (`:=`) updates an existing binding.
- **Validation** checks a declared type or result contract. It does not establish
  factual accuracy, permission to perform an action, or security of embedded Python.
