# Complete offline documentation catalog

Read one relevant page at a time. This is the complete 154-page supplied documentation snapshot, with local cross-links and per-snippet parse status. The task guides in SKILL.md distill the rules; these pages retain exact settings, APIs, edge cases, and examples. The documentation remains ground truth; version-specific conflicts are recorded in [pitfalls](pitfalls.md).

All Kedi snippets have a parse annotation. They are not all standalone programs. References to an external service or optional dependency are not evidence that it was tested.

## Contents

- [a2a](#a2a)
- [agent-adapters](#agent-adapters)
- [agentic-engineering](#agentic-engineering)
- [core-language](#core-language)
- [evals-and-optimization](#evals-and-optimization)
- [examples](#examples)
- [getting-started](#getting-started)
- [overview](#overview)
- [modules-and-packaging](#modules-and-packaging)
- [python-api](#python-api)
- [python-interop](#python-interop)
- [reference](#reference)
- [runtime](#runtime)
- [tooling](#tooling)

## a2a

| Topic | Reference |
| --- | --- |
| Kedi A2A Structured Output Extension v1 | [a2a/extensions/structured-output/v1.md](doc-a2a-extensions-structured-output-v1.md) |

## agent-adapters

| Topic | Reference |
| --- | --- |
| A2A Cloud Agents | [agent-adapters/a2a.md](doc-agent-adapters-a2a.md) |
| ACP Agents | [agent-adapters/acp.md](doc-agent-adapters-acp.md) |
| Artifact Integration Contract | [agent-adapters/artifact-contract.md](doc-agent-adapters-artifact-contract.md) |
| Claude | [agent-adapters/claude.md](doc-agent-adapters-claude.md) |
| Codex-Authenticated Models | [agent-adapters/codex-models.md](doc-agent-adapters-codex-models.md) |
| Codex | [agent-adapters/codex.md](doc-agent-adapters-codex.md) |
| Custom Adapters | [agent-adapters/custom-adapters.md](doc-agent-adapters-custom-adapters.md) |
| DSPy | [agent-adapters/dspy.md](doc-agent-adapters-dspy.md) |
| Models and Integrations { #agent-adapters } | [agent-adapters/index.md](doc-agent-adapters-index.md) |
| Jev Claims and Thresholds | [agent-adapters/jev-claims.md](doc-agent-adapters-jev-claims.md) |
| Jev Criteria and Output Types | [agent-adapters/jev-criteria.md](doc-agent-adapters-jev-criteria.md) |
| Jev Decision Evidence | [agent-adapters/jev-evidence.md](doc-agent-adapters-jev-evidence.md) |
| Finite Extraction with Jev | [agent-adapters/jev-extraction.md](doc-agent-adapters-jev-extraction.md) |
| Jev Tool Selection | [agent-adapters/jev-tools.md](doc-agent-adapters-jev-tools.md) |
| Generate, Evaluate, Route | [agent-adapters/jev-workflow.md](doc-agent-adapters-jev-workflow.md) |
| LangChain | [agent-adapters/langchain.md](doc-agent-adapters-langchain.md) |
| Local Laya | [agent-adapters/laya.md](doc-agent-adapters-laya.md) |
| Pydantic AI | [agent-adapters/pydantic-ai.md](doc-agent-adapters-pydantic-ai.md) |
| TypeSafe Jev | [agent-adapters/typesafe.md](doc-agent-adapters-typesafe.md) |

## agentic-engineering

| Topic | Reference |
| --- | --- |
| Approvals and Sensitive Operations | [agentic-engineering/approvals.md](doc-agentic-engineering-approvals.md) |
| Backend Selection | [agentic-engineering/backend-selection.md](doc-agentic-engineering-backend-selection.md) |
| Capabilities | [agentic-engineering/capabilities.md](doc-agentic-engineering-capabilities.md) |
| CodeMode Sandbox and Recovery | [agentic-engineering/codemode-sandbox.md](doc-agentic-engineering-codemode-sandbox.md) |
| CodeMode | [agentic-engineering/codemode.md](doc-agentic-engineering-codemode.md) |
| Dynamic Workflows | [agentic-engineering/dynamic-workflows.md](doc-agentic-engineering-dynamic-workflows.md) |
| Agent Lifecycle Hooks | [agentic-engineering/hooks.md](doc-agentic-engineering-hooks.md) |
| Agents and Orchestration { #agentic-engineering } | [agentic-engineering/index.md](doc-agentic-engineering-index.md) |
| Instructions and Settings | [agentic-engineering/instructions-and-settings.md](doc-agentic-engineering-instructions-and-settings.md) |
| MCP Servers | [agentic-engineering/mcp.md](doc-agentic-engineering-mcp.md) |
| Models and Reasoning | [agentic-engineering/models-and-reasoning.md](doc-agentic-engineering-models-and-reasoning.md) |
| Profiles | [agentic-engineering/profiles.md](doc-agentic-engineering-profiles.md) |
| Reviewed Evidence Workflow | [agentic-engineering/reviewed-evidence.md](doc-agentic-engineering-reviewed-evidence.md) |
| Agent Scope and Adapter Requirements {#scoping-and-capabilities} | [agentic-engineering/scoping-and-capabilities.md](doc-agentic-engineering-scoping-and-capabilities.md) |
| Skills | [agentic-engineering/skills.md](doc-agentic-engineering-skills.md) |
| Stream Events | [agentic-engineering/stream-events.md](doc-agentic-engineering-stream-events.md) |
| Continuations and Persistence | [agentic-engineering/subagent-continuations.md](doc-agentic-engineering-subagent-continuations.md) |
| Foreground and Background Runs | [agentic-engineering/subagent-lifecycle.md](doc-agentic-engineering-subagent-lifecycle.md) |
| Limits, Isolation and Safety | [agentic-engineering/subagent-limits.md](doc-agentic-engineering-subagent-limits.md) |
| Task Input and Redirection | [agentic-engineering/subagent-messaging.md](doc-agentic-engineering-subagent-messaging.md) |
| Concurrent Result Processing | [agentic-engineering/subagent-processing.md](doc-agentic-engineering-subagent-processing.md) |
| Embedding Subagent Programs in Python | [agentic-engineering/subagent-python.md](doc-agentic-engineering-subagent-python.md) |
| Typed Child Results | [agentic-engineering/subagent-results.md](doc-agentic-engineering-subagent-results.md) |
| Native Tasks and Await | [agentic-engineering/subagent-tasks.md](doc-agentic-engineering-subagent-tasks.md) |
| Subagents | [agentic-engineering/subagents.md](doc-agentic-engineering-subagents.md) |
| Tool Reasons and LLM Approval | [agentic-engineering/tool-reasons.md](doc-agentic-engineering-tool-reasons.md) |
| Tools and `> use:` | [agentic-engineering/tools-and-use.md](doc-agentic-engineering-tools-and-use.md) |

## core-language

| Topic | Reference |
| --- | --- |
| Control Flow | [core-language/control-flow.md](doc-core-language-control-flow.md) |
| Language Reference { #core-language } | [core-language/index.md](doc-core-language-index.md) |
| Loops and Map Continuations | [core-language/loops-and-map.md](doc-core-language-loops-and-map.md) |
| Multiline Syntax and Escaping | [core-language/multiline-and-escaping.md](doc-core-language-multiline-and-escaping.md) |
| Outputs, Initialization, and Assignment | [core-language/outputs-and-assignments.md](doc-core-language-outputs-and-assignments.md) |
| Parameters and Returns | [core-language/parameters-and-returns.md](doc-core-language-parameters-and-returns.md) |
| Procedures | [core-language/procedures.md](doc-core-language-procedures.md) |
| Scopes and Binding Lifetime | [core-language/scopes-and-bindings.md](doc-core-language-scopes-and-bindings.md) |
| Source Structure | [core-language/source-structure.md](doc-core-language-source-structure.md) |
| Substitutions and Calls | [core-language/substitutions-and-calls.md](doc-core-language-substitutions-and-calls.md) |
| Templates and Invokes | [core-language/templates-and-invokes.md](doc-core-language-templates-and-invokes.md) |
| Types | [core-language/types.md](doc-core-language-types.md) |

## evals-and-optimization

| Topic | Reference |
| --- | --- |
| AI-Generated Procedures | [evals-and-optimization/ai-generated-procedures.md](doc-evals-and-optimization-ai-generated-procedures.md) |
| Datasets and Metrics | [evals-and-optimization/datasets-and-metrics.md](doc-evals-and-optimization-datasets-and-metrics.md) |
| Evaluation Suites | [evals-and-optimization/evaluation-suites.md](doc-evals-and-optimization-evaluation-suites.md) |
| GEPA Optimization | [evals-and-optimization/gepa.md](doc-evals-and-optimization-gepa.md) |
| Testing and Optimization { #evals-and-optimization } | [evals-and-optimization/index.md](doc-evals-and-optimization-index.md) |
| Prompt Optimization | [evals-and-optimization/prompt-optimization.md](doc-evals-and-optimization-prompt-optimization.md) |
| Reproducible Comparisons | [evals-and-optimization/reproducibility.md](doc-evals-and-optimization-reproducibility.md) |
| Test Blocks | [evals-and-optimization/test-blocks.md](doc-evals-and-optimization-test-blocks.md) |
| A Complete Validation Workflow | [evals-and-optimization/validation-workflow.md](doc-evals-and-optimization-validation-workflow.md) |

## examples

| Topic | Reference |
| --- | --- |
| Agent Delegation | [examples/agent-delegation.md](doc-examples-agent-delegation.md) |
| Complete Program | [examples/complete-program.md](doc-examples-complete-program.md) |
| Evaluation and Optimization | [examples/evaluation-and-optimization.md](doc-examples-evaluation-and-optimization.md) |
| Cookbook { #examples } | [examples/index.md](doc-examples-index.md) |
| Modules and Packaging | [examples/modules-and-packaging.md](doc-examples-modules-and-packaging.md) |
| Structured Extraction | [examples/structured-extraction.md](doc-examples-structured-extraction.md) |
| Tools and Approvals | [examples/tools-and-approvals.md](doc-examples-tools-and-approvals.md) |

## getting-started

| Topic | Reference |
| --- | --- |
| Your First Kedi Program | [getting-started/first-program.md](doc-getting-started-first-program.md) |
| Learn Kedi { #start-with-kedi } | [getting-started/index.md](doc-getting-started-index.md) |
| Installation | [getting-started/installation.md](doc-getting-started-installation.md) |
| Projects and Execution | [getting-started/projects-and-execution.md](doc-getting-started-projects-and-execution.md) |

## overview

| Topic | Reference |
| --- | --- |
| The Kedi language { #kedi-programming-language } | [index.md](doc-index.md) |

## modules-and-packaging

| Topic | Reference |
| --- | --- |
| Built-In Modules | [modules-and-packaging/built-in-modules.md](doc-modules-and-packaging-built-in-modules.md) |
| Example Modules | [modules-and-packaging/example-modules.md](doc-modules-and-packaging-example-modules.md) |
| Filesystem Module | [modules-and-packaging/filesystem.md](doc-modules-and-packaging-filesystem.md) |
| Imports and Exports | [modules-and-packaging/imports-and-exports.md](doc-modules-and-packaging-imports-and-exports.md) |
| Modules and Packages { #modules-and-packaging } | [modules-and-packaging/index.md](doc-modules-and-packaging-index.md) |
| Installation and Registries | [modules-and-packaging/installation-and-registries.md](doc-modules-and-packaging-installation-and-registries.md) |
| Build a Local Package | [modules-and-packaging/local-package.md](doc-modules-and-packaging-local-package.md) |
| Modules | [modules-and-packaging/modules.md](doc-modules-and-packaging-modules.md) |
| Package Manifests | [modules-and-packaging/package-manifests.md](doc-modules-and-packaging-package-manifests.md) |

## python-api

| Topic | Reference |
| --- | --- |
| Approvals, MCP, and Skills | [python-api/approvals-mcp-and-skills.md](doc-python-api-approvals-mcp-and-skills.md) |
| Artifacts and Sessions | [python-api/artifacts-and-sessions.md](doc-python-api-artifacts-and-sessions.md) |
| `@kedi.bind` | [python-api/bind.md](doc-python-api-bind.md) |
| Python Cache Control | [python-api/cache-control.md](doc-python-api-cache-control.md) |
| Caching, Runtime, and Executors | [python-api/caching-runtime-and-executors.md](doc-python-api-caching-runtime-and-executors.md) |
| Capability Definitions | [python-api/capabilities.md](doc-python-api-capabilities.md) |
| Concurrency and Promises | [python-api/concurrency-and-promises.md](doc-python-api-concurrency-and-promises.md) |
| Configuration and Context | [python-api/configuration-and-context.md](doc-python-api-configuration-and-context.md) |
| Decision Evidence | [python-api/decisions.md](doc-python-api-decisions.md) |
| Embedding and Resource Ownership | [python-api/embedding.md](doc-python-api-embedding.md) |
| Executors and Debug Exporters | [python-api/executors.md](doc-python-api-executors.md) |
| Hooks from Python | [python-api/hooks.md](doc-python-api-hooks.md) |
| Python API | [python-api/index.md](doc-python-api-index.md) |
| MCP from Python | [python-api/mcp.md](doc-python-api-mcp.md) |
| Public Export Index | [python-api/public-exports.md](doc-python-api-public-exports.md) |
| Public Parameters | [python-api/public-parameters.md](doc-python-api-public-parameters.md) |
| `@kedi.query` | [python-api/query.md](doc-python-api-query.md) |
| Skills from Python | [python-api/skills.md](doc-python-api-skills.md) |
| Types and Tools | [python-api/types-and-tools.md](doc-python-api-types-and-tools.md) |

## python-interop

| Topic | Reference |
| --- | --- |
| Python Interop | [python-interop/index.md](doc-python-interop-index.md) |
| Inline Python Expressions | [python-interop/inline-expressions.md](doc-python-interop-inline-expressions.md) |
| Prelude, Globals, and Scope | [python-interop/prelude-globals-and-scope.md](doc-python-interop-prelude-globals-and-scope.md) |
| Python Blocks | [python-interop/python-blocks.md](doc-python-interop-python-blocks.md) |

## reference

| Topic | Reference |
| --- | --- |
| Capability Matrix | [reference/capability-matrix.md](doc-reference-capability-matrix.md) |
| CLI Reference | [reference/cli.md](doc-reference-cli.md) |
| Diagnostics and Troubleshooting | [reference/diagnostics-and-troubleshooting.md](doc-reference-diagnostics-and-troubleshooting.md) |
| Directive Index | [reference/directives.md](doc-reference-directives.md) |
| Environment Variables | [reference/environment-variables.md](doc-reference-environment-variables.md) |
| Find a Reference { #reference } | [reference/index.md](doc-reference-index.md) |
| Package Manifest Reference | [reference/package-manifest.md](doc-reference-package-manifest.md) |
| Python API Reference | [reference/python-api.md](doc-reference-python-api.md) |
| Semantic Coverage Map | [reference/semantic-coverage.md](doc-reference-semantic-coverage.md) |
| Syntax Index | [reference/syntax.md](doc-reference-syntax.md) |

## runtime

| Topic | Reference |
| --- | --- |
| Artifact Storage and Lifetime | [runtime/artifact-lifecycle.md](doc-runtime-artifact-lifecycle.md) |
| Artifact Admission and Configuration | [runtime/artifact-policy.md](doc-runtime-artifact-policy.md) |
| Reducing Artifact Data | [runtime/artifact-reduction.md](doc-runtime-artifact-reduction.md) |
| Reading and Querying Artifacts | [runtime/artifact-retrieval.md](doc-runtime-artifact-retrieval.md) |
| Artifact Safety and Failures | [runtime/artifact-safety.md](doc-runtime-artifact-safety.md) |
| Caching | [runtime/caching.md](doc-runtime-caching.md) |
| Conversation Compaction | [runtime/compaction.md](doc-runtime-compaction.md) |
| Concurrency | [runtime/concurrency.md](doc-runtime-concurrency.md) |
| Errors, Frames, and Tracebacks | [runtime/errors-and-debugging.md](doc-runtime-errors-and-debugging.md) |
| Execution and Dataflow | [runtime/execution-and-dataflow.md](doc-runtime-execution-and-dataflow.md) |
| Conversation History | [runtime/history.md](doc-runtime-history.md) |
| Context and Runtime { #runtime } | [runtime/index.md](doc-runtime-index.md) |
| Interactive Execution | [runtime/interactive-execution.md](doc-runtime-interactive-execution.md) |
| Provider Prefix Caching | [runtime/prefix-cache.md](doc-runtime-prefix-cache.md) |
| Run Budgets | [runtime/run-budgets.md](doc-runtime-run-budgets.md) |
| Telemetry | [runtime/telemetry.md](doc-runtime-telemetry.md) |
| Artifacts { #tool-artifacts } | [runtime/tool-artifacts.md](doc-runtime-tool-artifacts.md) |

## tooling

| Topic | Reference |
| --- | --- |
| Browser and Playground | [tooling/browser-and-playground.md](doc-tooling-browser-and-playground.md) |
| Debugger and Inspector | [tooling/debugger.md](doc-tooling-debugger.md) |
| Tools and Environments { #tooling } | [tooling/index.md](doc-tooling-index.md) |
| Language Server | [tooling/language-server.md](doc-tooling-language-server.md) |
| Notebook Files and Recovery | [tooling/notebook-files.md](doc-tooling-notebook-files.md) |
| Notebook Runtimes | [tooling/notebook-runtimes.md](doc-tooling-notebook-runtimes.md) |
| Local Notebook | [tooling/notebook.md](doc-tooling-notebook.md) |
| Package Commands | [tooling/package-commands.md](doc-tooling-package-commands.md) |
| Terminal REPL | [tooling/repl.md](doc-tooling-repl.md) |
| Run and Parse Commands | [tooling/run-and-parse.md](doc-tooling-run-and-parse.md) |
| Terminal-Bench 2.1 Results | [tooling/terminal-bench-results.md](doc-tooling-terminal-bench-results.md) |
| Terminal-Bench 2.1 | [tooling/terminal-bench.md](doc-tooling-terminal-bench.md) |
| Test, Eval, and Optimize Commands | [tooling/test-eval-and-optimize.md](doc-tooling-test-eval-and-optimize.md) |
| VS Code | [tooling/vscode.md](doc-tooling-vscode.md) |
| Zed | [tooling/zed.md](doc-tooling-zed.md) |
