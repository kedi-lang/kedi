<span id="dspy"></span>

# Archived DSPy Runtime Adapter

The DSPy runtime adapter is no longer built into Kedi. Its implementation and
adapter-specific tests are preserved in the unmaintained
[kedi-lang/dspy-adapter](https://github.com/kedi-lang/dspy-adapter) repository.

Use Pydantic AI or LangChain for maintained framework integration. The
`dspy` runtime shortname is no longer accepted by `> adapter:`, Python
`adapter=`, or the CLI.

<span id="installation"></span>
<span id="language-models"></span>
<span id="signatures-and-structured-outputs"></span>
<span id="react-tool-calls"></span>
<span id="blocking-subagents"></span>
<span id="reasoning-effort-and-settings"></span>
<span id="stdio-mcp"></span>
<span id="current-capability-limits"></span>

## Existing DSPy Applications

The external package is installed from GitHub and imported separately:

```bash
python -m pip install "git+https://github.com/kedi-lang/dspy-adapter.git"
```

It exports `kedi_dspy_adapter.DSPyAdapter`. Pass an adapter object through
`kedi.context(adapter=...)` or `compile_program(..., adapter=...)` rather than
using the removed built-in shortname. See the external repository for its
tested Python example, compatibility requirements, and original limitations.

DSPy-only `> settings:` keys such as `model_type`, `cache`, `num_retries`, and
`provider` are no longer accepted in Kedi source. Shared settings such as
`temperature` and `max_tokens` remain available on the backends that support them.

<span id="gepa-integration"></span>

## GEPA Is Unchanged

`kedi[dspy]` still installs DSPy for [GEPA optimization](../evals-and-optimization/gepa.md).
GEPA's DSPy modules, optimizer LM, and reflection LM remain in Kedi and do not
depend on the external runtime adapter. Calls outside an optimized span use
the selected runtime `--adapter` and `--adapter-model`.
