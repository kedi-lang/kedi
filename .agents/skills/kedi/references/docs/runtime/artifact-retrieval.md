# Reading and Querying Artifacts

A reference means the producer already ran. Inspect that stored result instead of repeating the source tool.

## Management Tools

Artifact-enabled adapters receive four runtime-owned tools. Their schemas and
instructions remain stable across turns.

### `search_artifacts`

```text
search_artifacts(query: str | None = None, limit: int = 20)
```

Searches active reference metadata without opening payloads. Search covers the
reference ID, logical type, media type, summary, and source. Results are newest
first, include status and timestamps, and are capped at 100 items.

Use this when the required reference is not already present in the current
context. Do not search before reading a known reference.

### `read_artifact`

```text
read_artifact(
    ref_id: str,
    max_chars: int = -1,
    offset: int = 0,
    offset_from: "start" | "end" = "start",
    path: str | None = None,
    pattern: str | None = None,
    max_matches: int = 20,
)
```

This is the preferred operation for a known literal, one JSON field, the head
or tail of a value, or another bounded inspection.

- `max_chars=-1` selects `read_max_chars`; it never means unlimited.
- A positive `max_chars` is clamped to `read_max_chars`.
- `offset_from="start"` pages forward from the beginning.
- `offset_from="end"` addresses a tail-relative window without reversing text.
- `path` is an RFC 6901 JSON Pointer into structured JSON content.
- `pattern` performs bounded literal substring search and returns match
  contexts instead of an `ArtifactChunk`.
- Pattern search cannot be combined with `offset`, `offset_from="end"`, or
  `path`.

An `ArtifactChunk` returns `content`, the requested `offset`, an optional
`next_offset`, `complete`, `media_type`, `path`, `offset_from`,
`requested_max_chars`, `applied_max_chars`, `returned_chars`, and an optional
`continuation`. A policy may make `applied_max_chars` smaller than the requested
limit. When `complete` is false, `continuation` contains the exact tool name and
arguments for the next page. The absence of more text must not be inferred from
chunk length alone.

### `query_artifact` (opt-in)

Enable lexical passage retrieval in Kedi source where it is needed:

```kedi
> artifacts:
    query_artifacts: enabled
```

The setting is lexical and source ordered. A nested
`query_artifacts: disabled` removes the tool and its instructions in that scope,
then the outer setting is restored when the scope ends. The default is
`disabled`.

Python callers can establish an inherited adapter default when constructing an
adapter:

```python
from kedi.agent_adapter import PydanticAdapter, LangChainAdapter

adapter = PydanticAdapter(model, query_artifact=True)
adapter = LangChainAdapter(chat_model, query_artifact=True)
```

`ClaudeAdapter` and `CodexAdapter` accept the same option. An explicit Kedi
artifact policy overrides this constructor default. Disabled scopes expose
neither the tool nor its instructions.
Artifacts must also be enabled. Kedi installs the tool in runtime-managed
runs and bound agent surfaces, just like `read_artifact`. A bare adapter call
does not create an artifact session by itself.

```text
query_artifact(ref_id: str, query: str, max_chars: int = -1, top_k: int = 3)
```

Use this when you know the subject or an identifier but not its exact location
in an existing artifact. It searches that artifact's text projection using
Unicode-aware lexical matching and BM25 ranking, with English stemming,
identifier components, and overlapping passages. It makes no model or embedding
calls. This is lexical retrieval: it cannot reliably resolve synonyms or infer
facts absent from the matching text.

- `query` accepts 1-2048 characters and at most 64 distinct searchable terms.
  Search syntax is treated as text, not as SQL or a user-controlled query language.
- `top_k` selects at most 1-10 passages, defaulting to 3.
- `max_chars` limits the **sum of excerpt characters**, not each result.
  `-1` selects the artifact's `read_max_chars`; larger requests are capped.
  Metadata is additional to this content budget.
- Results contain `matches`, `ref_id`, `media_type`, `retrieval`,
  `requested_max_chars`, `applied_max_chars`, `returned_chars`, `total_chars`,
  `searched_passages`, and `exhaustive=false`.
- Each match has the original `content`, zero-based `offset`, exclusive
  `end_offset`, and a relative `score`. Scores are not confidence estimates.
  Matches are ranked by relevance and do not overlap. Use `read_artifact` with
  the returned offsets for surrounding content.

An empty match list does not prove the answer is absent. Retrieval is not an
exhaustive filter or aggregate; use `run_artifact_code` for those operations.
There is no need to query an artifact when its preview already answers the task.

The search scans the stored projection using bounded reads and builds a
temporary SQLite FTS5 index for that call. The index is deleted on completion,
including on failure; repeated queries rebuild it. Python's SQLite build must
include FTS5. File-backed payloads are not loaded as complete Python objects.
Search respects the existing session, release, and expiry lifecycle and records
an `artifact_query` history event without storing the query text in that event.

Python applications can use the same retrieval directly through
`manager.query(ref_id, query, max_chars=-1, top_k=3)`. The constructor flag
controls model-facing tool exposure, not access by trusted Python code.

### `run_artifact_code`

```text
run_artifact_code(code: str, artifact_refs: list[str])
```

Runs bounded Python over an explicit reference allowlist. Use it for filtering,
aggregation, joins, ranking, cross-artifact comparison, and any operation where
returning pages to the model would cost more context than returning the reduced
answer.

The sandbox exposes:

```python
artifact_metadata(ref_id)
read_artifact(ref_id, offset=0, offset_from="start", max_chars=-1)
find_artifact(ref_id, pattern, max_matches=20, context_chars=120)
iter_artifact(ref_id, chunk_chars=4000)
get_artifact(ref_id)
```

`get_artifact` is limited to small values. Large values must be processed with
bounded reads, literal search, or iteration. The final Python expression is the
semantic result; captured stdout is diagnostic output.

Use the provided helpers and ordinary expressions, loops, and comprehensions;
`eval` and `exec` are unavailable. Their errors include a bounded recovery hint,
not a host-Python fallback. Avoid imports: Monty's limited built-in modules are
not the host Python standard library. The execution cannot access host files,
environment variables, the network, models, adapters, tools, or subagents. It can access
only references listed in `artifact_refs`. Host calls, total bytes read,
materialization, stdout, execution time, memory, recursion, result size, result
depth, and result node count are bounded.

The default runtime limits are:

| Limit | Default |
| --- | ---: |
| Duration | 5 seconds |
| Memory | 64 MB |
| Recursion depth | 100 |
| Allowed references | 32 |
| Host calls | 256 |
| Characters per host read | 64,000 |
| Cumulative bytes read | 32 MB |
| `get_artifact` materialization | 256 KB |
| Captured stdout | 16 KB |
| Result size | 8 MB |
| Result depth | 64 |
| Result nodes | 100,000 |

The result is admitted under the active artifact policy. A small reduction is
returned inline. A reduction at or above the threshold becomes a new
`artifact_code_result_N` with provenance linking its source references and a
hash of the executed code. Its `derivation_depth` is one plus the maximum depth
of its accessed parents, with ordinary source artifacts at depth zero. Only
actually accessed references are recorded as parents, not every allowed ref.
The derived artifact remains valid if its source artifacts are later released;
provenance does not keep their payloads alive.

Choose a threshold larger than the expected reduced result. An unnecessarily
low threshold can recursively artifact a useful small reduction and force an
extra model turn to read it.

### `release_artifact`

```text
release_artifact(ref_id: str)
```

Releases payload storage and quota after the agent has consumed all evidence it
will need from that reference. Release is mutating and follows the active
approval policy. It is idempotent: releasing an already released artifact
returns `already_released=true`.

Release does not delete or rewrite prior conversation messages. Later access to
the payload fails with `ArtifactReleasedError`.


## Agent Routing Contract

The runtime instructs artifact-aware agents to follow this decision table:

| Need | Operation |
| --- | --- |
| Known literal, one bounded range, head, tail, or one JSON path | `read_artifact` |
| Locate a literal in one artifact | `read_artifact(pattern=...)` |
| Find passages by keywords or identifiers in one artifact | `query_artifact` (opt-in) |
| Locate an artifact whose reference is unknown | `search_artifacts` |
| Filter, aggregate, rank, join, or compare one or more artifacts | `run_artifact_code` |
| Payload is no longer needed | `release_artifact` |

The agent must not repeat a source tool after receiving its artifact reference.
It must not page an entire large payload through `read_artifact` when a bounded
code reduction can produce the answer. Conversely, it should not invoke the
code sandbox for a single known literal or one small bounded read.

This routing is the source of the context reduction: the model sees compact
references and small semantic results rather than every byte touched by the
runtime.
