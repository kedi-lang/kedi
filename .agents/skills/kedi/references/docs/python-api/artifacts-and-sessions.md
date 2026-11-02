# Artifacts and Sessions

The Python API exposes the same artifact policy as `> artifacts:` and an
explicit session boundary for calls that must share conversation history.

## Configure Artifact Handling

Use a mapping for concise configuration:

```python
import kedi

kedi.configure(
    adapter="pydantic",
    artifacts={
        "enabled": True,
        "query_artifacts": "enabled",
        "threshold": "100kb",
        "ttl": "1h",
        "store": "memory",
    },
)
```

Artifact handling is enabled by default. `artifacts=False` explicitly disables
the inherited policy in `configure()`, `context()`, `query()`, or `bind()`.

For a typed policy object:

```python
from kedi import ArtifactPolicy

policy = ArtifactPolicy.patch(
    {
        "enabled": True,
        "store": "file",
        "path": ".kedi/artifacts",
        "threshold": "64kb",
        "ttl": "30m",
    }
)

kedi.configure(artifacts=policy)
```

Mappings use DSL field names such as `query_artifacts`, `threshold`, `ttl`, `idle_ttl`,
`session_quota`, and `cleanup_interval`. A normalized `ArtifactPolicy` exposes
their internal byte/second forms as `threshold_bytes`, `ttl_seconds`,
`idle_ttl_seconds`, `session_quota_bytes`, and `cleanup_interval_seconds`.
`query_artifacts` accepts only `"enabled"` or `"disabled"` and defaults to
disabled unless an adapter constructor supplies an inherited default.

## Per-Callable and Scoped Overrides

Artifact configuration is accepted by `configure`, `context`, `query`, and
`bind`:

```python
import kedi


@kedi.query(artifacts={"enabled": True, "threshold": "32kb"})
def create_report(topic: str) -> str:
    """kedi
    >> A detailed report about <topic> is [report].
    = `report`
    """
    ...


with kedi.context(artifacts=False):
    report = create_report("typed orchestration")
```

The decorator above explicitly enables artifacts, so it overrides that outer
disabled context for this callable. Omit the decorator's `artifacts=` argument
when the caller should control the policy.

An artifact-backed result still returns as the original annotated Python value.
`ArtifactRef` is model-visible transport metadata, not a replacement return
type for ordinary application code.

## Stateful Sessions

Kedi is stateless unless a `ConversationState` is supplied. `session()` creates
or activates one state and closes its artifact manager on exit:

```python
import kedi

kedi.configure(
    adapter="pydantic",
    artifacts={"enabled": True, "threshold": "100kb"},
)

with kedi.session() as conversation:
    first = create_report("release readiness")
    second = review_report()
```

Use the async form in asynchronous code:

```python
async with kedi.session() as conversation:
    first = await create_report_async("release readiness")
    second = await review_report_async()
```

An existing open `ConversationState` may be supplied when the caller owns its
lifecycle:

```python
from kedi import ConversationState

state = ConversationState()

with kedi.session(state):
    first = create_report("runtime safety")
    second = review_report()
```

Exiting `session()` closes its artifact manager, so that state cannot then be
resumed. The CLI does not create persistent conversation state implicitly.

For direct decorator control, `query(..., conversation=state)` and
`bind(..., conversation=state)` use the same state without changing unrelated
configuration contexts.

## Public Artifact Types

These types are exported from `kedi`:

| Type | Role |
| --- | --- |
| `ArtifactPolicy` | Validated policy and lexical overlay |
| `ArtifactRef[T]` | Compact metadata sent across model context |
| `ArtifactChunk` | Bounded result from an artifact read |
| `ArtifactSearchResult` | Metadata-only search result |
| `ArtifactReleaseResult` | Payload release status |
| `ArtifactHandle[T]` | Internal lazy native-value handle |
| `ConversationState` | Portable history and artifact ownership |

Application code normally configures `ArtifactPolicy` and `ConversationState`.
The remaining types are useful for custom adapters, tooling, and diagnostics.

See [Tool Artifacts](../runtime/tool-artifacts.md) for storage guarantees,
management tools, expiry, quotas, and cache-epoch history semantics.
