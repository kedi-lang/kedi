# Artifact Safety and Failures

Bounded references do not make source tools, embedded Python, or sensitive payloads safe to execute or disclose.

## Failure Semantics

Artifact failures are explicit and do not silently expose the full payload:

| Error | Meaning |
| --- | --- |
| `ArtifactPolicyError` | Invalid field, unit, store, duration, or bound |
| `ArtifactInputError` | Invalid read/search arguments, incompatible modes, or missing JSON path; also a `ValueError` |
| `ArtifactSerializationError` | Value cannot be represented by the selected store |
| `ArtifactQuotaExceededError` | Count or byte quota would be exceeded |
| `ToolOutputTooLargeError` | Artifacts are disabled and a tool result exceeds 100,000 bytes |
| `ArtifactNotFoundError` | Reference is unknown to the active session |
| `ArtifactAccessError` | Cross-session access or unsafe store path |
| `ArtifactReleasedError` | Payload was explicitly released or is pending release |
| `ArtifactExpiredError` | TTL or idle TTL elapsed |
| `ArtifactStreamError` | Stream kind, chunk, reuse, or transactional consumption failed |
| `ArtifactCodeAccessError` | Code attempted to access a ref outside its allowlist |
| `ArtifactCodeBudgetError` | Code exceeded a host-call, read, materialization, or result bound |
| `ArtifactCodeError` | Sandboxed artifact computation failed |

Quota failure occurs before publishing a reference. Stream failure aborts the
transaction. Artifact Compute failure does not create a derived artifact. None of
these errors include the rejected raw payload in their message or telemetry.

For agent calls, invalid read/search arguments and unknown, expired, or released
references become tool-error responses. The model may correct its arguments under
the adapter's existing retry/turn limits; Kedi does not retry automatically or
increase those limits. Pattern search requires `path=None`, `offset=0`, and
`offset_from="start"`. It cannot be combined with JSON selection or pagination.

Direct Python callers still receive typed exceptions. Storage corruption,
configuration failures and unexpected runtime exceptions are not reclassified
as correctable artifact arguments. Provider harnesses retain their own ordinary
tool-failure reporting behavior.


## Security and Observability

Artifact metadata, summaries, bounded previews, read sizes, lifecycle states,
and Artifact Compute provenance may appear in telemetry. Raw payloads are not attached
to artifact lifecycle spans. Code provenance contains a code hash and source
references, not an unrestricted payload copy.

Artifact storage is not a secret-management boundary. Sensitive data still
requires a suitable file-store root, TTL, approval policy, process isolation,
and telemetry configuration. The `sensitive` marker is metadata for policy and
observability consumers; it does not encrypt the payload.


## Operational Guidance

- Keep artifacts enabled for agentic programs. Disabling them does not permit
  unbounded tool context; it converts oversized results into hard failures.
- Set `threshold` above the expected size of useful reductions so
  `run_artifact_code` can return them inline.
- Use `ArtifactStream` for producer-memory control. Ordinary artifact admission
  controls model context, not the allocation already made inside a tool.
- Prefer `read_artifact` for one bounded fact and `run_artifact_code` for data
  reduction. Paging an entire payload through the model defeats the design.
- Release payloads only after the needed evidence and derived values have been
  obtained. Release frees quota but intentionally leaves history immutable.
- Use the file store when process memory offload or restart persistence matters;
  use the memory store when native opaque Python values are required.

Explicitly setting `enabled: false` removes artifact references and management
tools from that scope. A tool result over the hard inline limit still fails
before model history insertion.
