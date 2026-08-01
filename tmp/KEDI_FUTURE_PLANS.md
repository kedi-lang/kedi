# Kedi Future Plans

This file records tool-artifact work that is intentionally outside the current
PR. The current implementation already provides transactional memory and file
stores, explicit sync/async `ArtifactStream` transport, bounded reads and
search, CodeMode processing, quotas, lifecycle management, telemetry, and
deterministic concurrency and crash-recovery coverage.

## Future Artifact Producers

Any future Kedi-owned producer that naturally emits large output should return
`ArtifactStream` at the producer boundary. Likely examples include web fetches,
downloads, recursive repository search, shell transcript capture, and trace or
report exports. Existing filesystem reads and skill reads already follow this
rule.

Wrapping a fully materialized value in `ArtifactStream` after its producer
returns does not reduce peak producer memory and must not be presented as an
optimization. User-defined tools receive producer-memory savings only when
they explicitly return an `ArtifactStream`; ordinary values still receive the
central output-admission and artifactization guarantees.

## Distributed Artifact Storage

A distributed or object-backed store is a separate architecture from the
current single-process memory and local-file stores. It requires:

- an S3, blob, or database-backed `ArtifactStore` implementation;
- atomic ref allocation and quota accounting across workers;
- per-session ownership leases with renewal and crash reclamation;
- safe same-session multi-process reads, writes, release, and cleanup;
- tenant authorization independent of predictable artifact reference IDs;
- encryption and key-management policy for persisted sensitive artifacts;
- transactional metadata/payload publication and object-store fault recovery.

The local file store assumes that one process owns a restored session. It now
removes interrupted temporary files and uncommitted payload/index files during
restore while protecting writers active in that process. It must not claim
same-session multi-process safety until the lease protocol above exists.

## Deployment-Level Soak and Fault Testing

Repository tests cover bounded concurrent independent sessions, transactional
failures, quota rollback, expiry, release during reads, corrupt metadata, and
restart restoration. Production infrastructure should later add:

- hours-long mixed read/write/release/cleanup soak runs;
- forced process termination at each file and object-store commit boundary;
- multi-host contention against one session after leases are implemented;
- object-store latency, partial-write, stale-read, and outage fault injection;
- sustained telemetry cardinality and cleanup-lag monitoring under real load.

These tests require deployment infrastructure and should not make the current
unit suite slow or nondeterministic.
