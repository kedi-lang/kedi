# Investigate an incident through an artifact

Read a longer incident record without carrying the entire tool result into each
model request. The fixture includes plausible distractions, an unsuccessful
rollback, and measured recovery after a separate configuration correction.

## Run

Complete the [shared setup](../README.md#setup), then:

```bash
kedi examples/artifact_investigation/main.kedi
```

The `Finding` should connect eight workers with pool size 40 to a database limit
of 180 connections. Reducing the pool size to 20 restored service. Rolling back
the receipt formatter did not. Supporting timestamps include 09:30, 09:32,
09:42, 09:55, and 10:02 in [incident.md](incident.md).

## What is bounded

The local procedure reads the file. Because it exceeds the **2 KB** threshold,
the model receives an artifact reference with at most **300 preview characters**.
`query_artifacts: enabled` adds lexical retrieval; one read is limited to
**1,800 characters**. The low threshold is intentional for this small example,
not a proposed production default.

The prompt asks the model to locate relevant sections with `query_artifact`
and read surrounding text when necessary. It does not need to rerun the source
tool to reopen its result.
The example does not force a fixed number of retrievals: enough evidence to
distinguish correlation from a supported cause matters more than tool count.

No embeddings or Jev calls are involved. Artifact tools bound what enters the
prompt; they do not make every retrieval relevant or guarantee a correct answer.
The offline test exercises the real source tool and `query_artifact` and checks
that the full source was not returned inline. Live outcomes are documented
separately from those plumbing checks.

The source tool only reads this fixture. `> approval: allow` must be reconsidered
before adding side-effecting tools to this program.
