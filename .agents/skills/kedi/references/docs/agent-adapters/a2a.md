# A2A Cloud Agents

Kedi can serve an exported agent profile as a remote
[A2A](https://a2a-protocol.org/) agent and can use any compatible A2A agent as
an `agent-harness`. The coordinator may stay local while repository work,
browser automation, or other expensive execution runs in a remote sandbox.

Install the optional protocol and server dependencies:

```bash
uv add 'kedi[a2a]'
```

For persistent task and session state, install `kedi[a2a-persistence]` instead.

Kedi uses the official `a2a-sdk` Python package. A2A is a transport boundary,
not a model provider: the serving process owns its model, tools, sandbox, and
approval policy.

## Serve a Kedi Profile

Only an explicitly exported profile can become the server entry:

```kedi
> profile: researcher:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > instructions:
        Inspect the available evidence before answering.
        State uncertainty instead of inventing missing facts.

> export:
    researcher
```

The entry must resolve to a local framework or harness capable of executing a
Kedi child profile. It cannot use `a2a` as its own primary backend. A model must
be selected by the profile, `--model`, or `KEDI_A2A_MODEL`; startup fails before
binding a socket when the executable configuration is incomplete.

Serve it with authentication:

```bash
export RESEARCH_AGENT_TOKEN='replace-with-a-secret'

kedi a2a serve research-agent.kedi \
  --entry researcher \
  --host 0.0.0.0 \
  --port 8000 \
  --public-url https://agents.example.com \
  --auth bearer \
  --secret-env RESEARCH_AGENT_TOKEN
```

`--public-url` is required when binding beyond loopback. Terminate TLS at the
service or a trusted reverse proxy, and ensure the public URL reaches the same
process. The default JSON-RPC route is `/a2a`; the protected Agent Card is
published through the official SDK routes.

Supported server authentication modes are Basic, Bearer, and header API key:

```bash
kedi a2a serve research-agent.kedi \
  --entry researcher \
  --auth api-key \
  --api-key-header X-Research-Key \
  --secret-file /run/secrets/research-agent
```

Credentials may come from `--secret-env`, `--secret-file`, or
`KEDI_A2A_SECRET`. An interactive terminal may prompt for the secret. Literal
secrets should not be placed in source, shell history, or process arguments.
Use `--allow-unauthenticated` only for an explicitly loopback-bound development
server.

### Persist Tasks and Sessions

Persistence is opt-in. Give one server process a dedicated local directory on a
persistent volume:

```bash
uv add 'kedi[a2a-persistence]'

kedi a2a serve research-agent.kedi \
  --entry researcher \
  --auth bearer \
  --secret-env RESEARCH_AGENT_TOKEN \
  --state-dir /var/lib/kedi/research-agent
```

`KEDI_A2A_STATE_DIR` is the environment equivalent of `--state-dir`. The
directory must be private to the serving user; Kedi will create it with
restricted permissions when it does not exist. A process lock prevents a second
server from opening the same directory. The directory is bound to the served
program, its Kedi imports, entry, adapter, model, and history configuration.
Changing these requires a new state directory; Kedi will not silently discard
or reinterpret existing state.

Completed A2A tasks, owner-scoped results, conversation checkpoints, and
referenced artifact payloads survive a process restart. Pydantic AI and
LangChain profiles with history enabled restore their native message
history, not a reconstructed text transcript. Other adapters can persist
stateless task results, but history-enabled serving with an unsupported adapter
fails at startup. Without `--state-dir`, the existing in-memory behavior remains.

Artifact-producing tools and `read_artifact`/`query_artifact` use the same
session-owned store. Tools are rebound to that store after restoration. They do
not share the compiled program's artifact storage with other remote sessions.

A task that was `submitted` or `working` when the server stopped becomes
`failed` with `server_interrupted` metadata on the next startup. It is not
replayed automatically: a model call or tool may already have had external
effects. A previously canceled task remains canceled. The last committed
session turn is the recovery point; uncommitted turns are not added to history.
Use `GetTask` to inspect the old task, then start a new attempt deliberately.

Terminal task records cannot be overwritten by late completion, failure, or
cancellation events. Task completion and the session checkpoint are committed
in one database transaction. Cancellation and shutdown wait for in-flight
storage work before releasing session resources or the directory lock; this
does not make external tool effects transactional.

Failed checkpoint writes also restore the SDK's in-memory task view. A transient
storage error can then be recorded as `failed` instead of leaving the client
polling a stale `working` task. If storage remains unwritable, Kedi cannot promise
a durable failure update: the operation errors, and the last committed task is
reconciled as interrupted after storage recovers and the server restarts. Neither
case automatically repeats model calls or external tool effects.

This mode is single-process and requires a local filesystem with OS file locks.
It does not preserve provider sockets, runtime globals, tool-created files,
remote sandbox filesystems, or effects in external services. Back up the state
directory and its artifact files together. Persistent artifact payloads must
use Kedi's safe file codecs; an unsupported value cannot be checkpointed as a
usable reference. Each session checkpoint is limited to 64 MiB; session,
retained-task, and artifact quotas still apply. Released and expired artifact
payloads are compacted after a successful commit while their unavailable status
is retained. Artifact expiry and idle deadlines survive restart; reopening the
server does not extend an artifact's lifetime. Retention evicts only terminal
tasks, including tasks marked interrupted during restart, and never running
tasks. The state directory has no global byte quota, so place it on a
volume with an explicit capacity limit and free-space monitoring. Disk-full or
corrupt referenced payloads fail closed rather than reporting a successful task.
Basic-auth
password rotation retains the username's state, while rotating a Bearer/API key
changes the principal and does not transfer old records.

## Connect from Kedi

The connection belongs to the selected agent profile, so lexical profile
scoping also scopes the endpoint and credential reference:

```kedi
> profile: cloud_researcher:
    > agent: a2a:
        endpoint: https://agents.example.com
        auth:
            scheme: bearer
            token_env: RESEARCH_AGENT_TOKEN
        request_timeout: 30
        task_timeout: 21600
        poll_interval: 0.5

@research(question: str) -> str:
    > use: cloud_researcher
    [answer] << Investigate <question> and return a concise evidence-based answer.
    = `answer`
```

Pass the discovery base URL as `endpoint`, not the JSON-RPC route from the
Agent Card. Non-loopback endpoints require HTTPS. Embedded credentials, query
strings, and fragments are rejected.

Client authentication fields are:

| Scheme | Required fields | Optional fields |
| --- | --- | --- |
| `basic` | `username`, `password_env` or `password_file` | none |
| `bearer` | `token_env` or `token_file` | none |
| `api_key` | `api_key_env` or `api_key_file` | `header` (default `X-API-Key`) |

The generic aliases `credential_env` and `credential_file` are accepted for all
three schemes. Exactly one environment, file, or Python provider source is
required; combining an alias with a scheme-specific source is rejected.
Kedi snapshots the reference, never the resolved credential value.

## Structured Results

A generic A2A peer is a text agent, so it supports raw invocation such as
`[answer] << ...`. Typed template filling is available only when the Agent Card
advertises Kedi's versioned structured-output extension:

```kedi
> profile: cloud_researcher:
    > agent: a2a:
        endpoint: https://agents.example.com
        auth:
            scheme: api_key
            api_key_env: RESEARCH_AGENT_KEY

@inspect(topic: str) -> tuple[str, list[str]]:
    > use: cloud_researcher
    >> The evidence about <topic> supports [summary: str].
    >> The material risks are [risks: list[str]].
    = `(summary, risks)`
```

Kedi sends the JSON Schema through the negotiated extension and validates the
returned data before assigning any capture. It does not ask a second model to
repair a generic text response. See the
[structured-output v1 contract](../a2a/extensions/structured-output/v1.md).

## Remote Subagents

An A2A profile can be a child of a local Pydantic AI, LangChain, Codex, or other
subagent-capable coordinator:

```kedi
> profile: cloud_worker:
    > agent: a2a:
        endpoint: https://agents.example.com
        auth:
            scheme: bearer
            token_env: RESEARCH_AGENT_TOKEN

> profile: coordinator:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > subagent: cloud_worker

@coordinate(question: str) -> str:
    > use: coordinator
    [answer] << Delegate repository inspection when useful, then answer <question>.
    = `answer`
```

Kedi sends only the task text and, when requested, the final JSON Schema. Local
tools, MCP servers, model settings, effort, CodeMode, approval handlers, and
replayable model history are not projected into the remote process. Configure
those concerns in the served agent.

## Python API

```python
from pydantic import BaseModel

from kedi.agent_adapter import A2AAdapter
from kedi.agent_profile import A2AAuthConfig, A2AConnection


class Finding(BaseModel):
    summary: str
    risks: list[str]


connection = A2AConnection(
    endpoint="https://agents.example.com",
    auth=A2AAuthConfig(
        scheme="bearer",
        secret_env="RESEARCH_AGENT_TOKEN",
    ),
    task_timeout=21_600,
)
agent = A2AAdapter(connection=connection)

text = agent.invoke_sync(prompt="Inspect the release candidate.")
finding = agent.produce_sync(
    "The release evidence supports [summary], with risks [risks].",
    output_type=Finding,
)
```

`get_task()` reads a known task without resending it, `cancel_task()` requests
cancellation, and `continue_task()` resumes an `input_required` or
`auth_required` task. Synchronous variants use the `_sync` suffix. Lifecycle
errors retain known `task_id`, `context_id`, and state coordinates.

## Recovery and Timeouts

- Stream failure after Kedi learns the task ID falls back to status polling.
- An ambiguous send with no task ID is never resent automatically; duplication
  would be less safe than surfacing `A2AAmbiguousSendError`.
- `task_timeout` bounds local observation. It raises `A2ATaskTimeoutError` but
  does not cancel remote work. Use the task ID to inspect or cancel it later.
- Conversation continuation stores only remote task/context coordinates. A2A
  does not provide full replayable model history to Kedi.

## Security and Deployment Limits

- A remote endpoint must use HTTPS; plain HTTP is limited to loopback hosts.
- Agent Card authentication requirements must match the configured client
  credential, and advertised JSON-RPC interfaces must remain same-origin.
- Request body, input, schema, result, pending-task, session, concurrency, and
  retained-task counts are bounded. Tune the corresponding `kedi a2a serve`
  options for the deployment.
- Tasks and sessions are scoped by authenticated principal. Queued and running
  calls keep their in-memory sessions alive; only idle sessions can be evicted.
  Calls sharing a context run sequentially without consuming concurrency slots
  while waiting for that context. Without `--state-dir`, restart loses that
  state. With it, completed state is restored, but live execution is not.
  Neither mode supports multiple workers sharing one state directory.
- Shutdown stops SDK tasks and drains active and queued invocations before
  closing session and runtime resources.
- A2A is not a sandbox. Run the server with filesystem, process, network, and
  secret permissions appropriate for the exported profile.
- OAuth/OIDC, mTLS identity, push notifications, remote file transfer, webhooks,
  and cross-process task ownership are not implemented by this release.

Use an external gateway for rate limiting and production identity controls.
The built-in server provides bounded capacity and credential authentication,
not a complete public-edge security product.
