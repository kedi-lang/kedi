# Generate, Evaluate, Route

A generative model writes a draft; Jev evaluates it against explicit evidence;
Kedi selects the next operation. Neither the draft nor a high probability is
permission to send a message or perform a financial action.

This example requires `kedi[typesafe]`, `pydantic-ai-slim[openai]` and
credentials for both integrations. It prints a review result; it does not send
email. Model availability is a provider prerequisite, not a compiler guarantee.

```kedi
> adapter: pydantic

@draft_reply(ticket: str, policy: str) -> str:
  > model: openai:gpt-6-luna
  >> Customer request: <ticket>.
  Policy: <policy>.
  A brief reply that does not invent completed actions is [reply].
  = `reply`

@check_reply(reply: str, policy: str) -> bool:
  > model: typesafe/jev-latest
  > settings:
    typesafe_threshold: 0.9
  >> Given policy <policy>, does reply <reply> avoid unsupported commitments?
  It is [supported: bool] that the reply is supported by the policy.
  = `supported`

[ticket] = Please refund my duplicate payment today.
[policy] = A reviewer must verify duplicate charges before approving refunds.
[reply] = `draft_reply(ticket, policy)`
[supported: bool] = `check_reply(reply, policy)`

> if: `supported`:
  = Draft for human review: <reply>
> else:
  = Manual review required; do not send the draft.
```

## Boundaries

The first procedure can generate arbitrary prose. The second produces a boolean
using Jev's strict threshold policy, not a generative text fallback. Unsupported
schemas and provider failures remain errors rather than becoming a negative
decision. The final branch is deterministic because it reads an existing bool.

The program deliberately does not retry until acceptance. A revision loop needs
an explicit attempt budget, fresh evaluation of each revision, and a terminal
manual-review outcome. Such a policy belongs to the application, not to hidden
adapter behavior.

Use [decision capture](../python-api/decisions.md) around an embedding call to
retain the `check_reply` output's evidence after its procedure frame ends.
The caller's deterministic `supported` initialization is not itself a new Jev
evaluation record.
