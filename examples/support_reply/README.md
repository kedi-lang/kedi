# A policy-grounded support reply

Draft a delivery response from a real local tool result instead of a policy
invented by the model. The supplied policy distinguishes opening an investigation
from eligibility for a refund and from actually issuing one.

## Run

Complete the [shared setup](../README.md#setup), then:

```bash
kedi examples/support_reply/main.kedi
```

The typed `Reply` contains `message` and `policy_sections`. A suitable draft asks
for the order number and tracking update, explains the investigation option,
and does not claim a refund has been issued. The cited headings must exist in
[delivery_policy.md](delivery_policy.md).

## Why a profile and a procedure

`read_delivery_policy` is a normal Kedi procedure. Its docstring becomes the tool
description when `> use:` exposes it. The `support` profile groups that tool and
its system instructions; `draft_reply` applies the profile locally. Other
procedures do not acquire those instructions by accident.

`> approval: allow` is deliberately scoped to this trusted, local read-only
example. Do not copy it unchanged into a profile with write, payment, shell, or
other side-effecting tools. No order API, MCP server, filesystem package, or
external business system is required.

The test drives the tool loop, reads the actual file, and checks the final typed
value. A real model still decides how to phrase the response. Nothing sends it.
