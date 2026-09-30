# Meeting notes to actions

Extract an action list without turning tentative suggestions into commitments.
The bundled [notes](notes.md) include completed work, an explicit deadline,
an owner with no deadline, and an action without either.

## Run

Complete the [shared setup](../README.md#setup), then from the repository root:

```bash
kedi examples/meeting_actions/main.kedi
```

The result is a native `list[Action]`, not text containing JSON. It should contain:

- Sam's duplicate-payment regression test, due `2026-09-25`.
- Alex's rollback-runbook update, with a null deadline.
- The production rehearsal, with a null owner and deadline.

The completed staging migration and proposed dashboard redesign must be absent.
Wording of the task descriptions may vary.

## Read the program

`~Action` defines the result type. `Annotated` describes what belongs in `task`;
`Constraints(min_length=1)` rejects an empty task. Optional fields are genuinely
nullable instead of being padded with fabricated names or dates.

One `>>` block requests all actions. Continuation lines are part of that same
prompt. The `=` returns the typed list; it does not issue a second model call.
Python only reads the input file. Replace `notes.md` to try another meeting.

Schema validation can reject malformed values; it cannot prove that an action
was actually agreed. The regression test verifies the schema and native return
types, and the fixture gives a small human-checkable semantic acceptance test.
