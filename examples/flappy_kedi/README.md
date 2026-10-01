# Flappy Kedi

A pixel cat, a Bosphorus skyline, and two possible movements: **flap** or
**coast**. Play yourself, or let a Kedi program ask Jev or local Laya for each decision.

![The pixel cat flying past Istanbul rooftops](flappy_kedi/assets/preview.png)

This is a small working game, not a model benchmark. Python handles graphics
and physics; Kedi owns the pilot's typed decision and its repeated interaction
with the game. The model cannot execute code or modify the simulation.

## Run

From the repository root, using the checkout's environment. Clone the separate
decision-model monorepo first if `decisions/` does not already exist:

```bash
source .venv/bin/activate
git clone https://github.com/kedi-lang/kedi-decisions.git decisions
uv pip install -e decisions -e decisions/packages/kedi-typesafe
uv pip install -e 'examples/flappy_kedi[jev]'
kedi examples/flappy_kedi/main.kedi
```

The editable install makes the local Python engine importable and installs
Pygame, Pymunk, and shared decision criteria. The `jev` extra adds TypeSafe;
Laya-only and manual installations do not need it. It does not publish a package or
replace the Kedi checkout. **The entrypoint is `main.kedi`, not a Python script.**

Put `TYPESAFE_API_KEY` in the checkout's `.env` or the environment. The Kedi CLI
loads the environment; the pilot inherits it. Jev uses `jev-latest`
through `kedi-typesafe`; the example does not require another model provider.
The pilot starts only when you press **Take off**. The initial loading screen
prepares its first decision before the flight begins. After takeoff the game
does not pause for model replies. Automatic pilots default to **0.5x speed**:
one wall-clock second advances half a second of game time, leaving room for
decision latency. Manual play defaults to 1x. The current speed appears in the
header. The default limit is 80 decisions per round.
Restarting starts a new round with a new decision budget.
`--max-decisions 1000` permits a longer round. The worker scales its Kedi host
budget with this limit: up to three model requests per decision for bounded
output repair, and at least 1,000,000 cumulative input tokens (otherwise 8,192
per permitted request). It no longer inherits the generic 128-request ceiling.
These limits apply only to this game worker, not other Kedi programs. They do
not change the provider's rate limits, request timeouts, or collision rules.

Override either mode with `--speed`, from 0.1 to 1.0. For example, use
`kedi examples/flappy_kedi/main.kedi --pilot jev --speed 1` for full speed.
The scale is fixed for the round: it never freezes the world or automatically
slows it near an obstacle. Collisions remain enabled; surviving is not guaranteed.

For manual play, no API key or model call is needed (use the same installation):

```bash
kedi examples/flappy_kedi/main.kedi --pilot manual
```

Click the scene or press **Space** to jump in manual mode. **P** pauses,
**R** restarts, and **Esc** closes the window. The top controls switch pilots,
pause, and restart. The window can be resized without changing the physics.
On Windows, activate `.venv\Scripts\Activate.ps1` instead of using `source`.

## What Kedi does

### Local Laya pilot

Install the separate local Laya integration; it is not part of `kedi-typesafe`:

```bash
uv pip install -e decisions -e 'decisions/packages/kedi-laya[mlx,pydantic]'
uv pip install -e examples/flappy_kedi
kedi examples/flappy_kedi/main.kedi --pilot laya
```

This uses `aac6fef/laya-multilingual-mlx` on Apple Silicon, with no API key.
The first run downloads roughly 678 MB. To use a checkpoint already on disk:

```bash
kedi examples/flappy_kedi/main.kedi --pilot laya \
  --model laya/tmp/laya-multilingual-mlx
```

`laya/<checkpoint>` selects MLX when installed on Apple Silicon, otherwise
PyTorch. Use `[torch,pydantic]` with a compatible checkpoint on other hosts;
`laya-mlx/<checkpoint>` still forces MLX. The Kedi pilot,
physics, observation, and direct-action policy stay unchanged. Confidence is
displayed and recorded, not used to veto a movement. Laya and Jev confidence
are calculated differently and must not be treated as matched probabilities.
A model startup/validation error is an error, not a game loss.
Weights stay resident for the round. Restart creates a fresh pilot process.

### Decision program

[main.kedi](main.kedi) imports the Python engine and opens the game itself.
[pilot.kedi](pilot.kedi) contains the decision program, importing
`ChoiceCriteria` with `> import: decisions`. The same criteria work with
either provider; the import does not select a model.
The `Movement` type names the two legal actions and describes when each is
appropriate. `@decide` puts the current observation into a natural-language
template and captures that finite choice. Its typed `Decision` includes the
selected action and `decision_info("movement").confidence`. Every valid choice
is returned with `accepted=True`, even when confidence is low or unavailable.
The host applies that action unchanged; it does not substitute `coast` based
on a confidence threshold.

`@play` repeatedly receives an observation, calls `decide`, and sends the
result back. A `None` observation ends the loop. The Python pilot
binds `pilot.kedi` with the public `@kedi.bind(file=...)` API and supplies
`PydanticAdapter(TypeSafeModel(...))` or the selected Laya model through
`kedi.context`.

| File | Responsibility |
| --- | --- |
| [main.kedi](main.kedi) | Entrypoint: import the engine and launch it with the Kedi pilot program |
| [pilot.kedi](pilot.kedi) | Choice criteria, typed capture, decision metadata, pilot loop |
| [flappy_kedi/app.py](flappy_kedi/app.py) | Imported host: input, round state, pause/restart, recording |
| [flappy_kedi/pilot.py](flappy_kedi/pilot.py) | Isolated pilot process and bounded request channel |
| [flappy_kedi/world.py](flappy_kedi/world.py) | Seeded course and fixed-step Pymunk physics |
| [flappy_kedi/view.py](flappy_kedi/view.py) | Original pixel sprites, skyline, controls, responsive canvas |

## Timing and information boundaries

- The model receives the cat's height and vertical speed, the next opening, and two
  approximate short trajectory forecasts computed from the public physics.
  This is state-based decision making, **not vision-based play**. A deterministic
  controller could also play this game; no advantage over one is claimed.
- Both `flap` and `coast` are applied as selected. Low or missing confidence
  does not replace a flap with coasting, and no extra request retries an
  uncertain choice. Traces retain the model's `action`, `accepted=True`, and
  identical `applied_action`. Invalid answers or provider failures still stop
  the round with an error.
- After the initial preparation, physics advances continuously at 120 fixed
  steps per simulated second, driven by elapsed frame time multiplied by the
  selected speed. Gravity, jump strength, and obstacle geometry are unchanged.
  A pending model
  request does not stop falling, scrolling, or collisions. Provider latency
  now affects how far the world has moved before an action is applied.
- Requests are spaced by at least 28 physics steps (about 0.233 simulated
  seconds, or 0.467 wall-clock seconds at 0.5x), with
  at most one in flight. This interval overlaps the model request; a slow reply
  does not add another full animation interval before the next request.
  Reaching the decision budget allows the last action 28 steps before ending
  the round, unless a collision occurs first.
- The main process continues processing window events while the pilot waits.
  Only the explicit pause control holds an active flight's state;
  a response already in flight can finish, but is not applied until resumed.
- Collision, restart, and close stop the old worker. Late replies cannot affect
  a finished or replacement round. A 25-second host
  deadline bounds each outstanding decision, including initial worker startup;
  the model client's request timeout is 12 seconds. SDK retries may mean more
  provider requests than the displayed number of decisions.
- Confidence is provider-specific evidence, not a calibrated probability of
  surviving the next obstacle. Laya's choice confidence is entropy-based and
  differs from the selected option's probability. Neither value changes the
  movement in this example.

## Record a bounded flight

```bash
kedi examples/flappy_kedi/main.kedi --headless --seed 7 --max-decisions 80 \
  --trace /tmp/flappy-kedi.jsonl --screenshot /tmp/flappy-kedi.png
```

Headless mode uses SDL's dummy display and the same speed-scaled pacing as the
visible game; it does not fast-forward while a real provider is thinking.
The output JSONL is
append-only and contains public observations, actions, confidence, latency,
and round outcomes, never API credentials. Start events record `speed` and
`timing: scaled_realtime` (or `realtime` at 1x); summaries also record `speed`.
Observations and decisions include `simulation_seconds`;
decisions also report `observation_age_seconds`, the simulated time elapsed
since their input snapshot. Model latency and deadlines remain in wall-clock
seconds, not scaled game time. `startup_seconds` separates initial preparation
from active flight. Use a fresh filename to isolate a run. Screenshots are
optional; parent directories must already exist.

`complete` means the configured decision limit was reached while still alive,
not that an endless game was won. `over` means collision; `error` means a pilot
or credential problem, not a scored loss. `stopped` means the user or frame
limit interrupted a running round. Provider errors propagate as Kedi runtime
errors; the CLI exits unsuccessfully instead of reporting a completed flight.

### Error logs

Pilot exceptions are logged immediately to the terminal's error output, with
the decision number and traceback. The window shows the underlying exception
type rather than only `KediExecutionError`. The pilot logger masks secret
environment values and Bearer/Basic credentials; it does not dump local variables.
Host-side errors such as a worker timeout are logged with the completed-decision
count. Redirect stderr to keep a log:

```bash
kedi examples/flappy_kedi/main.kedi --max-decisions 1000 2> flappy-kedi.log
```

This log is separate from the optional `--trace` JSONL recording. Review logs
before sharing: stack traces can still include local paths and source snippets.

## Verification

### Historical stop-and-wait flights

On September 30, 2026, the direct-action pilot was tested with real Jev and
local `laya-multilingual-mlx` calls. Each provider ran five headless flights:
seed 7 twice, followed by seeds 19, 42, and 101. The prompt, physics, observation
format, and 80-decision limit were unchanged between providers.
These flights paused the simulation during each model request. **They do not
establish survival or accuracy for the current continuous-physics game.**

| Measurement | Jev (`jev-latest`) | Laya (local MLX) |
| --- | --- | --- |
| Flights reaching the limit alive | 5/5 | 0/5 |
| Gates passed per flight | 5 | 0 |
| Decisions per flight | 80 | 8 |
| Simulated seconds per flight | 18.667 | 1.733 |
| Warm decision latency, median | 314 ms | 75 ms |
| Warm decision latency, p95 | 507 ms | 80 ms |
| Observed end state | Decision limit | Ceiling collision |

Jev selected 135 flaps and 265 coasts. Laya selected flap on all 40 decisions.
All selected actions were applied, including Jev's 203 decisions below 60%
confidence. These are bounded example checks on four distinct seeds, not
independent general-accuracy samples or evidence of unlimited survival.
Warm latency excludes the first decision of each flight and includes Kedi,
provider work, and network time for Jev. It is not isolated inference latency.
The old desktop mode additionally played each 0.233-second simulation batch at
its normal speed. The application makes fresh model requests, not replayed moves.

An earlier 60%-confidence policy blocked low-confidence flaps. In that version,
one Jev seed-7 flight collided after 15 decisions with zero gates, and all five
Laya flights collided with the ground after four decisions. Rejecting a flap
implicitly chose coast; it was not a neutral action. That gate is no longer
part of the example. The shared decision API's confidence semantics are unchanged.

### Real-time regression checks

The long-round budget fix was checked with 1,000 controlled model decisions
through the actual Kedi worker, plus 160 consecutive real Jev decisions over
recorded game observations. Before the fix the real loop stopped after 128
decisions with `ModelRequestBudgetError`; afterward all 160 completed in 53.09
seconds. This checks worker longevity, not survival in a live flight.

On October 1, 2026, a real Jev seed-7 flight **at 1x** advanced 2.792 simulated seconds
in 2.785 seconds of active wall time, after 1.541 seconds of initial preparation.
It collided after seven decisions, with no gates passed. After takeoff, decision
inputs were already 0.308-0.408 simulated seconds old when their answers arrived.
Every selected action was applied unchanged. This single flight verifies that
model requests no longer freeze the game; it does not establish reliable
real-time play, and the historical survival results above do not carry over.

The same day, a 0.6x seed-7 flight collided after 12 decisions with no gates.
At the new 0.5x default, seed 7 reached all 80 decisions alive with five gates
(18.950 simulated seconds, 39.258 total wall-clock seconds). Seed 19 collided
after 42 decisions with two gates (10.275 simulated seconds, 21.959 total
wall-clock seconds). Median warm reply latency was 322-324 ms; median input
age at application was 167 simulated ms in both half-speed flights. No model
choices were replaced. These exploratory runs justify leaving latency headroom,
not a survival guarantee or a statistically established success-rate improvement.

The regression tests exercise continuously advancing physics during pending
requests, startup preparation, explicit pause, request pacing, cancellation
after collision, restart isolation, speed scaling with unscaled decision latency,
and equal headless/desktop clock behavior.
They also run the actual Kedi loop with a controlled model and check rendering
without paid requests:

```bash
python -m pytest tests/test_flappy_kedi.py tests/test_curated_examples.py --record-mode=none
```

The game tests skip when its optional graphics dependencies are not installed.

## Assets and libraries

The cat and Istanbul scenery are original, code-defined pixel art. The bundled
[Pixelify Sans font](https://github.com/google/fonts/tree/main/ofl/pixelifysans)
uses the [SIL Open Font License](flappy_kedi/assets/OFL.txt).
[Lucide](https://github.com/lucide-icons/lucide) supplies the control icons;
its license is included in [LUCIDE_LICENSE](flappy_kedi/assets/LUCIDE_LICENSE).

[Pygame](https://www.pygame.org/docs/ref/event.html) owns the display/event loop;
[Pymunk](https://www.pymunk.org/en/latest/pymunk.html) supplies collision detection
and rigid-body integration. No browser or background service is required.
Manual and Jev modes do not require a local model runtime; Laya mode uses its
selected MLX or PyTorch backend.
