# Example Modules

These modules demonstrate Kedi composition. Importing them can execute Python
setup; they are not guaranteed to work in a headless or dependency-free host.

## Wordle

`wordle` exports:

| Export | Purpose |
| --- | --- |
| `WordleSetup` | Typed generated game setup |
| `GuessValidation` | Typed guess assessment |
| `wordle` | Game agent profile |
| `prepare_wordle_setup(theme, seed_hint, blocked_answers)` | Model-backed setup procedure |
| `validate_wordle_guess(guess)` | Model-backed guess validator |
| `play_wordle(theme)` | Graphical game session, returns its summary |
| `play_default_wordle()` | Starts the default everyday-objects theme |

The graphical path needs its optional dependencies and a display. Model-backed
procedures also need a configured provider. Import only the explicit exports
you intend to call; private rendering and scoring helpers are implementation
details. Do not use this game module as a general schema for production agents.

## This

`this` prints its bundled demonstration text during module initialization. It
does not define a reusable application API. A separate root compilation may
execute that import-time effect again, consistent with ordinary module lifetime.
