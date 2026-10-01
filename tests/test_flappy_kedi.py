"""Exercise the published game without paid model calls or a visible display."""

import importlib
from io import StringIO
import json
from pathlib import Path
from queue import Queue
from typing import Any
from unittest.mock import MagicMock

import pytest

import kedi
from kedi.agent_adapter import PydanticAdapter

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/flappy_kedi"


@pytest.mark.parametrize("observations", [[], ["Cat is below the gap", "Cat is above the gap"]])
@pytest.mark.parametrize("confidence", [0.0, 0.59, 0.60, 0.9])
def test_kedi_pilot_receives_decides_and_stops_at_eof(observations, confidence, game):
    pytest.importorskip("kedi_decisions")
    from kedi_typesafe import TypeSafeModel
    from typesafe_sdk import Choice, ChoiceAnswer, SystemOneResponse, Usage

    class Client:
        calls = 0

        async def system_one(self, state, questions, *, model=None):
            assert observations[self.calls] in str(state)
            assert len(questions) == 1
            question = next(iter(questions.values()))
            assert isinstance(question, Choice)
            action = ["flap", "coast"][self.calls]
            self.calls += 1
            return SystemOneResponse(
                model="test-jev",
                answers={
                    name: ChoiceAnswer(
                        choice=action,
                        confidence=confidence,
                        probabilities={action: 0.9, "coast" if action == "flap" else "flap": 0.1},
                    )
                    for name in questions
                },
                usage=Usage(input_tokens=100, output_tokens=1),
            )

        async def aclose(self):
            pass

    class Channel:
        def __init__(self):
            self.states = iter(observations)
            self.answers = []

        def receive(self):
            return next(self.states, None)

        def send(self, decision):
            self.answers.append(decision)

    @kedi.bind(file=EXAMPLE / "pilot.kedi")
    def play(turns: Any) -> str: ...

    client, channel = Client(), Channel()
    with kedi.context(adapter=PydanticAdapter(TypeSafeModel(client=client)), model=None):
        assert play(channel) == "stopped"
    assert client.calls == len(observations)
    assert [answer.action for answer in channel.answers] == ["flap", "coast"][: len(observations)]
    assert all(answer.confidence == confidence for answer in channel.answers)
    assert all(answer.accepted is True for answer in channel.answers)


def test_missing_decision_evidence_does_not_block_selected_action(game):
    pytest.importorskip("kedi_decisions")
    from pydantic_ai.models.test import TestModel

    class Channel:
        answer = None

        def receive(self):
            return "Cat is falling" if self.answer is None else None

        def send(self, answer):
            self.answer = answer

    @kedi.bind(file=EXAMPLE / "pilot.kedi")
    def play(turns: Any) -> str: ...

    channel = Channel()
    with kedi.context(
        adapter=PydanticAdapter(TestModel(call_tools=[], custom_output_args={"movement": "flap"})),
        model=None,
    ):
        assert play(channel) == "stopped"
    assert channel.answer.action == "flap"
    assert channel.answer.confidence is None
    assert channel.answer.accepted is True


@pytest.mark.parametrize("max_decisions", [160, 1000])
def test_worker_budget_supports_the_full_requested_round(game, monkeypatch, max_decisions):
    from flappy_kedi.pilot import Answer, serve
    from kedi.agent_adapter.request_budget import ModelRequestBudget
    from pydantic_ai.models.test import TestModel
    import kedi_typesafe

    class Channel:
        answers = 0

        def get(self):
            return (
                "The cat is falling toward the opening." if self.answers < max_decisions else None
            )

        def put(self, answer):
            assert isinstance(answer, Answer), answer
            self.answers += 1

    monkeypatch.setattr(
        kedi_typesafe,
        "TypeSafeModel",
        lambda **kwargs: TestModel(call_tools=[], custom_output_args={"movement": "coast"}),
    )
    channel = Channel()
    previous_limits = ModelRequestBudget().limits
    serve(EXAMPLE / "pilot.kedi", channel, channel, max_decisions=max_decisions)
    assert channel.answers == max_decisions
    assert ModelRequestBudget().limits == previous_limits


def test_pilot_errors_log_details_without_credentials(game, monkeypatch, caplog):
    from flappy_kedi.pilot import serve
    from kedi.errors import KediExecutionError
    import kedi_typesafe

    secret = "test-only-secret-not-a-real-api-key"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)

    def broken_model(**kwargs):
        original = RuntimeError(
            f"Service failed; key={secret}; Authorization: Bearer sensitive-value"
        )
        raise KediExecutionError("Model unavailable", original=original) from original

    monkeypatch.setattr(kedi_typesafe, "TypeSafeModel", broken_model)
    outgoing = Queue()
    serve(EXAMPLE / "pilot.kedi", Queue(), outgoing)
    assert "Pilot failed at decision 0" in caplog.text
    assert "Traceback" in caplog.text and "Service failed" in caplog.text
    assert secret not in caplog.text and "sensitive-value" not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert outgoing.get_nowait() == "Decision 0 failed (RuntimeError). See terminal error log."


def test_worker_logs_failed_decision_and_unwraps_runtime_error(game, monkeypatch, caplog):
    from flappy_kedi.pilot import Answer, serve
    from pydantic_ai.models.test import TestModel
    import kedi_typesafe

    calls = 0

    class FailingModel(TestModel):
        async def request(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("Simulated provider failure")
            return await super().request(*args, **kwargs)

    monkeypatch.setattr(
        kedi_typesafe,
        "TypeSafeModel",
        lambda **kwargs: FailingModel(call_tools=[], custom_output_args={"movement": "coast"}),
    )
    incoming, outgoing = Queue(), Queue()
    for _ in range(3):
        incoming.put("The cat is falling.")
    incoming.put(None)
    serve(EXAMPLE / "pilot.kedi", incoming, outgoing, max_decisions=1000)
    assert isinstance(outgoing.get_nowait(), Answer)
    assert isinstance(outgoing.get_nowait(), Answer)
    assert outgoing.get_nowait() == "Decision 3 failed (RuntimeError). See terminal error log."
    assert "Pilot failed at decision 3" in caplog.text
    assert "Simulated provider failure" in caplog.text


@pytest.fixture
def game(monkeypatch):
    pytest.importorskip("pymunk")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    monkeypatch.syspath_prepend(str(EXAMPLE))
    module = importlib.import_module("flappy_kedi.app")
    pygame.display.init()
    pygame.font.init()
    yield module
    pygame.quit()


def test_physics_is_seeded_and_actions_are_real(game):
    world = game.World(7)
    assert [g.center for g in world.gates] == [g.center for g in game.World(7).gates]
    y = world.cat.position.y
    world.apply_turn("flap")
    assert world.cat.position.y < y
    with pytest.raises(ValueError, match="Unknown movement"):
        world.apply_turn("teleport")
    for _ in range(1000):
        world.step()
    assert not world.alive
    stopped = world.elapsed
    world.step()
    world.flap()
    assert world.elapsed == stopped


def test_course_can_be_passed_and_recycles_without_adding_shapes(game):
    world = game.World(7)
    shapes = len(world.space.shapes)
    first_gate = world.gates[0]
    # Test-only controller verifies course geometry, not model quality.
    for _ in range(100):
        projected = world.cat.position.y + world.cat.velocity.y * 0.22 + 6
        world.apply_turn("flap" if projected > world.next_gate.center else "coast")
        assert world.alive
    assert world.score >= 5
    assert first_gate not in world.gates
    assert len(world.space.shapes) == shapes
    observation = world.observation()
    assert "COAST ends at" in observation and "FLAP ends at" in observation


def test_manual_controls_pause_restart_and_do_not_need_credentials(game, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    flight = game.Flight(EXAMPLE / "pilot.kedi", mode="manual")
    flight.command("flap")
    assert flight.status == "running" and flight.pilot is None
    flight.tick(0.1)
    flight.command("pause")
    y = flight.world.cat.position.y
    flight.tick(0.1)
    flight.command("flap")
    assert flight.world.cat.position.y == y
    flight.command("restart")
    assert not flight.paused and flight.world.elapsed == 0
    flight.command("jev")
    assert flight.status == "error" and "TYPESAFE_API_KEY" in flight.error
    flight.command("manual")
    assert flight.status == "running" and not flight.error


@pytest.fixture
def pilot_game(game, monkeypatch):
    from flappy_kedi.pilot import Answer

    class FakePilot:
        def __init__(self, source, model_id=None, *, max_decisions=80):
            assert source == EXAMPLE / "pilot.kedi"
            self.max_decisions = max_decisions
            self.pending = False
            self.closed = False
            self.requests = []
            self.answer = None

        def request(self, observation):
            self.pending = True
            self.requests.append(observation)

        def poll(self):
            answer, self.answer = self.answer, None
            if answer is not None:
                self.pending = False
            return answer

        def close(self):
            self.closed = True

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-not-a-credential")
    monkeypatch.setattr(game, "Pilot", FakePilot)
    monkeypatch.setattr(game, "find_spec", lambda name: True)
    return game, Answer


def test_pending_pause_budget_restart_and_recording(pilot_game):
    game, Answer = pilot_game
    record = StringIO()
    flight = game.Flight(EXAMPLE / "pilot.kedi", max_decisions=2, record=record, speed=1)
    flight.start()
    old = flight.pilot
    assert old.max_decisions == 2
    for _ in range(5):
        flight.tick(0.1)
    assert flight.status == "preparing" and flight.world.elapsed == 0
    old.answer = Answer("coast", 0.9, 0.3, True)
    flight.tick(0)
    assert flight.status == "running" and flight.decisions == 1
    start_y = flight.world.cat.position.y
    start_gate_x = flight.world.next_gate.x
    for _ in range(5):
        flight.tick(0.1)
    assert len(old.requests) == 2 and flight.world.elapsed == pytest.approx(0.5)
    assert flight.world.cat.position.y > start_y
    assert flight.world.next_gate.x < start_gate_x
    paused_at = flight.world.elapsed
    flight.command("pause")
    old.answer = Answer("flap", 0.9, 0.3, True)
    flight.tick(0.1)
    assert old.answer is not None and flight.decisions == 1
    assert flight.world.elapsed == paused_at
    flight.command("pause")
    for _ in range(5):
        flight.tick(0.1)
    assert flight.status == "complete" and flight.decisions == 2
    assert flight.world.elapsed == pytest.approx(paused_at + game.STEP * game.TURN_STEPS)
    assert old.closed and flight.pilot is None
    flight.command("restart")
    next_pilot = flight.pilot
    flight.command("restart")
    assert next_pilot.closed and flight.pilot is not next_pilot
    flight.finish("stopped")
    events = [json.loads(row) for row in record.getvalue().splitlines()]
    decisions = [row for row in events if row["event"] == "decision"]
    assert len(decisions) == 2
    assert decisions[0]["observation_age_seconds"] == 0
    assert decisions[1]["observation_age_seconds"] > 0
    assert [row["status"] for row in events if row["event"] == "finish"] == [
        "complete",
        "stopped",
        "stopped",
    ]


def test_late_answer_does_not_add_an_extra_animation_wait(pilot_game):
    game, Answer = pilot_game
    flight = game.Flight(EXAMPLE / "pilot.kedi", speed=1)
    flight.start()
    pilot = flight.pilot
    pilot.answer = Answer("coast", 0.9, 0.3, True)
    flight.tick(0)
    for _ in range(6):
        flight.tick(0.1)
    assert len(pilot.requests) == 2
    pilot.answer = Answer("flap", 0.3, 0.4, True)
    flight.tick(game.STEP)
    assert flight.decisions == 2
    before = flight.world.elapsed
    flight.tick(game.STEP)
    assert len(pilot.requests) == 3 and pilot.pending
    assert flight.world.elapsed == pytest.approx(before + game.STEP)
    flight.finish("stopped")


def test_fast_answers_are_rate_limited_without_stopping_physics(pilot_game):
    game, Answer = pilot_game
    flight = game.Flight(EXAMPLE / "pilot.kedi", speed=1)
    flight.start()
    pilot = flight.pilot
    pilot.answer = Answer("flap", 0.2, 0.01, True)
    for _ in range(game.TURN_STEPS):
        flight.tick(game.STEP)
    assert len(pilot.requests) == 1
    assert flight.world.elapsed == pytest.approx(game.TURN_STEPS * game.STEP)
    flight.tick(game.STEP)
    assert len(pilot.requests) == 2 and pilot.pending
    flight.finish("stopped")


def test_collision_cancels_pending_decision_and_ignores_late_answer(pilot_game):
    game, Answer = pilot_game
    record = StringIO()
    flight = game.Flight(EXAMPLE / "pilot.kedi", record=record, speed=1)
    flight.start()
    pilot = flight.pilot
    pilot.answer = Answer("coast", 0.9, 0.3, True)
    flight.tick(0)
    for _ in range(12):
        flight.tick(0.1)
    assert flight.status == "over" and pilot.closed and flight.pilot is None
    assert flight.decisions == 1 and len(pilot.requests) == 2
    stopped_at = flight.world.elapsed
    pilot.answer = Answer("flap", 0.9, 1.0, True)
    flight.tick(0.1)
    assert flight.world.elapsed == stopped_at and flight.decisions == 1
    assert (
        sum(row["event"] == "decision" for row in map(json.loads, record.getvalue().splitlines()))
        == 1
    )


def test_restart_cannot_apply_previous_worker_answer(pilot_game):
    game, Answer = pilot_game
    flight = game.Flight(EXAMPLE / "pilot.kedi")
    flight.start()
    old = flight.pilot
    flight.tick(0.1)
    old.answer = Answer("flap", 0.9, 0.1, True)
    flight.command("restart")
    assert old.closed and flight.pilot is not old
    flight.tick(0.1)
    assert flight.decisions == 0 and flight.last_action == "--"
    assert flight.status == "preparing" and flight.world.elapsed == 0
    flight.pilot.answer = Answer("coast", 0.9, 0.1, True)
    flight.tick(0.1)
    assert flight.world.cat.velocity.y > 0
    flight.finish("stopped")


@pytest.mark.parametrize("speed", [0.4, 0.5, 0.6, 1.0])
def test_speed_scales_continuous_physics_not_pilot_latency(pilot_game, speed):
    game, Answer = pilot_game
    record = StringIO()
    flight = game.Flight(EXAMPLE / "pilot.kedi", speed=speed, record=record)
    flight.start()
    pilot = flight.pilot
    pilot.answer = Answer("coast", 0.9, 0.3, True)
    flight.tick(0)
    for _ in range(game.TURN_STEPS):
        flight.tick(game.STEP / speed)
    flight.tick(0)
    assert pilot.pending and len(pilot.requests) == 2
    observed_at = flight.world.elapsed
    for _ in range(3):
        flight.tick(0.1)
    assert flight.world.elapsed - observed_at == pytest.approx(0.3 * speed, abs=game.STEP)
    assert flight.world.elapsed - observed_at + flight.accumulator == pytest.approx(0.3 * speed)
    pilot.answer = Answer("flap", 0.8, 0.3, True)
    flight.tick(0)
    assert flight.last_action == "flap" and flight.latency == 0.3
    events = list(map(json.loads, record.getvalue().splitlines()))
    assert events[0]["speed"] == speed
    assert events[0]["timing"] == ("realtime" if speed == 1 else "scaled_realtime")
    assert events[-1]["observation_age_seconds"] == pytest.approx(0.3 * speed, abs=game.STEP)
    flight.finish("stopped")
    assert flight.summary()["speed"] == speed


@pytest.mark.parametrize("speed", [None, 0.4])
def test_speed_default_follows_pilot_but_explicit_speed_survives_restart(pilot_game, speed):
    game, _ = pilot_game
    flight = game.Flight(EXAMPLE / "pilot.kedi", speed=speed)
    flight.start()
    assert flight.speed == (0.5 if speed is None else speed)
    flight.command("manual")
    assert flight.speed == (1.0 if speed is None else speed)
    flight.command("jev")
    flight.command("restart")
    assert flight.speed == (0.5 if speed is None else speed)
    flight.finish("stopped")


@pytest.mark.parametrize("answer", ["Connection failed", "invalid"])
def test_provider_errors_never_turn_into_a_scored_game_loss(pilot_game, answer):
    game, Answer = pilot_game
    flight = game.Flight(EXAMPLE / "pilot.kedi")
    flight.start()
    pilot = flight.pilot
    pilot.answer = Answer(answer, 0.1, 0.2, False) if answer == "invalid" else answer
    flight.tick(0.1)
    assert flight.status == "error" and pilot.closed
    assert flight.decisions == 0 and flight.world.alive


@pytest.mark.parametrize("action", ["flap", "coast"])
@pytest.mark.parametrize("confidence", [0.0, 0.59, None])
def test_selected_action_is_applied_with_low_or_missing_confidence(pilot_game, action, confidence):
    game, Answer = pilot_game
    record = StringIO()
    flight = game.Flight(EXAMPLE / "pilot.kedi", record=record)
    flight.start()
    flight.pilot.answer = Answer(action, confidence, 0.3, True)
    flight.tick(0.1)
    assert flight.last_action == action and flight.accepted is True
    assert flight.confidence == confidence
    assert (flight.world.cat.velocity.y < 0) is (action == "flap")
    decision = next(
        row for row in map(json.loads, record.getvalue().splitlines()) if row["event"] == "decision"
    )
    assert decision["action"] == decision["applied_action"] == action
    assert decision["accepted"] is True and decision["confidence"] == confidence
    flight.finish("stopped")


def test_rejected_flap_is_recorded_but_not_applied(pilot_game):
    game, Answer = pilot_game
    record = StringIO()
    flight = game.Flight(EXAMPLE / "pilot.kedi", record=record)
    flight.start()
    flight.pilot.answer = Answer("flap", 0.59, 0.3, False)
    flight.tick(0.1)
    assert flight.last_action == "coast" and flight.accepted is False
    assert flight.world.cat.velocity.y > 0
    decision = next(
        row for row in map(json.loads, record.getvalue().splitlines()) if row["event"] == "decision"
    )
    assert decision["action"] == "flap" and decision["applied_action"] == "coast"
    assert decision["accepted"] is False
    flight.finish("stopped")


def test_pilot_deadline_and_close_are_bounded_and_idempotent(game, monkeypatch):
    from flappy_kedi import pilot

    context = MagicMock()
    process = context.Process.return_value
    process.is_alive.return_value = True
    incoming, outgoing = MagicMock(), Queue()
    context.Queue.side_effect = [incoming, outgoing]
    outgoing.cancel_join_thread = MagicMock()
    outgoing.close = MagicMock()
    monkeypatch.setattr(pilot.mp, "get_context", lambda mode: context)
    worker = pilot.Pilot(EXAMPLE / "pilot.kedi", max_decisions=1000)
    assert context.Process.call_args.kwargs["args"][-1] == 1000
    worker.request("public state")
    with pytest.raises(RuntimeError, match="Only one decision"):
        worker.request("duplicate state")
    monkeypatch.setattr(pilot.time, "monotonic", lambda: worker.started + 26)
    assert "timed out" in worker.poll()
    worker.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    process.close.assert_called_once()
    assert "closed" in worker.poll()
    with pytest.raises(RuntimeError, match="closed"):
        worker.request("late state")


def test_real_spawned_worker_can_be_cancelled_without_any_model_request(game):
    worker = game.Pilot(EXAMPLE / "pilot.kedi")
    process = worker.process
    worker.close()
    worker.close()
    assert worker.closed and process._closed


@pytest.mark.parametrize("size", [(960, 680), (480, 340), (400, 700)])
@pytest.mark.parametrize("status", ["ready", "preparing", "running", "error", "complete"])
def test_rendering_and_letterboxed_button_mapping(game, size, status):
    pygame = pytest.importorskip("pygame")
    view = game.View()
    screen = pygame.display.set_mode(size)
    view.draw(
        screen,
        game.World(),
        mode="jev",
        status=status,
        paused=False,
        pending=True,
        decisions=80,
        last_action="coast",
        confidence=0.99,
        latency=12.0,
        best=5,
        error="Pilot stopped. Check credentials and connection.",
        hover="restart",
    )
    assert len(set(pygame.image.tobytes(screen, "RGB"))) > 50
    scale = min(size[0] / 960, size[1] / 680)
    offset = ((size[0] - 960 * scale) / 2, (size[1] - 680 * scale) / 2)
    for button, rect in view.buttons.items():
        position = (rect.centerx * scale + offset[0], rect.centery * scale + offset[1])
        result = view.button_at(screen, position, status)
        assert result == (
            None if button == "launch" and status in {"preparing", "running"} else button
        )


def test_bounded_kedi_entrypoint_saves_nonblank_frame(game, tmp_path, monkeypatch):
    from kedi.lang import compile_program, parse_program

    image = tmp_path / "frame.png"
    source = EXAMPLE / "main.kedi"
    monkeypatch.chdir(tmp_path)
    runtime = compile_program(
        parse_program(source.read_text(), source_path=str(source)),
        runtime_globals={
            "args": {"headless": True, "pilot": "manual", "frames": "3", "screenshot": str(image)}
        },
    )
    try:
        result = json.loads(runtime.run_main())
    finally:
        runtime.artifacts.close()
    assert image.stat().st_size > 1000
    assert result["status"] == "stopped" and result["mode"] == "manual"


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("speed", ["0.25", "0.6", "1.0"])
def test_display_modes_use_elapsed_time_not_accelerated_batches(game, monkeypatch, headless, speed):
    pygame = pytest.importorskip("pygame")
    clock = MagicMock()
    clock.tick.return_value = 10
    monkeypatch.setattr(pygame.time, "Clock", lambda: clock)
    result = json.loads(
        game.run_game(
            str(EXAMPLE / "pilot.kedi"),
            pilot="manual",
            headless=headless,
            autostart=True,
            frames=10,
            speed=speed,
        )
    )
    assert result["simulation_seconds"] == pytest.approx(0.1 * float(speed), abs=0.009)
    assert result["speed"] == float(speed)
    assert clock.tick.call_count == 10
    clock.tick.assert_called_with(60)


@pytest.mark.parametrize(
    "options",
    [{"pilot": "unknown"}, {"max_decisions": "0"}, {"frames": "0"}, {"headless": "false"}],
)
def test_invalid_options_fail_before_opening_a_display(game, options):
    with pytest.raises(ValueError):
        game.run_game(str(EXAMPLE / "main.kedi"), **options)


@pytest.mark.parametrize("speed", [0, -1, 0.01, 1.1, "nan", "inf", "slow", True])
def test_invalid_speed_fails_before_opening_a_display(game, monkeypatch, speed):
    display = MagicMock()
    monkeypatch.setattr(game.pygame.display, "set_mode", display)
    with pytest.raises(ValueError):
        game.run_game(str(EXAMPLE / "main.kedi"), speed=speed)
    display.assert_not_called()
