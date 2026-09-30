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
@pytest.mark.parametrize("confidence,accepted", [(0.59, False), (0.60, True), (0.9, True)])
def test_kedi_pilot_receives_decides_and_stops_at_eof(observations, confidence, accepted, game):
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
    assert all(answer.accepted is accepted for answer in channel.answers)


def test_missing_decision_evidence_is_not_accepted(game):
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
    assert channel.answer.accepted is False


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
        def __init__(self, source):
            assert source == EXAMPLE / "pilot.kedi"
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
    flight = game.Flight(EXAMPLE / "pilot.kedi", max_decisions=1, record=record)
    flight.start()
    old = flight.pilot
    for _ in range(5):
        flight.tick(0.1)
    assert len(old.requests) == 1 and flight.world.elapsed == 0
    flight.command("pause")
    old.answer = Answer("flap", 0.9, 0.3, True)
    flight.tick(0.1)
    assert old.answer is not None and flight.decisions == 0
    flight.command("pause")
    for _ in range(5):
        flight.tick(0.1)
    assert flight.status == "complete" and flight.decisions == 1
    assert flight.world.elapsed == pytest.approx(game.STEP * game.TURN_STEPS)
    assert old.closed and flight.pilot is None
    flight.command("restart")
    next_pilot = flight.pilot
    flight.command("restart")
    assert next_pilot.closed and flight.pilot is not next_pilot
    flight.finish("stopped")
    events = [json.loads(row) for row in record.getvalue().splitlines()]
    assert sum(row["event"] == "decision" for row in events) == 1
    assert [row["status"] for row in events if row["event"] == "finish"] == [
        "complete",
        "stopped",
        "stopped",
    ]


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
    worker = pilot.Pilot(EXAMPLE / "pilot.kedi")
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
@pytest.mark.parametrize("status", ["ready", "running", "error", "complete"])
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
        assert result == (None if button == "launch" and status == "running" else button)


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


@pytest.mark.parametrize(
    "options",
    [{"pilot": "unknown"}, {"max_decisions": "0"}, {"frames": "0"}, {"headless": "false"}],
)
def test_invalid_options_fail_before_opening_a_display(game, options):
    with pytest.raises(ValueError):
        game.run_game(str(EXAMPLE / "main.kedi"), **options)
