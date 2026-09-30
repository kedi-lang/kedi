"""Game host imported by main.kedi; the Kedi CLI remains the entrypoint."""

from __future__ import annotations

from dataclasses import asdict
from importlib.util import find_spec
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .pilot import Pilot
from .view import SIZE, View
from .world import STEP, TURN_STEPS, World


class Flight:
    def __init__(
        self, source: Path, *, seed=7, mode="jev", model=None, max_decisions=80, record=None
    ):
        self.source = source
        self.seed = seed
        self.mode = mode
        self.auto_mode = "laya" if mode == "laya" else "jev"
        self.model = model or "laya/aac6fef/laya-multilingual-mlx"
        self.max_decisions = max_decisions
        self.record = record
        self.best = 0
        self.pilot = None
        self.world = World(seed)
        self.status = "ready"
        self.paused = False
        self.decisions = 0
        self.steps_remaining = 0
        self.accumulator = 0.0
        self.last_action = "--"
        self.accepted = None
        self.confidence = None
        self.latency = None
        self.error = ""
        self.started = time.monotonic()

    def emit(self, event: str, **fields):
        if self.record is not None:
            self.record.write(json.dumps({"event": event, **fields}) + "\n")
            self.record.flush()

    def stop_pilot(self):
        if self.pilot is not None:
            self.pilot.close()
            self.pilot = None

    def start(self, mode=None):
        if self.status == "running":
            self.finish("stopped")
        else:
            self.stop_pilot()
        if mode is not None:
            self.mode = mode
        self.world = World(self.seed)
        self.decisions = self.steps_remaining = 0
        self.accumulator = 0.0
        self.paused = False
        self.last_action = "--"
        self.accepted = None
        self.confidence = self.latency = None
        self.error = ""
        self.status = "running"
        self.started = time.monotonic()
        self.emit(
            "start",
            seed=self.seed,
            mode=self.mode,
            max_decisions=self.max_decisions,
            model=self.model if self.mode == "laya" else "jev-latest",
        )
        if self.mode != "manual":
            package = "kedi_laya" if self.mode == "laya" else "kedi_typesafe"
            if find_spec("kedi_decisions") is None or find_spec(package) is None:
                self.error = "Install kedi-decisions and the selected provider integration."
            elif self.mode == "jev" and not os.getenv("TYPESAFE_API_KEY"):
                self.error = "TYPESAFE_API_KEY is missing. Manual play is available."
            else:
                self.pilot = (
                    Pilot(self.source, self.model) if self.mode == "laya" else Pilot(self.source)
                )
            if self.error:
                self.finish("error")

    def summary(self):
        return {
            "status": self.status,
            "mode": self.mode,
            "seed": self.seed,
            "score": self.world.score,
            "decisions": self.decisions,
            "simulation_seconds": round(self.world.elapsed, 3),
            "wall_seconds": round(time.monotonic() - self.started, 3),
        }

    def finish(self, status):
        self.status = status
        self.best = max(self.best, self.world.score)
        self.stop_pilot()
        if self.error:
            self.emit("error", message=self.error)
        self.emit("finish", **self.summary())

    def tick(self, seconds: float):
        if self.status != "running" or self.paused:
            return
        if self.mode != "manual" and self.steps_remaining == 0:
            self.accumulator = 0.0
            if self.decisions >= self.max_decisions:
                self.finish("complete")
                return
            if not self.pilot.pending:
                observation = self.world.observation()
                self.emit("observation", turn=self.decisions + 1, state=observation)
                self.pilot.request(observation)
            answer = self.pilot.poll()
            if answer is None:
                return
            if isinstance(answer, str):
                self.error = answer
                self.finish("error")
                return
            if answer.action not in {"flap", "coast"}:
                self.error = "Pilot returned an invalid movement."
                self.finish("error")
                return
            self.decisions += 1
            self.accepted = answer.accepted
            self.last_action, self.confidence, self.latency = (
                answer.action if answer.accepted else "coast",
                answer.confidence,
                answer.seconds,
            )
            self.emit(
                "decision", turn=self.decisions, applied_action=self.last_action, **asdict(answer)
            )
            if self.last_action == "flap":
                self.world.flap()
            self.steps_remaining = TURN_STEPS
        self.accumulator += min(seconds, 0.1)
        while self.accumulator >= STEP and self.world.alive:
            if self.mode != "manual" and self.steps_remaining == 0:
                break
            self.world.step()
            self.accumulator -= STEP
            if self.mode != "manual":
                self.steps_remaining -= 1
        if not self.world.alive:
            self.finish("over")

    def command(self, command):
        if command in {"manual", "jev"}:
            self.start(self.auto_mode if command == "jev" else "manual")
        elif command in {"launch", "restart"}:
            self.start()
        elif command == "pause" and self.status == "running":
            self.paused = not self.paused
        elif command == "flap" and self.mode == "manual" and not self.paused:
            if self.status != "running":
                self.start()
            self.world.flap()
            self.last_action = "flap"


def run_game(
    source: str,
    *,
    pilot="jev",
    model=None,
    seed=7,
    max_decisions=80,
    headless=False,
    autostart=False,
    frames=None,
    trace=None,
    screenshot=None,
):
    seed, max_decisions = int(seed), int(max_decisions)
    frames = int(frames) if frames is not None else None
    if pilot not in {"manual", "jev", "laya"}:
        raise ValueError("--pilot must be manual, jev or laya")
    if model is not None and (pilot != "laya" or not model.startswith(("laya/", "laya-mlx/"))):
        raise ValueError("--model requires --pilot laya and a laya/ or laya-mlx/ checkpoint")
    if not 1 <= max_decisions <= 1000:
        raise ValueError("--max-decisions must be between 1 and 1000")
    if frames is not None and frames < 1:
        raise ValueError("--frames must be positive")
    if not isinstance(headless, bool) or not isinstance(autostart, bool):
        raise ValueError("--headless and --autostart are flags; do not supply a value")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("flappy kedi")
    view = View()
    pygame.display.set_icon(view.cat)
    clock = pygame.time.Clock()
    record = Path(trace).open("a", encoding="utf-8") if trace else None
    flight = Flight(
        Path(source).resolve(),
        seed=seed,
        mode=pilot,
        model=model,
        max_decisions=max_decisions,
        record=record,
    )
    frame_count, running = 0, True
    try:
        if headless or autostart:
            flight.start()
        while running:
            seconds = clock.tick(120 if headless else 60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        command = {
                            pygame.K_SPACE: "flap",
                            pygame.K_p: "pause",
                            pygame.K_r: "restart",
                        }.get(event.key)
                        flight.command(command)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    command = view.button_at(screen, event.pos, flight.status)
                    flight.command(command or "flap")
            if not running:
                break
            flight.tick(0.1 if headless else seconds)
            view.draw(
                screen,
                flight.world,
                mode=flight.mode,
                auto_mode=flight.auto_mode,
                status=flight.status,
                paused=flight.paused,
                pending=bool(flight.pilot and flight.pilot.pending),
                decisions=flight.decisions,
                last_action=flight.last_action,
                confidence=flight.confidence,
                accepted=flight.accepted,
                latency=flight.latency,
                best=flight.best,
                error=flight.error,
                hover=view.button_at(screen, pygame.mouse.get_pos(), flight.status),
            )
            pygame.display.flip()
            frame_count += 1
            if headless and flight.status not in {"ready", "running"}:
                running = False
            if frames and frame_count >= frames:
                running = False
        if screenshot:
            pygame.image.save(view.canvas, str(screenshot))
        if flight.status == "running":
            flight.finish("stopped")
        if flight.status == "error":
            raise RuntimeError(flight.error)
        return json.dumps(flight.summary())
    finally:
        flight.stop_pilot()
        if record is not None:
            record.close()
        pygame.quit()
