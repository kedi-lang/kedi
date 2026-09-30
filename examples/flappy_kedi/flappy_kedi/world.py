"""Fixed-step Pymunk simulation; no model calls or rendering."""

from __future__ import annotations

from dataclasses import dataclass
import random

import pymunk

WIDTH = 480
HEIGHT = 270
FLOOR = 250
CAT_X = 90
RADIUS = 9
GRAVITY = 260
FLAP_SPEED = -92
SCROLL_SPEED = 58
GATE_WIDTH = 32
GATE_SPACING = 200
GAP = 108
STEP = 1 / 120
TURN_STEPS = 28
TURN_SECONDS = TURN_STEPS * STEP


@dataclass
class Gate:
    body: pymunk.Body
    shapes: tuple[pymunk.Poly, pymunk.Poly]
    center: float
    passed: bool = False

    @property
    def x(self) -> float:
        return self.body.position.x


class World:
    def __init__(self, seed: int = 7):
        self.random = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        self.cat = pymunk.Body(1, float("inf"))
        self.cat.position = (CAT_X, 130)
        self.shape = pymunk.Circle(self.cat, RADIUS)
        self.shape.sensor = True
        self.shape.collision_type = 1
        self.space.add(self.cat, self.shape)
        for y in (10, FLOOR):
            boundary = pymunk.Segment(self.space.static_body, (0, y), (WIDTH, y), 1)
            boundary.collision_type = 2
            self.space.add(boundary)
        self.alive = True
        self.score = 0
        self.elapsed = 0.0
        self.space.on_collision(1, 2, begin=self.collide)
        self.gates = [self.add_gate(260 + i * GATE_SPACING) for i in range(3)]

    def collide(self, arbiter, space, data) -> None:
        self.alive = False

    def add_gate(self, x: float) -> Gate:
        center = self.random.randint(107, 163)
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (x, 0)
        body.velocity = (-SCROLL_SPEED, 0)
        upper = center - GAP / 2
        lower = center + GAP / 2
        shapes = (
            pymunk.Poly(body, [(0, 0), (GATE_WIDTH, 0), (GATE_WIDTH, upper), (0, upper)]),
            pymunk.Poly(body, [(0, lower), (GATE_WIDTH, lower), (GATE_WIDTH, FLOOR), (0, FLOOR)]),
        )
        for shape in shapes:
            shape.collision_type = 2
        self.space.add(body, *shapes)
        return Gate(body, shapes, center)

    @property
    def next_gate(self) -> Gate:
        return next(gate for gate in self.gates if gate.x + GATE_WIDTH >= CAT_X - RADIUS)

    def flap(self) -> None:
        if self.alive:
            self.cat.velocity = (0, FLAP_SPEED)

    def step(self) -> None:
        if not self.alive:
            return
        self.space.step(STEP)
        self.elapsed += STEP
        for gate in self.gates:
            if not gate.passed and gate.x + GATE_WIDTH < CAT_X - RADIUS:
                gate.passed = True
                self.score += 1
        while self.gates[0].x + GATE_WIDTH < 0:
            gate = self.gates.pop(0)
            self.space.remove(*gate.shapes, gate.body)
            self.gates.append(self.add_gate(self.gates[-1].x + GATE_SPACING))

    def apply_turn(self, action: str) -> None:
        if action not in {"flap", "coast"}:
            raise ValueError(f"Unknown movement: {action}")
        if action == "flap":
            self.flap()
        for _ in range(TURN_STEPS):
            self.step()

    def observation(self) -> str:
        gate = self.next_gate
        safe_top = gate.center - GAP / 2 + RADIUS + 3
        safe_bottom = gate.center + GAP / 2 - RADIUS - 3
        y = self.cat.position.y
        drift = GRAVITY * TURN_SECONDS**2 / 2
        coast_y = y + self.cat.velocity.y * TURN_SECONDS + drift
        flap_y = y + FLAP_SPEED * TURN_SECONDS + drift
        return (
            f"Cat center y={y:.1f}; vertical speed={self.cat.velocity.y:+.1f} px/s. "
            f"Next opening: safe cat-center range y={safe_top:.1f} to {safe_bottom:.1f}, "
            f"target center y={gate.center:.1f}. "
            f"Its left edge is {gate.x - CAT_X:.1f} px ahead; obstacles move left at "
            f"{SCROLL_SPEED} px/s. Ceiling y=10; ground y={FLOOR}. "
            f"Over the next {TURN_SECONDS:.2f}s, approximate forecasts: "
            f"COAST ends at y={coast_y:.1f}; FLAP ends at y={flap_y:.1f}. "
            "FLAP resets upward speed; COAST preserves momentum before gravity. "
            "A flap is not continuous upward thrust."
        )
