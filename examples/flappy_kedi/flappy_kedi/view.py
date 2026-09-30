"""Original pixel scenery, sprites, and a resizable Pygame interface."""

from __future__ import annotations

from pathlib import Path

import pygame

from .world import CAT_X, FLOOR, GAP, GATE_WIDTH, HEIGHT, WIDTH, World

SIZE = (960, 680)
SCENE_Y = 88
INK = (30, 48, 52)
PAPER = (244, 248, 239)
MINT = (157, 229, 195)
CORAL = (239, 131, 119)
ASSETS = Path(__file__).with_name("assets")

CAT = (
    "  XX            XX  ",
    "  XfX          XfX  ",
    "  XppX        XppX  ",
    "  XfffXXXXXXXXfffX  ",
    " XffffssssssffffffX ",
    " XfffssssssssfffffX ",
    "XfffssssssssssfffffX",
    "XffffssffffssffffffX",
    "XfffXXffffffXXfffffX",
    "XfffXeXffffXeXfffffX",
    "XffffffffffssffffffX",
    "XffXffffppffffXffffX",
    "XfffffffppfffffffffX",
    " XfffXffffffXfffffX ",
    "  XfffXXXXXXfffffX  ",
    "   XffffffffffffX   ",
    "    XXXXXXXXXXXX    ",
)


def cat_sprite() -> pygame.Surface:
    sprite = pygame.Surface((20, 18), pygame.SRCALPHA)
    colors = {"X": INK, "f": PAPER, "s": (122, 143, 143), "p": CORAL, "e": MINT}
    for y, row in enumerate(CAT):
        for x, pixel in enumerate(row):
            if pixel in colors:
                sprite.set_at((x, y), colors[pixel])
    return sprite


class View:
    def __init__(self):
        self.canvas = pygame.Surface(SIZE)
        self.scene = pygame.Surface((WIDTH, HEIGHT))
        self.cat = cat_sprite()
        self.fonts = {
            size: pygame.font.Font(str(ASSETS / "PixelifySans.ttf"), size)
            for size in (16, 20, 24, 30, 40, 54)
        }
        self.icons = {
            name: pygame.transform.scale(pygame.image.load(str(ASSETS / f"{name}.svg")), (22, 22))
            for name in ("play", "pause", "restart")
        }
        self.buttons = {
            "manual": pygame.Rect(382, 26, 100, 38),
            "jev": pygame.Rect(482, 26, 112, 38),
            "pause": pygame.Rect(848, 26, 38, 38),
            "restart": pygame.Rect(898, 26, 38, 38),
            "launch": pygame.Rect(374, 382, 212, 48),
        }

    def text(self, text: str, position, size=20, color=INK, center=False):
        rendered = self.fonts[size].render(text, False, color)
        rect = rendered.get_rect(center=position) if center else rendered.get_rect(topleft=position)
        self.canvas.blit(rendered, rect)

    def logical_position(self, screen: pygame.Surface, position):
        scale = min(screen.get_width() / SIZE[0], screen.get_height() / SIZE[1])
        x = (screen.get_width() - SIZE[0] * scale) / 2
        y = (screen.get_height() - SIZE[1] * scale) / 2
        return ((position[0] - x) / scale, (position[1] - y) / scale)

    def button_at(self, screen, position, status):
        point = self.logical_position(screen, position)
        for name, rect in self.buttons.items():
            if name == "launch" and status not in {"ready", "over", "complete", "error"}:
                continue
            if rect.collidepoint(point):
                return name
        return None

    def background(self, elapsed: float):
        surface = self.scene
        surface.fill((179, 222, 220))
        # Stepped clouds and distant rooftops share the scene's two-pixel grid.
        for x, y, width in ((30, 26, 56), (198, 43, 76), (389, 18, 50)):
            x = int((x - elapsed * 3) % (WIDTH + 90) - 45)
            pygame.draw.rect(surface, (224, 241, 226), (x, y + 8, width, 12))
            pygame.draw.rect(surface, (224, 241, 226), (x + 12, y, width - 26, 20))
        pygame.draw.rect(surface, (119, 177, 187), (0, 205, WIDTH, 45))
        for i, height in enumerate((25, 37, 20, 43, 30, 22, 32, 24, 40, 28, 19, 31)):
            x = i * 43
            pygame.draw.rect(surface, (106, 155, 162), (x, 205 - height, 34, height))
            pygame.draw.polygon(
                surface,
                (94, 137, 148),
                [(x - 2, 205 - height), (x + 17, 194 - height), (x + 36, 205 - height)],
            )
            for floor in range(205 - height + 8, 202, 10):
                for column in (8, 22):
                    pygame.draw.rect(surface, (173, 209, 194), (x + column, floor, 4, 5))
        # A small Galata-like tower anchors the skyline, not the foreground.
        pygame.draw.rect(surface, (87, 127, 143), (339, 142, 20, 63))
        pygame.draw.rect(surface, (96, 144, 153), (334, 150, 30, 6))
        pygame.draw.polygon(surface, (91, 115, 136), [(333, 144), (349, 113), (365, 144)])
        for x in (343, 351):
            pygame.draw.rect(surface, (180, 215, 196), (x, 161, 4, 9))
        for row in range(214, 246, 9):
            for x in range(-30, WIDTH, 48):
                drift = int(elapsed * 7 + row) % 48
                pygame.draw.line(surface, (172, 210, 202), (x + drift, row), (x + drift + 21, row))
        ship_x = int(150 - elapsed * 2) % WIDTH
        pygame.draw.rect(surface, PAPER, (ship_x, 216, 25, 5))
        pygame.draw.rect(surface, INK, (ship_x + 4, 214, 14, 3))
        pygame.draw.polygon(
            surface,
            (214, 119, 112),
            [(ship_x - 3, 222), (ship_x + 29, 222), (ship_x + 24, 227), (ship_x + 2, 227)],
        )

    def gate(self, gate):
        x = int(gate.x)
        top, bottom = int(gate.center - GAP / 2), int(gate.center + GAP / 2)
        for rect, edge in (
            (pygame.Rect(x, 0, GATE_WIDTH, top), top - 9),
            (pygame.Rect(x, bottom, GATE_WIDTH, FLOOR - bottom), bottom),
        ):
            pygame.draw.rect(self.scene, INK, rect)
            pygame.draw.rect(self.scene, (69, 123, 113), rect.inflate(-4, 0))
            pygame.draw.rect(self.scene, (101, 166, 136), (x + 3, rect.y, 5, rect.height))
            pygame.draw.rect(self.scene, INK, (x, edge, GATE_WIDTH, 9))
            pygame.draw.rect(self.scene, CORAL, (x + 1, edge + 2, GATE_WIDTH - 2, 5))
            for y in range(rect.y + 16, rect.bottom - 10, 21):
                pygame.draw.line(self.scene, (50, 98, 96), (x + 9, y), (x + 28, y))

    def draw(
        self,
        screen,
        world: World,
        *,
        mode: str,
        status: str,
        paused: bool,
        pending: bool,
        decisions: int,
        last_action: str,
        confidence: float | None,
        latency: float | None,
        best: int,
        accepted: bool | None = None,
        auto_mode: str = "jev",
        error: str = "",
        hover=None,
    ):
        self.canvas.fill(INK)
        self.canvas.blit(pygame.transform.scale(self.cat, (40, 36)), (24, 26))
        self.text("flappy kedi", (78, 17), 30, PAPER)
        self.text("BOSPHORUS HOP", (80, 51), 16, MINT)
        for name, label in (("manual", "YOU"), ("jev", f"{auto_mode.upper()} PILOT")):
            rect = self.buttons[name]
            active = (name == "manual") == (mode == "manual")
            pygame.draw.rect(self.canvas, MINT if active else (49, 68, 70), rect)
            self.text(label, rect.center, 20, INK if active else PAPER, center=True)
        self.text("SCORE", (640, 20), 16, MINT)
        self.text(f"{world.score:02}", (652, 40), 30, PAPER)
        self.text("BEST", (744, 20), 16, MINT)
        self.text(f"{best:02}", (754, 40), 30, PAPER)
        for name in ("pause", "restart"):
            rect = self.buttons[name]
            pygame.draw.rect(self.canvas, PAPER if hover == name else (194, 217, 206), rect)
            icon = "play" if name == "pause" and paused else name
            self.canvas.blit(self.icons[icon], (rect.x + 8, rect.y + 8))

        self.background(world.elapsed)
        for gate in world.gates:
            self.gate(gate)
        for x in range(0, WIDTH, 20):
            pygame.draw.rect(self.scene, (86, 106, 102), (x, FLOOR, 18, 20))
            pygame.draw.rect(self.scene, (192, 194, 158), (x, FLOOR, 18, 3))
        angle = max(-25, min(20, -world.cat.velocity.y / 5))
        cat = pygame.transform.rotate(self.cat, angle)
        self.scene.blit(cat, cat.get_rect(center=(CAT_X, int(world.cat.position.y))))
        pygame.transform.scale(
            self.scene, (960, 540), self.canvas.subsurface((0, SCENE_Y, 960, 540))
        )

        pygame.draw.rect(self.canvas, INK, (0, 628, 960, 52))
        state = "PAUSED" if paused else "JEV THINKING" if pending else "IN FLIGHT"
        if not paused and not pending and accepted is False:
            state = "LOW CONFIDENCE"
        if status != "running":
            state = status.upper()
        pygame.draw.rect(self.canvas, CORAL if pending else MINT, (24, 648, 7, 7))
        self.text(state, (44, 642), 20, PAPER)
        self.text(f"TURNS {decisions:02}", (266, 642), 20, MINT)
        self.text(last_action.upper(), (426, 642), 20, PAPER)
        self.text(
            "CONF --" if confidence is None else f"CONF {confidence:.0%}", (598, 642), 20, PAPER
        )
        self.text("-- ms" if latency is None else f"{latency * 1000:.0f} ms", (810, 642), 20, MINT)

        if status in {"ready", "over", "complete", "error"}:
            panel = pygame.Rect(220, 228, 520, 225)
            pygame.draw.rect(self.canvas, INK, panel.move(6, 6))
            pygame.draw.rect(self.canvas, PAPER, panel)
            pygame.draw.rect(self.canvas, MINT, (220, 228, 520, 7))
            heading = {
                "ready": "A little leap.",
                "over": "Missed the gap!",
                "complete": "Round complete.",
                "error": "Pilot unavailable.",
            }[status]
            self.text(heading, (480, 278), 40, center=True)
            sub = "Your turn." if mode == "manual" else f"{auto_mode.title()} at the controls."
            if status in {"over", "complete"}:
                sub = f"{world.score} gates / {world.elapsed:.1f}s of flight"
            if status == "error":
                sub = error
            words, lines, current = sub.split(), [], ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if self.fonts[20].size(candidate)[0] > 465:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
            for i, line in enumerate(lines[:2]):
                self.text(line, (480, 326 + i * 22), 20, center=True)
            rect = self.buttons["launch"]
            pygame.draw.rect(self.canvas, INK, rect)
            self.text(
                "TAKE OFF" if status == "ready" else "TRY AGAIN", rect.center, 24, PAPER, True
            )
        elif paused:
            self.text("PAUSED", (480, 312), 54, INK, center=True)
        if hover in {"pause", "restart"}:
            self.text(
                "Resume" if hover == "pause" and paused else hover.title(),
                (self.buttons[hover].centerx - 40, 70),
                16,
                PAPER,
            )
        scale = min(screen.get_width() / SIZE[0], screen.get_height() / SIZE[1])
        fitted = pygame.transform.scale(
            self.canvas, (round(SIZE[0] * scale), round(SIZE[1] * scale))
        )
        screen.fill(INK)
        screen.blit(fitted, fitted.get_rect(center=screen.get_rect().center))
