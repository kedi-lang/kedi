from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

from kedi.lang import compile_program, parse_program
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter


class WordleModuleAdapter(MockAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._setup_payloads = [
            {
                "answer": "crane",
                "theme_label": "Space Tools",
                "clue": "Used in orbit.",
                "intro": "Guess the five-letter word.",
            },
            {
                "answer": "slate",
                "theme_label": "Desk Objects",
                "clue": "Seen near a notebook.",
                "intro": "A fresh five-letter round.",
            },
        ]

    def produce_sync(self, *, template, output_schema=None, **kwargs):
        self.model_calls.append(self.last_effective_model)
        self.effort_calls.append(self.last_effective_effort)
        self.instruction_calls.append(kwargs.get("instructions"))
        self.tool_calls.append(tuple(tool.name for tool in self.scoped_tools))
        keys = tuple(sorted((output_schema or {}).keys()))
        if keys == ("answer", "clue", "intro", "theme_label"):
            payload = self._setup_payloads.pop(0)
            return AttrDict(payload)
        if keys == ("valid",):
            return AttrDict({"valid": "qzxjk" not in template})
        return super().produce_sync(template=template, output_schema=output_schema, **kwargs)


class _FakeRenderedText:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_height(self) -> int:
        return 20

    def get_rect(self, *, center) -> object:
        return types.SimpleNamespace(center=center)


class _FakeFont:
    def size(self, text: str) -> tuple[int, int]:
        return (max(len(text), 1) * 10, 20)

    def render(self, text: str, *_args, **_kwargs) -> _FakeRenderedText:
        return _FakeRenderedText(text)


class _FakeScreen:
    def __init__(self) -> None:
        self.fill_calls: list[tuple[int, int, int]] = []
        self.blit_calls: list[tuple[object, object]] = []

    def fill(self, color: tuple[int, int, int]) -> None:
        self.fill_calls.append(color)

    def blit(self, surface: object, dest: object) -> None:
        self.blit_calls.append((surface, dest))


class _FakeRect:
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


def _install_pygame_stub(
    event_batches: list[list[object]] | None = None,
) -> types.ModuleType:
    pygame = types.ModuleType("pygame")
    pygame.__spec__ = importlib.machinery.ModuleSpec("pygame", loader=None)
    pygame._draw_calls = []
    pygame._screen = None
    pygame._event_batches = list(event_batches or [])
    pygame.font = types.SimpleNamespace(Font=_FakeFont, SysFont=lambda *a, **k: _FakeFont())
    pygame.Surface = _FakeScreen
    pygame.Rect = _FakeRect

    def _draw_rect(_screen, color, rect, width=0, border_radius=0):
        pygame._draw_calls.append(
            {
                "color": color,
                "rect": rect,
                "width": width,
                "border_radius": border_radius,
            }
        )

    pygame.draw = types.SimpleNamespace(rect=_draw_rect)

    def _set_mode(*_args, **_kwargs):
        screen = _FakeScreen()
        pygame._screen = screen
        return screen

    pygame.display = types.SimpleNamespace(
        set_caption=lambda *a, **k: None,
        set_mode=_set_mode,
        flip=lambda: None,
    )
    pygame.time = types.SimpleNamespace(
        Clock=lambda: types.SimpleNamespace(tick=lambda *a, **k: None)
    )
    pygame.event = types.SimpleNamespace(
        get=lambda: pygame._event_batches.pop(0) if pygame._event_batches else []
    )
    pygame.init = lambda: None
    pygame.quit = lambda: None
    pygame.QUIT = 0
    pygame.KEYDOWN = 1
    pygame.K_TAB = 9
    pygame.K_ESCAPE = 27
    pygame.K_RETURN = 13
    pygame.K_BACKSPACE = 8
    pygame.K_r = ord("r")
    sys.modules["pygame"] = pygame
    return pygame


def _compile_wordle_runtime(
    event_batches: list[list[object]] | None = None,
    adapter: WordleModuleAdapter | None = None,
):
    pygame = _install_pygame_stub(event_batches=event_batches)
    path = Path("src/kedi/modules/wordle.kedi").resolve()
    source = path.read_text(encoding="utf-8")
    runtime_adapter = adapter or WordleModuleAdapter(default_model="mock-model")
    rt = compile_program(
        parse_program(source, source_path=str(path)),
        adapter=runtime_adapter,
    )
    return rt, pygame


def test_validate_wordle_guess_returns_typed_object() -> None:
    rt, _pygame = _compile_wordle_runtime()
    adapter = rt.adapter
    assert adapter is not None

    result = rt._invoke_procedure(
        rt._procedures["validate_wordle_guess"],
        ["slate"],
    )

    assert result is not None
    assert result.valid is True
    assert adapter.model_calls == ["mock-model"]


def test_validate_wordle_guess_rejects_non_dictionary_word_locally() -> None:
    rt, _pygame = _compile_wordle_runtime()
    adapter = rt.adapter
    assert adapter is not None

    result = rt._invoke_procedure(
        rt._procedures["validate_wordle_guess"],
        ["qzxjk"],
    )

    assert result is not None
    assert result.valid is False
    assert adapter.model_calls == ["mock-model"]


def test_score_guess_marks_correct_present_and_miss() -> None:
    rt, _pygame = _compile_wordle_runtime()

    score_guess = rt._prelude_env["_score_guess"]

    assert score_guess("crane", "cigar") == [
        "correct",
        "miss",
        "miss",
        "present",
        "present",
    ]


def test_play_wordle_uses_llm_validation_and_renders_word_states() -> None:
    def keydown(key: int, unicode: str = "") -> object:
        return types.SimpleNamespace(type=1, key=key, unicode=unicode)

    event_batches = [
        [
            keydown(ord("c"), "c"),
            keydown(ord("i"), "i"),
            keydown(ord("g"), "g"),
            keydown(ord("a"), "a"),
            keydown(ord("r"), "r"),
            keydown(13, ""),
        ],
        [keydown(27, "")],
        [keydown(27, "")],
    ]
    rt, pygame = _compile_wordle_runtime(event_batches=event_batches)
    adapter = rt.adapter
    assert adapter is not None
    result = rt._invoke_procedure(rt._procedures["play_wordle"], ["space"])

    assert result == "Round ended. The word was CRANE."
    assert adapter.model_calls == ["mock-model", "mock-model"]
    board_fill_colors = [
        call["color"]
        for call in pygame._draw_calls
        if call["width"] == 0 and call["rect"].w == 72 and call["rect"].h == 72
    ]
    prelude = rt._prelude_env
    state_colors = prelude["STATE_COLORS"]
    assert state_colors["correct"] in board_fill_colors
    assert state_colors["present"] in board_fill_colors
    assert state_colors["miss"] in board_fill_colors


def test_play_wordle_can_reset_into_a_new_round() -> None:
    def keydown(key: int, unicode: str = "") -> object:
        return types.SimpleNamespace(type=1, key=key, unicode=unicode)

    event_batches = [
        [
            keydown(ord("c"), "c"),
            keydown(ord("r"), "r"),
            keydown(ord("a"), "a"),
            keydown(ord("n"), "n"),
            keydown(ord("e"), "e"),
            keydown(13, ""),
        ],
        [keydown(ord("r"), "")],
        [keydown(27, "")],
        [keydown(27, "")],
    ]
    rt, _pygame = _compile_wordle_runtime(event_batches=event_batches)
    adapter = rt.adapter
    assert adapter is not None

    result = rt._invoke_procedure(rt._procedures["play_wordle"], ["space"])

    assert result == "Round ended. The word was SLATE."
    assert adapter.model_calls == ["mock-model", "mock-model", "mock-model"]


def test_play_wordle_retries_when_model_repeats_the_same_answer() -> None:
    class RepeatingSetupAdapter(WordleModuleAdapter):
        def __init__(self) -> None:
            super().__init__(default_model="mock-model")
            self._setup_payloads = [
                {
                    "answer": "crane",
                    "theme_label": "Space Tools",
                    "clue": "Used in orbit.",
                    "intro": "Guess the five-letter word.",
                },
                {
                    "answer": "crane",
                    "theme_label": "Space Tools",
                    "clue": "Used in orbit.",
                    "intro": "Guess the five-letter word.",
                },
                {
                    "answer": "slate",
                    "theme_label": "Desk Objects",
                    "clue": "Seen near a notebook.",
                    "intro": "A fresh five-letter round.",
                },
            ]

    def keydown(key: int, unicode: str = "") -> object:
        return types.SimpleNamespace(type=1, key=key, unicode=unicode)

    event_batches = [
        [
            keydown(ord("c"), "c"),
            keydown(ord("r"), "r"),
            keydown(ord("a"), "a"),
            keydown(ord("n"), "n"),
            keydown(ord("e"), "e"),
            keydown(13, ""),
        ],
        [keydown(ord("r"), "")],
        [keydown(27, "")],
        [keydown(27, "")],
    ]
    adapter = RepeatingSetupAdapter()
    rt, _pygame = _compile_wordle_runtime(event_batches=event_batches, adapter=adapter)

    result = rt._invoke_procedure(rt._procedures["play_wordle"], ["space"])

    assert result == "Round ended. The word was SLATE."
    assert adapter.model_calls == ["mock-model", "mock-model", "mock-model", "mock-model"]
