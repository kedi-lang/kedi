"""An isolated Kedi pilot process; the main process owns Pygame and physics."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from typing import Any


@dataclass(frozen=True)
class Answer:
    action: str
    confidence: float | None
    seconds: float
    accepted: bool


class Turns:
    def __init__(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing
        self.started = 0.0

    def receive(self) -> str | None:
        state = self.incoming.get()
        self.started = time.monotonic()
        return state

    def send(self, decision) -> None:
        self.outgoing.put(
            Answer(
                decision.action,
                decision.confidence,
                time.monotonic() - self.started,
                decision.accepted,
            )
        )


def serve(source: Path, incoming, outgoing, model_id: str | None = None) -> None:
    try:
        import kedi
        from kedi.agent_adapter import PydanticAdapter

        @kedi.bind(file=source)
        def play(turns: Any) -> str: ...

        if model_id is not None:
            from kedi.agent_adapter.models import laya_model

            model = laya_model(model_id, "pydantic")
        else:
            from kedi_typesafe import TypeSafeModel

            model = TypeSafeModel(timeout=12)
        with kedi.context(adapter=PydanticAdapter(model), model=None):
            play(Turns(incoming, outgoing))
    except Exception as error:
        # Do not forward provider exceptions that may contain request credentials.
        outgoing.put(
            f"Pilot stopped ({type(error).__name__}). Check the model, input size and credentials."
        )


class Pilot:
    def __init__(self, source: Path, model_id: str | None = None):
        context = mp.get_context("spawn")
        self.incoming = context.Queue(maxsize=1)
        self.outgoing = context.Queue(maxsize=1)
        self.process = context.Process(
            target=serve, args=(source, self.incoming, self.outgoing, model_id), daemon=True
        )
        self.process.start()
        self.pending = False
        self.started = 0.0
        self.closed = False

    def request(self, observation: str) -> None:
        if self.closed:
            raise RuntimeError("Pilot is closed")
        if self.pending:
            raise RuntimeError("Only one decision may be in flight")
        self.incoming.put_nowait(observation)
        self.pending = True
        self.started = time.monotonic()

    def poll(self) -> Answer | str | None:
        if self.closed:
            return "Pilot is closed."
        try:
            answer = self.outgoing.get_nowait()
        except Empty:
            if not self.process.is_alive():
                return "Pilot exited before returning a decision."
            if self.pending and time.monotonic() - self.started > 25:
                self.close()
                return "Decision timed out. Simulation stopped."
            return None
        self.pending = False
        return answer

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pending = False
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.process.close()
        for queue in (self.incoming, self.outgoing):
            # The reader may have been stopped during a pending API request.
            queue.cancel_join_thread()
            queue.close()
