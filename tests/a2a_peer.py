"""Official-SDK suspension fixture that resumes into a real compiled Kedi agent."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from functools import partial

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

from kedi.a2a.auth import A2AServerAuth
from kedi.a2a.server import (
    _A2AAuthenticationBackend,
    _authentication_error,
    _agent_card,
    _KediAgentExecutor,
)
import socket


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class EmbeddedServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The test owns shutdown. Process-wide handlers would let one
        # embedded server's exit drain another server's active SSE connections.
        yield


class SuspensionPeer:
    def __init__(self, served, secret, state, events):
        self.served, self.secret, self.state, self.events = served, secret, state, events
        self.port = unused_port()
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.requests = []
        self.responses = []
        self.delegate = _KediAgentExecutor(served)
        self.server = self.running = None

    async def execute(self, context, event_queue):
        self.requests.append(
            {
                "task_id": context.task_id,
                "context_id": context.context_id,
                "input": context.get_user_input(),
                "continuation": context.current_task is not None,
                "submitted_before": sum(e["phase"] == "submitted" for e in self.events),
            }
        )
        if context.current_task is not None:
            await self.delegate.execute(context, event_queue)
            return
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            )
        )
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        message = updater.new_agent_message(
            [Part(text="Explicit confirmation is required before execution.")]
        )
        if self.state == "input_required":
            await updater.requires_input(message)
        else:
            await updater.requires_auth(message)

    async def cancel(self, context, event_queue):
        await self.delegate.cancel(context, event_queue)

    async def __aenter__(self):
        auth = A2AServerAuth(scheme="bearer", secret=self.secret)
        card = _agent_card(self.served, endpoint=self.endpoint, rpc_path="/a2a", auth=auth)
        self.handler = DefaultRequestHandler(
            agent_executor=self, task_store=InMemoryTaskStore(), agent_card=card
        )

        @asynccontextmanager
        async def lifespan(app):
            try:
                yield
            finally:
                try:
                    await self.handler.aclose()
                finally:
                    await self.served.aclose()

        app = Starlette(
            routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(self.handler, "/a2a")],
            middleware=[
                Middleware(
                    AuthenticationMiddleware,
                    backend=_A2AAuthenticationBackend(auth),
                    on_error=partial(_authentication_error, auth=auth),
                )
            ],
            lifespan=lifespan,
        )

        async def observed_app(scope, receive, send):
            if scope["type"] != "http":
                return await app(scope, receive, send)
            response = {"path": scope["path"], "complete": False, "disconnected": False}
            self.responses.append(response)

            async def observed_receive():
                message = await receive()
                if message["type"] == "http.disconnect":
                    response["disconnected"] = True
                return message

            async def observed_send(message):
                await send(message)
                if message["type"] == "http.response.start":
                    response["status"] = message["status"]
                elif message["type"] == "http.response.body" and not message.get(
                    "more_body", False
                ):
                    response["complete"] = True

            try:
                await app(scope, observed_receive, observed_send)
            except BaseException as exc:
                response["error"] = type(exc).__name__
                raise

        self.server = EmbeddedServer(
            uvicorn.Config(
                observed_app, host="127.0.0.1", port=self.port, log_level="error", access_log=False
            )
        )
        self.running = asyncio.create_task(self.server.serve())

        async def wait_started():
            while not self.server.started:
                if self.running.done():
                    await self.running
                    raise RuntimeError("Continuation peer did not start")
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(wait_started(), 10)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc):
        if self.server is not None:
            self.server.should_exit = True
        if self.running is not None:
            await asyncio.wait_for(self.running, 25)
