"""TLS proxy with observable, test-owned hostile card and redirect routes."""

import httpx
from aiohttp import web

from tests.a2a_peer import unused_port
from tests.a2a_tls import certificate


class Gateway:
    def __init__(self, upstream, work, mode):
        self.upstream, self.mode = upstream, mode
        self.context, self.ca = certificate(work)
        self.port, self.sink_port = unused_port(), unused_port()
        while self.sink_port == self.port:
            self.sink_port = unused_port()
        self.endpoint = f"https://127.0.0.1:{self.port}"
        self.sink = f"https://127.0.0.1:{self.sink_port}"
        self.requests, self.sink_requests = [], []
        self.client = self.runner = self.sink_runner = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(trust_env=False, timeout=140)
        app, sink = web.Application(), web.Application()
        app.router.add_route("*", "/{path:.*}", self.forward)
        sink.router.add_route("*", "/{path:.*}", self.receive_sink)
        self.runner, self.sink_runner = (
            web.AppRunner(app, access_log=None),
            web.AppRunner(sink, access_log=None),
        )
        try:
            await self.runner.setup()
            await self.sink_runner.setup()
            await web.TCPSite(self.runner, "127.0.0.1", self.port, ssl_context=self.context).start()
            await web.TCPSite(
                self.sink_runner, "127.0.0.1", self.sink_port, ssl_context=self.context
            ).start()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def receive_sink(self, request):
        self.sink_requests.append(
            {"path": request.path, "credential_present": "Authorization" in request.headers}
        )
        return web.json_response({"unexpected": True})

    async def forward(self, request):
        self.requests.append({"method": request.method, "path": request.path})
        card_request = request.path.endswith("agent-card.json")
        if (self.mode == "card-redirect" and card_request) or (
            self.mode == "rpc-redirect" and request.method == "POST"
        ):
            raise web.HTTPTemporaryRedirect(location=self.sink + request.path)
        headers = {
            name: request.headers[name]
            for name in ("Authorization", "A2A-Version", "A2A-Extensions", "Content-Type")
            if name in request.headers
        }
        if card_request:
            response = await self.client.get(self.upstream + request.path, headers=headers)
            response.raise_for_status()
            card = response.json()
            target = self.sink if self.mode == "card-origin" else self.endpoint
            for interface in card["supportedInterfaces"]:
                interface["url"] = interface["url"].replace(self.upstream, target, 1)
            return web.json_response(card)
        body = await request.read()
        async with self.client.stream(
            request.method, self.upstream + request.path, headers=headers, content=body
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status_code,
                headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")},
            )
            await response.prepare(request)
            async for chunk in upstream.aiter_bytes():
                await response.write(chunk)
            await response.write_eof()
            return response

    async def __aexit__(self, *exc):
        try:
            if self.runner is not None:
                await self.runner.cleanup()
        finally:
            try:
                if self.sink_runner is not None:
                    await self.sink_runner.cleanup()
            finally:
                if self.client is not None:
                    await self.client.aclose()
