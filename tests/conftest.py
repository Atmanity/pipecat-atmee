"""Shared fixtures: a local fake of the Atmee API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web


@dataclass
class FakeAPI:
    """Records requests; answers from ``responses`` (path -> list of (status, body))."""

    url: str = ""
    requests: list[tuple[str, str, dict[str, Any] | None]] = field(default_factory=list)
    responses: dict[str, list[tuple[int, Any]]] = field(default_factory=dict)

    def respond(self, path: str, *answers: tuple[int, Any]) -> None:
        self.responses.setdefault(path, []).extend(answers)

    def calls(self, path: str) -> list[dict[str, Any] | None]:
        return [body for _, p, body in self.requests if p == path]


@pytest.fixture
async def fake_api(aiohttp_unused_port: Callable[[], int]) -> AsyncIterator[FakeAPI]:
    fake = FakeAPI()

    async def handler(request: web.Request) -> web.StreamResponse:
        body = None
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        fake.requests.append((request.method, request.path, body))
        queue = fake.responses.get(request.path)
        if not queue:
            return web.json_response({"error": "not_found", "message": "no fake"}, status=404)
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        headers = {"Retry-After": "5"} if status == 503 else None
        return web.Response(
            status=status,
            text=json.dumps(payload),
            content_type="application/json",
            headers=headers,
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = aiohttp_unused_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    fake.url = f"http://127.0.0.1:{port}"
    try:
        yield fake
    finally:
        await runner.cleanup()


@pytest.fixture
def aiohttp_unused_port() -> Callable[[], int]:
    import socket

    def _port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    return _port
