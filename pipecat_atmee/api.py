# Copyright 2026 Atmanity
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin async client for the Atmee API.

Two resources matter here: **avatars** — Atmee v1 avatars, talking heads
generated from a single portrait — (create one from a portrait, read its
status) and **avatar sessions** (render an avatar for your bot, end it).
Authentication is your Atmee API key (``sk_atmee_...``) in the ``X-Api-Key``
header; keep it on your bot's side, it never belongs in a browser.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp
from loguru import logger

DEFAULT_API_URL = "https://api.atmanity.us"

WaitFor = Literal["initializing", "avatar_joined"]

# Overall budget of the start call: the renderer acknowledges within seconds,
# but waiting for the avatar to join also covers model priming on a cold
# machine plus the room join.
_START_TOTAL_TIMEOUT: dict[str, float] = {"initializing": 60.0, "avatar_joined": 180.0}
_DEFAULT_TOTAL_TIMEOUT = 120.0
# Ending a session runs on pipeline stop/cancel: bound each attempt and the
# whole call (retries included) so teardown cannot hang for minutes.
_END_ATTEMPT_TIMEOUT = 10.0
_END_TOTAL_TIMEOUT = 30.0


@dataclass(frozen=True)
class ConnectOptions:
    """Retry and timeout policy of :class:`AtmeeAPI`.

    Parameters:
        max_retry: Retries after the first attempt for transport errors and
            5xx answers of idempotent calls (never for creates).
        retry_interval: Pause between attempts, in seconds.
        timeout: TCP connect timeout of one attempt, in seconds.
    """

    max_retry: int = 3
    retry_interval: float = 2.0
    timeout: float = 10.0


DEFAULT_CONNECT_OPTIONS = ConnectOptions()


class AtmeeException(Exception):
    """An Atmee API call failed.

    ``status_code`` is the HTTP status (0 when the request never got an
    answer), ``code`` the machine-readable ``error`` field of the response
    body when the API sent one (``avatar_not_renderable``,
    ``insufficient_credits``, ...), and ``message`` its human-readable text.
    """

    def __init__(self, message: str, *, status_code: int = 0, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        return " ".join(parts)


class AtmeeNoCapacityError(AtmeeException):
    """Every rendering machine is busy (HTTP 503 ``no_capacity``).

    Never retried by the client: a render slot frees up on the order of
    seconds to minutes, not the milliseconds a transport retry covers.
    ``retry_after`` is the server's hint in seconds, when it sent one.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=503, code="no_capacity")
        self.retry_after = retry_after


class AtmeeAvatarNotReadyError(AtmeeException):
    """The avatar cannot be rendered yet (HTTP 409 ``avatar_not_renderable``):
    it has no portrait, or its appearance is still being processed."""


@dataclass
class AvatarSessionInfo:
    """The ``202`` body of ``POST /v1/avatars/{avatarId}/avatar_sessions``.

    ``livekit_url`` and ``agent_token`` are set for an Atmee-hosted room (no
    LiveKit credentials in the request): the media room the bot joins to
    exchange audio and video with the avatar.
    """

    session_id: str
    status: str
    avatar_participant_identity: str
    agent_identity: str
    room_name: str
    max_duration_seconds: int
    billing_mode: str
    livekit_url: str | None = None
    agent_token: str | None = field(default=None, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AvatarInfo:
    """What ``GET /v1/avatars/{avatarId}`` (and a create) report about an avatar.

    ``kind`` (``conversational`` | ``render_only``) says whether the avatar has
    a voice and persona of its own.
    """

    avatar_id: str
    status: str
    kind: str | None = None
    name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        """Whether the avatar can be rendered."""
        return self.status == "ready"


_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class AtmeeAPI:
    """Async client for the Atmee API.

    Args:
        api_key: Your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
        api_url: API base; defaults to ``ATMEE_API_URL`` or ``https://api.atmanity.us``.
        conn_options: Retry/timeout policy for transport failures and 5xx
            answers (a 503 ``no_capacity`` is never retried).
        session: An ``aiohttp.ClientSession`` to reuse. Without one a private
            session is created on first use and closed by :meth:`aclose`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        *,
        conn_options: ConnectOptions = DEFAULT_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the client (see the class docstring for the arguments)."""
        key = api_key or os.getenv("ATMEE_API_KEY")
        if not key:
            raise AtmeeException(
                "ATMEE_API_KEY must be set (or pass api_key=...); create one at "
                "https://www.atmee.ai/studio/api-keys"
            )
        self._api_key = key
        self._api_url = (api_url or os.getenv("ATMEE_API_URL") or DEFAULT_API_URL).rstrip("/")
        _check_api_url(self._api_url)
        self._conn_options = conn_options
        self._session = session
        self._owns_session = False

    @property
    def api_url(self) -> str:
        """The API base URL in use."""
        return self._api_url

    # --- avatar sessions -------------------------------------------------

    async def create_avatar_session(
        self,
        avatar_id: str,
        *,
        livekit_url: str | None = None,
        livekit_token: str | None = None,
        agent_identity: str | None = None,
        max_duration_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        wait_for: WaitFor = "initializing",
    ) -> AvatarSessionInfo:
        """Start rendering ``avatar_id``.

        Without ``livekit_url``/``livekit_token`` Atmee hosts the media room
        and the answer carries ``livekit_url`` and ``agent_token`` for the bot
        to join it as ``agent_identity``. With them, the avatar joins the
        LiveKit room ``livekit_token`` grants: a token you minted with YOUR
        project's secret for the avatar participant (``kind=agent``, a
        ``roomJoin`` grant, and the ``lk.publish_on_behalf`` attribute naming
        your bot's identity).

        Returns as soon as the renderer acknowledged the start, or once the
        avatar joined with ``wait_for="avatar_joined"``. Never retried: the
        call reserves and bills a session as soon as the server accepts it.
        """
        if bool(livekit_url) != bool(livekit_token):
            raise ValueError("livekit_url and livekit_token must be given together")
        payload: dict[str, Any] = {}
        if livekit_url and livekit_token:
            payload["livekitUrl"] = livekit_url
            payload["livekitToken"] = livekit_token
        if agent_identity:
            payload["agentIdentity"] = agent_identity
        if max_duration_seconds is not None:
            payload["maxDurationSeconds"] = int(max_duration_seconds)
        if metadata:
            payload["metadata"] = metadata
        params: dict[str, str] | None = (
            {"waitFor": wait_for} if wait_for != "initializing" else None
        )
        data = await self._request(
            "POST",
            f"/v1/avatars/{avatar_id}/avatar_sessions",
            json=payload,
            params=params,
            retry=False,
            total_timeout=_START_TOTAL_TIMEOUT[wait_for],
        )
        session_id = data.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AtmeeException(
                "the Atmee API accepted the avatar session but returned no sessionId",
                code="invalid_response",
            )
        info = AvatarSessionInfo(
            session_id=session_id,
            status=str(data.get("status", "")),
            avatar_participant_identity=str(data.get("avatarParticipantIdentity", "")),
            agent_identity=str(data.get("agentIdentity", "")),
            room_name=str(data.get("roomName", "")),
            max_duration_seconds=int(data.get("maxDurationSeconds") or 0),
            billing_mode=str(data.get("billingMode", "")),
            livekit_url=_str_or_none(data.get("livekitUrl")),
            agent_token=_str_or_none(data.get("agentToken")),
            raw={k: v for k, v in data.items() if k != "agentToken"},
        )
        if not livekit_url and not (info.livekit_url and info.agent_token):
            # Hosted room requested but the answer carries no way into it: the
            # session exists and bills, so the caller must end it.
            raise AtmeeHostedRoomMissingError(info)
        return info

    async def end_avatar_session(self, session_id: str) -> dict[str, Any]:
        """Stop billing for a session now.

        Idempotent: ending an ended session answers normally with
        ``alreadyEnded: true``.
        """
        return await self._request(
            "POST",
            f"/v1/avatar_sessions/{session_id}/end",
            total_timeout=_END_ATTEMPT_TIMEOUT,
            deadline=asyncio.get_running_loop().time() + _END_TOTAL_TIMEOUT,
        )

    async def get_avatar_session(self, session_id: str) -> dict[str, Any]:
        """Read a session's status (``initializing``, ``active``, ``ended``, ...)."""
        return await self._request("GET", f"/v1/avatar_sessions/{session_id}")

    # --- avatars -----------------------------------------------------------

    async def create_avatar(
        self,
        name: str,
        image: str | Path | bytes,
        *,
        description: str | None = None,
        content_type: str | None = None,
    ) -> str:
        """Create a render-only **v1 avatar** from a single portrait and return its id.

        ``image`` is a local file path, the image bytes, or an ``https://``
        URL Atmee can download (``.jpg``, ``.png``, ``.webp``). A
        portrait-only avatar is ready at once; nothing to poll.
        """
        info = await self.create_avatar_info(
            name, image, description=description, content_type=content_type
        )
        return info.avatar_id

    async def create_avatar_info(
        self,
        name: str,
        image: str | Path | bytes,
        *,
        description: str | None = None,
        content_type: str | None = None,
    ) -> AvatarInfo:
        """Like :meth:`create_avatar`, returning the whole response."""
        if isinstance(image, str) and _is_url(image):
            manifest: dict[str, Any] = {
                "schemaVersion": 1,
                "name": name,
                "assets": {"image": {"url": image}},
            }
            if description:
                manifest["description"] = description
            data = await self._request("POST", "/v1/avatars", json=manifest, retry=False)
            return _avatar_info(data)

        if isinstance(image, bytes | bytearray):
            blob = bytes(image)
            filename = "portrait" + _extension_for(content_type) if content_type else "portrait.jpg"
            ctype = content_type or "image/jpeg"
        else:
            blob, filename, ctype = await asyncio.to_thread(_load_image_file, image, content_type)
        form = aiohttp.FormData()
        form.add_field("name", name)
        if description:
            form.add_field("description", description)
        form.add_field("file", blob, filename=filename, content_type=ctype)
        data = await self._request("POST", "/v1/avatars", data=form, retry=False)
        return _avatar_info(data)

    async def get_avatar(self, avatar_id: str) -> AvatarInfo:
        """Read an avatar."""
        return _avatar_info(await self._request("GET", f"/v1/avatars/{avatar_id}"))

    # --- plumbing ------------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        self._session = aiohttp.ClientSession()
        self._owns_session = True
        return self._session

    async def aclose(self) -> None:
        """Close the HTTP session if this client created it."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        if self._owns_session:
            self._session = None
        self._owns_session = False

    async def __aenter__(self) -> AtmeeAPI:
        """Use the client as an async context manager."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the client's own HTTP session."""
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        data: aiohttp.FormData | None = None,
        params: dict[str, str] | None = None,
        retry: bool = True,
        total_timeout: float = _DEFAULT_TOTAL_TIMEOUT,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """One API call with the retry policy.

        Transport errors and 5xx answers are retried up to
        ``conn_options.max_retry`` times after the first attempt; a 503
        ``no_capacity`` and every 4xx are final. ``retry=False`` for calls that
        are not idempotent. ``deadline`` (event-loop time) bounds the whole
        call, retries and the pauses between them included.
        """
        loop = asyncio.get_running_loop()
        attempts = self._conn_options.max_retry + 1 if retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_timeout = total_timeout
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AtmeeException(f"{method} {path} timed out", code="timeout") from (
                        last_error
                    )
                attempt_timeout = min(total_timeout, remaining)
            try:
                async with self._ensure_session().request(
                    method,
                    f"{self._api_url}{path}",
                    headers={"X-Api-Key": self._api_key, "Accept": "application/json"},
                    json=json,
                    data=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(
                        sock_connect=self._conn_options.timeout, total=attempt_timeout
                    ),
                ) as response:
                    if response.ok:
                        if response.status == 204 or response.content_length == 0:
                            return {}
                        try:
                            payload = await response.json(content_type=None)
                        except ValueError as e:
                            raise AtmeeException(
                                "the Atmee API returned an invalid JSON response",
                                status_code=response.status,
                                code="invalid_response",
                            ) from e
                        return payload if isinstance(payload, dict) else {"data": payload}
                    raise await _error_from_response(response)
            except AtmeeException as e:
                if e.status_code >= 500 and not isinstance(e, AtmeeNoCapacityError):
                    last_error = e
                else:
                    raise
            except (TimeoutError, aiohttp.ClientError) as e:
                last_error = AtmeeException(f"could not reach the Atmee API: {e}")
            if attempt < attempts - 1:
                logger.warning(f"Atmee API call {method} {path} failed, retrying: {last_error}")
                pause = self._conn_options.retry_interval
                if deadline is not None:
                    pause = min(pause, max(0.0, deadline - loop.time()))
                await asyncio.sleep(pause)
        assert last_error is not None
        raise last_error


class AtmeeHostedRoomMissingError(AtmeeException):
    """The API accepted a hosted-room session but sent no room to join.

    The session exists (``info.session_id``) and must be ended.
    """

    def __init__(self, info: AvatarSessionInfo) -> None:
        super().__init__(
            "the Atmee API did not return a hosted room (livekitUrl/agentToken); "
            "this API version may not support hosted rooms yet",
            code="hosted_room_unavailable",
        )
        self.info = info


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _check_api_url(url: str) -> None:
    """Refuse a plaintext API base: the API key travels in every request.

    Plain ``http`` is allowed only for a loopback host (local development, tests).
    """
    parsed = urlsplit(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return
    raise AtmeeException(
        f"api_url must be an https:// URL (plain http only for localhost), got {url!r}"
    )


async def _error_from_response(response: aiohttp.ClientResponse) -> AtmeeException:
    text = await response.text()
    code: str | None = None
    message = text.strip() or response.reason or f"HTTP {response.status}"
    try:
        body = await response.json(content_type=None)
    except Exception:
        body = None
    if isinstance(body, dict):
        code = body.get("error") if isinstance(body.get("error"), str) else None
        message = str(body.get("message") or body.get("detail") or message)
    # Only the API's own no_capacity answer is final; a bare 503 (a gateway or
    # load balancer) is transient and retried like any other 5xx.
    if response.status == 503 and code == "no_capacity":
        retry_after: float | None = None
        header = response.headers.get("Retry-After")
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None
        return AtmeeNoCapacityError(message, retry_after=retry_after)
    if response.status == 409 and code == "avatar_not_renderable":
        return AtmeeAvatarNotReadyError(message, status_code=409, code=code)
    return AtmeeException(message, status_code=response.status, code=code)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _avatar_info(data: dict[str, Any]) -> AvatarInfo:
    return AvatarInfo(
        avatar_id=str(data.get("avatarId") or data.get("id") or ""),
        status=str(data.get("status", "")),
        kind=data.get("kind"),
        name=data.get("name"),
        raw=data,
    )


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _load_image_file(image: str | Path, content_type: str | None) -> tuple[bytes, str, str]:
    path = Path(image)
    ctype = (
        content_type
        or _IMAGE_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )
    return path.read_bytes(), path.name, ctype


def _extension_for(content_type: str) -> str:
    for ext, ctype in _IMAGE_TYPES.items():
        if ctype == content_type and ext != ".jpeg":
            return ext
    return mimetypes.guess_extension(content_type) or ""
