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

"""Atmee avatar that joins your bot's Daily room directly.

:class:`AtmeeDailyVideoService` is the Daily counterpart of
:class:`~pipecat_atmee.AtmeeVideoService`. Instead of rendering into a
separate media room and relaying the video through your bot, the avatar joins
your Daily room as its own participant and publishes its lip-synced video and
audio there. Your bot only sends it the speech, as Daily app messages; it
never receives or re-encodes any video.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from typing import Any

import aiohttp
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_atmee.api import (
    DEFAULT_CONNECT_OPTIONS,
    AtmeeException,
    ConnectOptions,
)
from pipecat_atmee.video import (
    AVATAR_SAMPLE_RATE,
    DEFAULT_AVATAR_JOIN_TIMEOUT,
    DEFAULT_AVATAR_WARMUP_TIMEOUT,
    DEFAULT_MAX_DURATION_SECONDS,
    AtmeeVideoService,
    AtmeeVideoSettings,
    _downmix,
)

try:
    from pipecat.transports.daily.transport import (
        DailyOutputTransportMessageUrgentFrame,
        DailyTransport,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(
        f"AtmeeDailyVideoService needs Pipecat's Daily transport (pipecat-ai[daily]): {e}"
    ) from e

# App messages to and from the avatar carry their type under this key.
MESSAGE_KEY = "atmee"
DEFAULT_AVATAR_NAME = "Atmee avatar"
DAILY_API_URL = "https://api.daily.co/v1"
# Speech goes to the avatar in chunks of this many milliseconds (16 kHz mono
# 16-bit PCM, base64): small enough for a Daily app message.
CHUNK_MS = 100
_CHUNK_BYTES = AVATAR_SAMPLE_RATE * 2 * CHUNK_MS // 1000
# How long the start waits for the bot to have joined its Daily room.
_JOIN_WAIT_SECONDS = 60.0
# While the avatar has not said it is ready, ask again this often.
_HELLO_INTERVAL_SECONDS = 2.0
# The avatar's meeting token outlives the session ceiling by this much.
_TOKEN_TTL_SLACK_SECONDS = 600


class AtmeeDailyVideoService(AtmeeVideoService):
    """Give your Pipecat bot on Daily a face: the Atmee avatar joins the room.

    Place it right after the TTS service, and pass it the bot's
    ``DailyTransport``. When the bot has joined its room, the service starts
    an Atmee avatar session for that room; the avatar joins as its own
    participant (named ``avatar_name``) and publishes its lip-synced video and
    audio. The bot's TTS audio is sent to the avatar and also passed on
    unchanged, so the pipeline still sees the bot speaking: turn the bot's own
    microphone off (``DailyParams(microphone_out_enabled=False)``) so users
    hear only the avatar.

    Interruptions stop the avatar mid-sentence. The session ends, and stops
    billing, when the pipeline ends or is cancelled; the avatar then leaves
    the room.

    For a private room the avatar needs a meeting token: the service mints one
    with ``daily_api_key`` (default ``DAILY_API_KEY``), the key your bot's
    rooms are created with.

    Event handlers are those of :class:`~pipecat_atmee.AtmeeVideoService`:
    ``on_avatar_connected`` fires once the avatar is in the room and ready.
    """

    def __init__(
        self,
        *,
        transport: DailyTransport,
        avatar_id: str,
        api_key: str | None = None,
        api_url: str | None = None,
        daily_api_key: str | None = None,
        avatar_name: str = DEFAULT_AVATAR_NAME,
        session: aiohttp.ClientSession | None = None,
        max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
        metadata: dict[str, Any] | None = None,
        avatar_join_timeout: float = DEFAULT_AVATAR_JOIN_TIMEOUT,
        avatar_warmup_timeout: float = DEFAULT_AVATAR_WARMUP_TIMEOUT,
        conn_options: ConnectOptions = DEFAULT_CONNECT_OPTIONS,
        settings: AtmeeVideoSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the service.

        Args:
            transport: The bot's ``DailyTransport``.
            avatar_id: The Atmee avatar to render.
            api_key: Your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
            api_url: API base; defaults to ``ATMEE_API_URL`` or the Atmee cloud.
            daily_api_key: Daily API key to mint the avatar's meeting token
                with; defaults to ``DAILY_API_KEY``. Without one the avatar
                joins without a token, which only a public room allows.
            avatar_name: The avatar's display name in the room.
            session: An aiohttp session to reuse for API calls.
            max_duration_seconds: Hard ceiling of the session (default 3600).
            metadata: Free-form JSON attached to the session on the Atmee side.
            avatar_join_timeout: How long the first bot speech waits, once the
                session has started, for the avatar to be ready.
            avatar_warmup_timeout: How long the start keeps retrying while a
                v2 avatar is being prepared on a rendering machine.
            conn_options: Retry/timeout policy of the API client.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``AIService``.
        """
        super().__init__(
            avatar_id=avatar_id,
            api_key=api_key,
            api_url=api_url,
            session=session,
            max_duration_seconds=max_duration_seconds,
            metadata=metadata,
            avatar_join_timeout=avatar_join_timeout,
            avatar_warmup_timeout=avatar_warmup_timeout,
            conn_options=conn_options,
            settings=settings,
            **kwargs,
        )
        self._transport = transport
        self._daily_api_key = daily_api_key or os.getenv("DAILY_API_KEY")
        self._avatar_name = avatar_name
        # The avatar's Daily participant id, once it said it is ready.
        self._avatar_participant: str | None = None
        self._segment: str | None = None
        self._pending = bytearray()
        self._outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._send_task: asyncio.Task | None = None
        self._hello_task: asyncio.Task | None = None

        transport.add_event_handler("on_app_message", self._on_app_message)
        transport.add_event_handler("on_participant_left", self._on_participant_left)

    # --- start -----------------------------------------------------------------

    async def _start_session(self) -> None:
        participant_id = await self._bot_participant_id()
        room_url = self._transport.room_url
        token = await self._mint_daily_token(room_url) if self._daily_api_key else None
        self._send_task = self.create_task(self._send_loop(), "atmee_daily_send")
        info = await self._create_session(
            daily_room_url=room_url,
            daily_token=token,
            daily_participant_id=participant_id,
            max_duration_seconds=self._max_duration_seconds,
            metadata=self._metadata,
        )
        self._remember(info)
        await self._end_if_closing()
        logger.info(
            f"{self}: Atmee avatar session {self.session_id} started in {room_url} "
            f"(billing {info.billing_mode}, ceiling {info.max_duration_seconds}s)"
        )
        self._hello_task = self.create_task(self._hello_loop(), "atmee_daily_hello")

    async def _bot_participant_id(self) -> str:
        """The bot's Daily participant id, once it has joined its room."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _JOIN_WAIT_SECONDS
        while not self._transport.participant_id:
            if self._closing:
                raise AtmeeException("the pipeline stopped before the bot joined its Daily room")
            if loop.time() > deadline:
                raise AtmeeException(
                    f"the bot did not join its Daily room within {_JOIN_WAIT_SECONDS:.0f}s"
                )
            await asyncio.sleep(0.1)
        return self._transport.participant_id

    async def _mint_daily_token(self, room_url: str) -> str:
        room_name = room_url.rstrip("/").rsplit("/", 1)[-1]
        properties = {
            "room_name": room_name,
            "user_name": self._avatar_name,
            "exp": int(time.time()) + self._max_duration_seconds + _TOKEN_TTL_SLACK_SECONDS,
        }
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(
                f"{DAILY_API_URL}/meeting-tokens",
                headers={"Authorization": f"Bearer {self._daily_api_key}"},
                json={"properties": properties},
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 300 or not isinstance(data, dict) or "token" not in data:
                    raise AtmeeException(
                        f"minting the avatar's Daily meeting token failed: {resp.status} {data}"
                    )
                return str(data["token"])

    async def _hello_loop(self) -> None:
        """Ask the avatar to say it is ready until it does (its first ``ready``
        may have been sent before it could reach us)."""
        while not self._avatar_ready.is_set() and not self._closing:
            self._send_now({MESSAGE_KEY: "hello"}, broadcast=True)
            await asyncio.sleep(_HELLO_INTERVAL_SECONDS)

    # --- avatar messages ------------------------------------------------------

    async def _on_app_message(self, _transport: Any, message: Any, sender: str) -> None:
        if not isinstance(message, dict) or self._closing:
            return
        kind = message.get(MESSAGE_KEY)
        if kind == "ready":
            if self._avatar_participant not in (None, sender):
                return
            self._avatar_participant = sender
            if not self._avatar_ready.is_set():
                self._avatar_wait_failed = False
                self._avatar_ready.set()
                logger.debug(f"{self}: the Atmee avatar is in the room ({sender})")
                await self._call_event_handler("on_avatar_connected", self.session_id)
        elif sender != self._avatar_participant:
            return
        elif kind == "playback_started":
            await self.stop_ttfb_metrics()
        elif kind == "playback_finished":
            if self._segment is None and time.monotonic() >= self._playout_end:
                self._avatar_busy = False
            await self._call_event_handler(
                "on_playback_finished",
                float(message.get("position") or 0.0),
                bool(message.get("interrupted")),
            )

    async def _on_participant_left(self, _transport: Any, participant: Any, _reason: Any) -> None:
        if (
            self._avatar_participant is not None
            and isinstance(participant, dict)
            and participant.get("id") == self._avatar_participant
            and not self._closing
        ):
            await self._handle_avatar_left()

    # --- frames -------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Send the bot's speech to the avatar and pass every frame on.

        Args:
            frame: The frame to process.
            direction: The direction of the frame.
        """
        await super(AtmeeVideoService, self).process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, TTSStoppedFrame):
            await self._end_utterance()
        elif isinstance(frame, InterruptionFrame):
            await self._interrupt()
        await self.push_frame(frame, direction)

    async def _write(self, pcm: bytes) -> None:
        """Queue 16 kHz speech for the avatar in ``CHUNK_MS`` app messages."""
        if self._segment is None:
            self._segment = uuid.uuid4().hex
        self._pending.extend(pcm)
        while len(self._pending) >= _CHUNK_BYTES:
            self._queue_audio(bytes(self._pending[:_CHUNK_BYTES]))
            del self._pending[:_CHUNK_BYTES]
        self._avatar_busy = True
        now = time.monotonic()
        self._playout_end = max(self._playout_end, now) + len(pcm) / (2 * AVATAR_SAMPLE_RATE)

    def _queue_audio(self, pcm: bytes) -> None:
        self._outbox.put_nowait(
            {
                MESSAGE_KEY: "audio",
                "segment": self._segment,
                "audio": base64.b64encode(pcm).decode("ascii"),
                "sample_rate": AVATAR_SAMPLE_RATE,
            }
        )

    async def _send_audio(self, frame: TTSAudioRawFrame) -> None:
        # Never waits: the speech is queued, and the send loop holds it until
        # the avatar is ready, so the pipeline keeps flowing meanwhile.
        if self._closing or self._avatar_wait_failed:
            return
        if self._start_done.is_set() and self._send_task is None:
            return  # the start failed (already reported)
        try:
            pcm = frame.audio
            if frame.num_channels != 1:
                pcm = _downmix(pcm, frame.num_channels)
            if frame.sample_rate != self._resampler_in_rate:
                self._resampler = create_stream_resampler()
                self._resampler_in_rate = frame.sample_rate
            pcm = await self._resampler.resample(pcm, frame.sample_rate, AVATAR_SAMPLE_RATE)
            await self._write(pcm)
        except Exception as e:
            await self.push_error(f"sending audio to the Atmee avatar failed: {e}", exception=e)

    async def _end_utterance(self) -> None:
        if self._segment is None or self._closing:
            return
        flush = getattr(self._resampler, "flush", None)  # absent in older pipecat
        if flush is not None:
            try:
                tail = await flush()
                if tail:
                    self._pending.extend(tail)
            except Exception as e:
                logger.warning(f"{self}: flushing the resampler failed: {e}")
        if self._pending:
            self._queue_audio(bytes(self._pending))
            self._pending.clear()
        self._outbox.put_nowait({MESSAGE_KEY: "audio_end", "segment": self._segment})
        self._segment = None

    async def _interrupt(self) -> None:
        if not self._avatar_ready.is_set() or not self._avatar_busy:
            return
        self._avatar_busy = False
        self._playout_end = 0.0
        self._segment = None
        self._pending.clear()
        # Speech not yet sent is dropped here; what the avatar already has is
        # dropped there.
        while not self._outbox.empty():
            self._outbox.get_nowait()
        self._send_now({MESSAGE_KEY: "clear_buffer"})

    # --- sending --------------------------------------------------------------------

    async def _send_loop(self) -> None:
        """Send queued messages to the avatar in order, once it is ready."""
        while True:
            message = await self._outbox.get()
            if not self._avatar_ready.is_set():
                if self._avatar_wait_failed:
                    continue  # already reported: drop until the avatar shows up
                # avatar_join_timeout counts from a started session.
                await self._wait_or_closed(self._start_done, None)
                if not await self._wait_or_closed(self._avatar_ready, self._avatar_join_timeout):
                    if self._closing:
                        return
                    self._avatar_wait_failed = True
                    while not self._outbox.empty():
                        self._outbox.get_nowait()
                    await self.push_error(
                        f"the Atmee avatar was not ready after {self._avatar_join_timeout:.0f}s; "
                        "dropping bot speech"
                    )
                    continue
            await self._send(message, self._avatar_participant)

    def _send_now(self, message: dict[str, Any], *, broadcast: bool = False) -> None:
        """Send out of band (``clear_buffer``, ``hello``, ``end``), ahead of
        queued speech."""
        target = None if broadcast else self._avatar_participant
        if target is None and not broadcast:
            return
        self.create_task(self._send(message, target), "atmee_daily_control")

    async def _send(self, message: dict[str, Any], participant_id: str | None) -> None:
        try:
            await self._transport.output().send_message(
                DailyOutputTransportMessageUrgentFrame(
                    message=message, participant_id=participant_id
                )
            )
        except Exception as e:
            logger.warning(f"{self}: sending to the Atmee avatar failed: {e}")

    # --- teardown -------------------------------------------------------------------

    async def _teardown(self) -> None:
        async with self._teardown_lock:
            first = not self._closing
            self._closing = True
            self._closed.set()
            if first and self._avatar_participant is not None:
                # The avatar leaves the room at once; ending the session below
                # stops the billing either way.
                await self._send({MESSAGE_KEY: "end"}, self._avatar_participant)
            await self._abort_start()
            await self._end_session()
            for task in (self._hello_task, self._send_task):
                if task is not None:
                    await self.cancel_task(task)
            self._hello_task = self._send_task = None
            self._avatar_ready.clear()
