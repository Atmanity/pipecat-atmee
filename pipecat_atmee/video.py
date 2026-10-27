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

"""Atmee avatar service for Pipecat.

:class:`AtmeeVideoService` sits between your TTS and your transport's output.
It sends the bot's speech to an Atmee avatar and pushes the avatar's
lip-synced video and audio back into the pipeline, so the avatar works with
any Pipecat transport (Daily, SmallWebRTC, LiveKit, ...).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import aiohttp
import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings

from pipecat_atmee._datastream import DataStreamAudioSender
from pipecat_atmee.api import (
    DEFAULT_CONNECT_OPTIONS,
    AtmeeAPI,
    AtmeeAvatarWarmingError,
    AtmeeException,
    AtmeeHostedRoomMissingError,
    AvatarSessionInfo,
    ConnectOptions,
)

try:
    from livekit import api as lk_api
    from livekit import rtc
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(f"pipecat-atmee needs the livekit SDK: {e}") from e

# The avatar consumes 16 kHz mono PCM; the bot's TTS audio is resampled to it.
AVATAR_SAMPLE_RATE = 16000
DEFAULT_AVATAR_IDENTITY = "atmee-avatar-agent"
DEFAULT_AGENT_IDENTITY = "pipecat-agent"
DEFAULT_MAX_DURATION_SECONDS = 3600
# How long the first bot audio waits for the avatar to be in the room.
# Covers a v2 avatar on a cold rendering machine, which takes a few minutes.
DEFAULT_AVATAR_JOIN_TIMEOUT = 330.0
# How long the start keeps retrying while a v2 avatar is being prepared on a
# rendering machine (503 avatar_warming), and the pause when the API sends no
# Retry-After.
DEFAULT_AVATAR_WARMUP_TIMEOUT = 300.0
_WARMING_RETRY_SECONDS = 30.0
# On stop/cancel, a start still waiting for the API gets this long to return
# the session it created (so it can be ended) before it is cancelled.
_START_ABORT_GRACE_SECONDS = 15.0


class _StartAborted(Exception):
    """The pipeline stopped while the avatar session was still starting."""


# The avatar's token outlives the session ceiling by this much.
_TOKEN_TTL_SLACK = timedelta(minutes=10)
# After an interruption the avatar's audio already on the wire (its playout
# buffer plus WebRTC jitter, measured ~0.4 s) is dropped for this long. A new
# reply cannot arrive sooner: it still needs LLM, TTS and the render.
INTERRUPT_MUTE_SECONDS = 0.6
_PUBLISH_ON_BEHALF = "lk.publish_on_behalf"


@dataclass
class AtmeeVideoSettings(ServiceSettings):
    """Runtime settings of :class:`AtmeeVideoService` (none beyond the base ones)."""


class AtmeeVideoService(AIService):
    """Give your Pipecat bot a face with an Atmee avatar.

    Place it right after the TTS service. It starts an Atmee avatar session
    when the pipeline starts, streams the TTS audio to the avatar (the TTS
    audio frames are consumed, the avatar's own lip-synced audio replaces
    them) and pushes the avatar's video as ``OutputImageRawFrame`` and its
    audio as ``SpeechOutputAudioRawFrame`` downstream to the transport.
    Interruptions stop the avatar mid-sentence. The session ends, and stops
    billing, when the pipeline ends or is cancelled.

    Any avatar of your account works, v1 or v2. A v2 avatar on a cold
    rendering machine can take a few minutes to start: the service waits
    (``avatar_warmup_timeout``, ``avatar_join_timeout``) instead of failing.

    By default Atmee hosts the media room the avatar is rendered into; only an
    Atmee API key is needed. Pass ``livekit_url``/``livekit_api_key``/
    ``livekit_api_secret`` to use a LiveKit project of your own for that room
    instead.

    Event handlers:

    - ``on_avatar_connected(service, session_id)``: the avatar's video is flowing.
    - ``on_avatar_disconnected(service, session_id)``: the avatar left
      unexpectedly (for example its renderer stopped). An error is pushed too.
    - ``on_playback_finished(service, playback_position, interrupted)``: the
      avatar finished (or was cut off in) an utterance.
    """

    Settings = AtmeeVideoSettings
    _settings: AtmeeVideoSettings

    def __init__(
        self,
        *,
        avatar_id: str,
        api_key: str | None = None,
        api_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
        max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
        metadata: dict[str, Any] | None = None,
        agent_identity: str = DEFAULT_AGENT_IDENTITY,
        avatar_participant_identity: str = DEFAULT_AVATAR_IDENTITY,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
        livekit_room_name: str | None = None,
        avatar_join_timeout: float = DEFAULT_AVATAR_JOIN_TIMEOUT,
        avatar_warmup_timeout: float = DEFAULT_AVATAR_WARMUP_TIMEOUT,
        conn_options: ConnectOptions = DEFAULT_CONNECT_OPTIONS,
        settings: AtmeeVideoSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Atmee video service.

        Args:
            avatar_id: The Atmee avatar to render (see ``AtmeeAPI.create_avatar``).
            api_key: Your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
            api_url: API base; defaults to ``ATMEE_API_URL`` or the Atmee cloud.
            session: An aiohttp session to reuse for API calls.
            max_duration_seconds: Hard ceiling of the session (default 3600).
            metadata: Free-form JSON attached to the session on the Atmee side.
            agent_identity: The identity your bot uses in the media room.
            avatar_participant_identity: The avatar's identity in your own
                LiveKit room (bring-your-own-LiveKit only).
            livekit_url: Your LiveKit server URL, to render into a room of your
                own LiveKit project instead of an Atmee-hosted one. Never read
                from the environment: pass it with key and secret explicitly.
            livekit_api_key: Your LiveKit API key (bring-your-own-LiveKit).
            livekit_api_secret: Your LiveKit API secret (bring-your-own-LiveKit).
                It never leaves your process: the avatar's token is minted locally.
            livekit_room_name: Room name in your LiveKit project; random by default.
            avatar_join_timeout: How long the first bot speech waits, once the
                session has started, for the avatar's video before it is
                dropped with an error.
            avatar_warmup_timeout: How long the start keeps retrying while a
                v2 avatar is being prepared on a rendering machine.
            conn_options: Retry/timeout policy of the API client.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``AIService``.
        """
        default_settings = AtmeeVideoSettings(model=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(settings=default_settings, **kwargs)
        if not avatar_id:
            raise AtmeeException("avatar_id is required")
        byo = [livekit_url, livekit_api_key, livekit_api_secret]
        if any(byo) and not all(byo):
            raise AtmeeException(
                "livekit_url, livekit_api_key and livekit_api_secret must be given together"
            )
        self._avatar_id = avatar_id
        self._api = AtmeeAPI(
            api_key=api_key, api_url=api_url, conn_options=conn_options, session=session
        )
        self._max_duration_seconds = int(max_duration_seconds)
        self._metadata = metadata
        self._agent_identity = agent_identity
        self._avatar_identity = avatar_participant_identity
        self._lk_url = livekit_url
        self._lk_key = livekit_api_key
        self._lk_secret = livekit_api_secret
        self._lk_room_name = livekit_room_name
        self._avatar_join_timeout = avatar_join_timeout
        self._avatar_warmup_timeout = avatar_warmup_timeout

        self.session_id: str | None = None
        """The Atmee avatar session id once the pipeline started."""
        self.session_info: AvatarSessionInfo | None = None
        """The full start response once the pipeline started."""

        self._room: rtc.Room | None = None
        self._sender: DataStreamAudioSender | None = None
        self._resampler = create_stream_resampler()
        self._resampler_in_rate: int | None = None
        self._out_sample_rate = 24000
        self._avatar_ready: asyncio.Event = asyncio.Event()
        # Set once a wait for the avatar timed out (already reported): later
        # speech is dropped at once instead of waiting the full timeout again.
        self._avatar_wait_failed = False
        self._transport_ready = False
        self._video_task: asyncio.Task | None = None
        self._audio_task: asyncio.Task | None = None
        self._started = False
        self._start_task: asyncio.Task | None = None
        # Set once the background start finished, successfully or not.
        self._start_done: asyncio.Event = asyncio.Event()
        self._ended = False
        self._closing = False
        # Set together with _closing: wakes waits that must end on shutdown.
        self._closed: asyncio.Event = asyncio.Event()
        self._teardown_lock = asyncio.Lock()
        self._end_lock = asyncio.Lock()
        # True from the first audio of an utterance until the avatar reports
        # its playback finished: only then is there anything to interrupt.
        self._avatar_busy = False
        # Monotonic time by which the avatar should have played out everything
        # sent so far. A late playback_finished of an earlier utterance must not
        # mark the avatar idle while a later one is still queued.
        self._playout_end = 0.0
        # After an interruption, the avatar's audio already in flight is
        # dropped until this deadline (see INTERRUPT_MUTE_SECONDS).
        self._muted_until = 0.0

        self._register_event_handler("on_avatar_connected")
        self._register_event_handler("on_avatar_disconnected")
        self._register_event_handler("on_playback_finished")

    @property
    def api(self) -> AtmeeAPI:
        """The API client this service uses."""
        return self._api

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, TTFB is measured from TTS start to the avatar's first speech.
        """
        return True

    # --- lifecycle -----------------------------------------------------------

    async def setup(self, setup: FrameProcessorSetup):  # type: ignore[override]
        """Remember the pipeline's output sample rate.

        Args:
            setup: The frame processor setup.
        """
        await super().setup(setup)
        self._out_sample_rate = setup.audio_out_sample_rate

    async def start(self, frame: StartFrame):
        """Start the avatar session and join its media room.

        The session starts in the background: a v2 avatar on a cold rendering
        machine takes minutes, far longer than the pipeline waits for its
        ``StartFrame``. Bot speech waits for the avatar (``avatar_join_timeout``).

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        if self._started:
            return
        self._started = True
        self._start_task = self.create_task(self._run_start(), "atmee_start")

    async def _run_start(self) -> None:
        try:
            await self._start_session()
        except _StartAborted:
            pass
        except Exception as e:
            if self._closing:
                return  # the pipeline is stopping anyway; its teardown ends the session
            await self._teardown()
            await self.push_error(
                f"Atmee avatar session failed to start: {e}",
                exception=e,
                force_treat_as_permanent=True,
            )
        finally:
            self._start_done.set()

    async def stop(self, frame: EndFrame):
        """End the avatar session gracefully.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """End the avatar session immediately.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        """Release every resource (idempotent)."""
        await super().cleanup()
        await self._teardown()
        await self._api.aclose()

    async def _start_session(self) -> None:
        if self._lk_url:
            room_name = self._lk_room_name or f"atmee-pipecat-{secrets.token_hex(6)}"
            await self._connect(self._lk_url, self._mint_token(room_name, self._agent_identity))
            info = await self._create_session(
                livekit_url=self._lk_url,
                livekit_token=self._mint_avatar_token(room_name),
                agent_identity=self._agent_identity,
                max_duration_seconds=self._max_duration_seconds,
                metadata=self._metadata,
            )
            self._remember(info)
            await self._end_if_closing()
        else:
            try:
                info = await self._create_session(
                    agent_identity=self._agent_identity,
                    max_duration_seconds=self._max_duration_seconds,
                    metadata=self._metadata,
                )
            except AtmeeHostedRoomMissingError as e:
                self._remember(e.info)
                raise
            self._remember(info)
            await self._end_if_closing()
            if info.avatar_participant_identity:
                self._avatar_identity = info.avatar_participant_identity
            assert info.livekit_url and info.agent_token
            await self._connect(info.livekit_url, info.agent_token)
        logger.info(
            f"{self}: Atmee avatar session {self.session_id} started "
            f"(billing {info.billing_mode}, ceiling {info.max_duration_seconds}s)"
        )
        self._check_avatar_present()

    async def _create_session(self, **kwargs: Any) -> AvatarSessionInfo:
        """Start the session, retrying while a v2 avatar is still warming up.

        A 503 ``avatar_warming`` rendered and billed nothing, so starting
        again is safe.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._avatar_warmup_timeout
        while True:
            try:
                return await self._api.create_avatar_session(self._avatar_id, **kwargs)
            except AtmeeAvatarWarmingError as e:
                pause = e.retry_after if e.retry_after is not None else _WARMING_RETRY_SECONDS
                if self._closing:
                    raise _StartAborted() from e
                if loop.time() + pause > deadline:
                    raise
                logger.info(f"{self}: the Atmee avatar is warming up; retrying in {pause:.0f}s")
                if await self._wait_or_closed(self._closed, pause):
                    raise _StartAborted() from e

    async def _end_if_closing(self) -> None:
        """The pipeline stopped while the start call was in flight: end the
        session it created instead of joining it."""
        if self._closing:
            await self._end_session()
            raise _StartAborted()

    async def _wait_or_closed(self, event: asyncio.Event, seconds: float | None) -> bool:
        """Wait up to ``seconds`` (``None``: no limit) for ``event``; return
        early on shutdown.

        Returns:
            Whether ``event`` is set.
        """
        waiters = {asyncio.ensure_future(event.wait()), asyncio.ensure_future(self._closed.wait())}
        try:
            await asyncio.wait(waiters, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        return event.is_set()

    def _remember(self, info: AvatarSessionInfo) -> None:
        self.session_info = info
        self.session_id = info.session_id

    def _mint_token(self, room_name: str, identity: str) -> str:
        assert self._lk_key and self._lk_secret
        return (
            lk_api.AccessToken(api_key=self._lk_key, api_secret=self._lk_secret)
            .with_identity(identity)
            .with_kind("agent")
            .with_grants(lk_api.VideoGrants(room_join=True, room=room_name))
            .with_ttl(timedelta(seconds=self._max_duration_seconds) + _TOKEN_TTL_SLACK)
            .to_jwt()
        )

    def _mint_avatar_token(self, room_name: str) -> str:
        assert self._lk_key and self._lk_secret
        return (
            lk_api.AccessToken(api_key=self._lk_key, api_secret=self._lk_secret)
            .with_kind("agent")
            .with_identity(self._avatar_identity)
            .with_name(self._avatar_identity)
            .with_grants(lk_api.VideoGrants(room_join=True, room=room_name))
            # the avatar publishes its audio and video on behalf of the bot
            .with_attributes({_PUBLISH_ON_BEHALF: self._agent_identity})
            .with_ttl(timedelta(seconds=self._max_duration_seconds) + _TOKEN_TTL_SLACK)
            .to_jwt()
        )

    async def _connect(self, url: str, token: str) -> None:
        room = rtc.Room()
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("participant_disconnected", self._on_participant_disconnected)
        self._room = room
        await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=True))
        self._sender = DataStreamAudioSender(
            room,
            destination_identity=self._avatar_identity,
            sample_rate=AVATAR_SAMPLE_RATE,
            on_playback_finished=self._on_playback_finished,
        )
        self._sender.register_rpc_methods()

    def _check_avatar_present(self) -> None:
        """Pick up tracks the avatar published before we joined."""
        if self._room is None:
            return
        participant = self._room.remote_participants.get(self._avatar_identity)
        if participant is None:
            return
        for pub in participant.track_publications.values():
            if pub.track is not None:
                self._on_track_subscribed(pub.track, pub, participant)  # type: ignore[arg-type]

    # --- room events -----------------------------------------------------------

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if participant.identity != self._avatar_identity or self._closing:
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO and self._video_task is None:
            # Let the SDK convert to RGB24 off the event loop.
            video = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
            self._video_task = self.create_task(self._video_loop(video), "atmee_video")
        elif track.kind == rtc.TrackKind.KIND_AUDIO and self._audio_task is None:
            stream = rtc.AudioStream(track, sample_rate=self._out_sample_rate, num_channels=1)
            self._audio_task = self.create_task(self._audio_loop(stream), "atmee_audio")

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity or self._closing:
            return
        self.create_task(self._handle_avatar_left(), "atmee_avatar_left")

    async def _handle_avatar_left(self) -> None:
        logger.warning(f"{self}: the Atmee avatar left the room")
        # The session is over: drop further speech instead of waiting for an
        # avatar that will not come back.
        self._closing = True
        self._closed.set()
        self._avatar_ready.clear()
        await self._call_event_handler("on_avatar_disconnected", self.session_id)
        await self._end_session()
        await self.push_error("the Atmee avatar left the room", force_treat_as_permanent=True)

    async def _on_playback_finished(self, position: float, interrupted: bool) -> None:
        idle = self._sender is None or not self._sender.speaking
        if idle and time.monotonic() >= self._playout_end:
            self._avatar_busy = False
        await self._call_event_handler("on_playback_finished", position, interrupted)

    async def _video_loop(self, stream: rtc.VideoStream) -> None:
        try:
            async for event in stream:
                frame = event.frame
                if not self._avatar_ready.is_set():
                    self._avatar_ready.set()
                    self._avatar_wait_failed = False
                    logger.debug(f"{self}: avatar video {frame.width}x{frame.height}")
                    await self._call_event_handler("on_avatar_connected", self.session_id)
                if not self._transport_ready:
                    continue
                if frame.type != rtc.VideoBufferType.RGB24:
                    frame = frame.convert(rtc.VideoBufferType.RGB24)
                await self.push_frame(
                    OutputImageRawFrame(
                        image=bytes(frame.data), size=(frame.width, frame.height), format="RGB"
                    )
                )
        finally:
            await stream.aclose()

    async def _audio_loop(self, stream: rtc.AudioStream) -> None:
        try:
            async for event in stream:
                if not self._transport_ready:
                    continue
                if self._muted_until and time.monotonic() < self._muted_until:
                    continue
                frame = event.frame
                await self.push_frame(
                    SpeechOutputAudioRawFrame(
                        audio=bytes(frame.data),
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    )
                )
        finally:
            await stream.aclose()

    # --- frames -------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Route the bot's speech to the avatar and pass everything else on.

        Args:
            frame: The frame to process.
            direction: The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStoppedFrame):
            await self._flush_resampler()
            if self._sender is not None:
                await self._sender.flush()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._interrupt()
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # The avatar's audio arrives continuously (silence when idle); the
            # output transport reports the bot speaking once it hears speech.
            await self.stop_ttfb_metrics()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _send_audio(self, frame: TTSAudioRawFrame) -> None:
        if self._closing:
            return
        if not self._avatar_ready.is_set():
            if self._avatar_wait_failed:
                return  # already reported: drop until the avatar shows up
            # The start (warm-up retries included) is bounded by its own
            # timeouts; avatar_join_timeout counts from a started session.
            if self._start_task is not None:
                await self._wait_or_closed(self._start_done, None)
                if self._closing:
                    return
            if not await self._wait_or_closed(self._avatar_ready, self._avatar_join_timeout):
                if self._closing:
                    return
                self._avatar_wait_failed = True
                await self.push_error(
                    f"the Atmee avatar was not ready after {self._avatar_join_timeout:.0f}s; "
                    "dropping bot speech"
                )
                return
            if self._sender is None or self._closing:
                return  # torn down while waiting
        try:
            pcm = frame.audio
            if frame.num_channels != 1:
                pcm = _downmix(pcm, frame.num_channels)
            if frame.sample_rate != self._resampler_in_rate:
                # a stream resampler is bound to one input rate
                self._resampler = create_stream_resampler()
                self._resampler_in_rate = frame.sample_rate
            pcm = await self._resampler.resample(pcm, frame.sample_rate, AVATAR_SAMPLE_RATE)
            await self._write(pcm)
        except Exception as e:
            await self.push_error(f"sending audio to the Atmee avatar failed: {e}", exception=e)

    async def _write(self, pcm: bytes) -> None:
        if self._sender is None:
            return
        await self._sender.write(pcm)
        self._avatar_busy = True
        now = time.monotonic()
        self._playout_end = max(self._playout_end, now) + len(pcm) / (2 * AVATAR_SAMPLE_RATE)

    async def _flush_resampler(self) -> None:
        """Send the tail the stream resampler still holds at the end of an utterance."""
        flush = getattr(self._resampler, "flush", None)  # absent in older pipecat
        if flush is None or self._sender is None or self._closing:
            return
        try:
            tail = await flush()
            if tail:
                await self._write(tail)
        except Exception as e:
            await self.push_error(f"sending audio to the Atmee avatar failed: {e}", exception=e)

    async def _interrupt(self) -> None:
        if self._sender is None or not self._avatar_ready.is_set():
            return
        if not self._avatar_busy and not self._sender.speaking:
            return  # the avatar is idle: nothing to stop
        self._avatar_busy = False
        self._playout_end = 0.0
        self._muted_until = time.monotonic() + INTERRUPT_MUTE_SECONDS
        self._resampler_in_rate = None  # a fresh resampler: it holds the old speech's tail
        await self._sender.clear()

    # --- teardown -------------------------------------------------------------------

    async def _teardown(self) -> None:
        async with self._teardown_lock:
            self._closing = True
            self._closed.set()
            await self._abort_start()
            await self._end_session()
            for task in (self._video_task, self._audio_task):
                if task is not None:
                    await self.cancel_task(task)
            self._video_task = self._audio_task = None
            if self._sender is not None:
                self._sender.unregister_rpc_methods()
                self._sender = None
            if self._room is not None:
                room, self._room = self._room, None
                try:
                    await room.disconnect()
                except Exception as e:
                    logger.warning(f"{self}: leaving the media room failed: {e}")
            self._avatar_ready.clear()

    async def _abort_start(self) -> None:
        """Let a start still in flight finish (it ends what it created), or
        cancel it after a grace period."""
        task = self._start_task
        if task is None or task.done() or task is asyncio.current_task():
            return
        done, _ = await asyncio.wait({task}, timeout=_START_ABORT_GRACE_SECONDS)
        if not done:
            logger.warning(f"{self}: the Atmee avatar session was still starting; giving up on it")
            await self.cancel_task(task)

    async def _end_session(self) -> None:
        """``POST /end``; marked done only once the API confirmed it, so a later
        teardown retries a failed end (the call is idempotent)."""
        async with self._end_lock:
            if self._ended or not self.session_id:
                return
            try:
                await self._api.end_avatar_session(self.session_id)
            except AtmeeException as e:
                if 400 <= e.status_code < 500 and e.status_code not in (408, 429):
                    self._ended = True  # unknown or already settled: nothing to end
                logger.warning(
                    f"{self}: ending Atmee session {self.session_id} failed ({e}); "
                    "the session ceiling settles it"
                )
                return
            except Exception as e:
                logger.warning(f"{self}: ending Atmee session {self.session_id} failed: {e}")
                return
            self._ended = True
            logger.info(f"{self}: Atmee avatar session {self.session_id} ended")


def _downmix(pcm: bytes, channels: int) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
    return samples.mean(axis=1).astype(np.int16).tobytes()
