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

"""Bot speech to the avatar over a LiveKit data stream.

The wire protocol of LiveKit Agents' remote avatars (``DataStreamAudioOutput``),
reimplemented on the plain ``livekit`` SDK so this package does not depend on
``livekit-agents``:

- every utterance is one byte stream on topic ``lk.audio_stream`` addressed
  to the avatar participant, carrying raw 16-bit little-endian PCM, with the
  ``sample_rate`` and ``num_channels`` attributes; closing the stream marks
  the end of the utterance;
- an interruption is the ``lk.clear_buffer`` RPC to the avatar;
- the avatar reports the end of each utterance with the
  ``lk.playback_finished`` RPC (JSON ``{"playback_position": float,
  "interrupted": bool}``), and may report its start with
  ``lk.playback_started``.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from livekit import rtc
from loguru import logger

RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
RPC_PLAYBACK_STARTED = "lk.playback_started"
AUDIO_STREAM_TOPIC = "lk.audio_stream"

PlaybackFinishedCallback = Callable[[float, bool], Awaitable[None]]


class DataStreamAudioSender:
    """Streams bot speech to one avatar participant and relays its playback events.

    Args:
        room: A connected ``rtc.Room``.
        destination_identity: The avatar participant's identity.
        sample_rate: Sample rate of the PCM passed to :meth:`write`.
        num_channels: Channel count of the PCM passed to :meth:`write`.
        on_playback_finished: Awaited with ``(playback_position, interrupted)``
            when the avatar finished (or was cut off in) an utterance.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        destination_identity: str,
        sample_rate: int,
        num_channels: int = 1,
        on_playback_finished: PlaybackFinishedCallback | None = None,
    ) -> None:
        """Initialize the sender (see the class docstring for the arguments)."""
        self._room = room
        self._destination = destination_identity
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._on_playback_finished = on_playback_finished
        self._writer: rtc.ByteStreamWriter | None = None
        self._pushed_seconds = 0.0
        self._registered = False

    @property
    def speaking(self) -> bool:
        """Whether an utterance stream is open."""
        return self._writer is not None

    @property
    def pushed_seconds(self) -> float:
        """Audio sent in the current (or last) utterance, in seconds."""
        return self._pushed_seconds

    def register_rpc_methods(self) -> None:
        """Answer the avatar's playback RPCs. Call once the room is connected."""
        if self._registered:
            return
        participant = self._room.local_participant
        participant.register_rpc_method(RPC_PLAYBACK_FINISHED, self._handle_playback_finished)
        participant.register_rpc_method(RPC_PLAYBACK_STARTED, self._handle_playback_started)
        self._registered = True

    def unregister_rpc_methods(self) -> None:
        """Stop answering the avatar's playback RPCs."""
        if not self._registered:
            return
        self._registered = False
        participant = self._room.local_participant
        for method in (RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED):
            try:
                participant.unregister_rpc_method(method)
            except Exception:  # the room may already be gone
                pass

    async def write(self, pcm: bytes) -> None:
        """Append 16-bit PCM to the current utterance, opening one if needed."""
        if not pcm:
            return
        if self._writer is None:
            self._writer = await self._room.local_participant.stream_bytes(
                name="AUDIO_" + secrets.token_hex(6),
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[self._destination],
                attributes={
                    "sample_rate": str(self._sample_rate),
                    "num_channels": str(self._num_channels),
                },
            )
            self._pushed_seconds = 0.0
        await self._writer.write(pcm)
        self._pushed_seconds += len(pcm) / (2 * self._num_channels * self._sample_rate)

    async def flush(self) -> None:
        """End the current utterance: the avatar plays out what it has."""
        writer, self._writer = self._writer, None
        if writer is None:
            return
        try:
            await writer.aclose()
        except Exception as e:
            logger.warning(f"Atmee: closing the audio stream failed: {e}")

    async def clear(self) -> None:
        """Interrupt: drop the open utterance and tell the avatar to stop speaking."""
        await self.flush()
        try:
            await self._room.local_participant.perform_rpc(
                destination_identity=self._destination,
                method=RPC_CLEAR_BUFFER,
                payload="",
            )
        except Exception as e:
            logger.warning(f"Atmee: clear_buffer RPC failed: {e}")

    async def _handle_playback_finished(self, data: rtc.RpcInvocationData) -> str:
        if data.caller_identity != self._destination:
            return "reject"
        position, interrupted = 0.0, False
        try:
            event: Any = json.loads(data.payload or "{}")
            position = float(event.get("playback_position", 0.0))
            interrupted = bool(event.get("interrupted", False))
        except (ValueError, TypeError, AttributeError):
            logger.warning("Atmee: malformed playback_finished payload")
        if self._on_playback_finished is not None:
            await self._on_playback_finished(position, interrupted)
        return "ok"

    async def _handle_playback_started(self, data: rtc.RpcInvocationData) -> str:
        return "ok" if data.caller_identity == self._destination else "reject"
