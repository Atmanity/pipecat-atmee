from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_atmee import AtmeeAPI, AtmeeDailyVideoService
from pipecat_atmee.daily import MESSAGE_KEY


class FakeOutput:
    def __init__(self, transport: FakeTransport) -> None:
        self._transport = transport

    async def send_message(self, frame: Any) -> None:
        self._transport.sent.append((frame.participant_id, frame.message))
        if frame.message.get(MESSAGE_KEY) == "hello" and self._transport.avatar_answers:
            # The avatar answers hello with ready.
            await self._transport.deliver({MESSAGE_KEY: "ready"}, "avatar-1")


class FakeTransport:
    """The part of DailyTransport the service uses."""

    def __init__(self, *, avatar_answers: bool = True) -> None:
        self.participant_id = "bot-1"
        self.room_url = "https://acme.daily.co/bot-room"
        self.sent: list[tuple[str | None, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}
        self.avatar_answers = avatar_answers
        self._output = FakeOutput(self)

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def output(self) -> FakeOutput:
        return self._output

    async def deliver(self, message: dict[str, Any], sender: str) -> None:
        await self.handlers["on_app_message"](self, message, sender)

    def to_avatar(self, kind: str) -> list[dict[str, Any]]:
        return [m for pid, m in self.sent if pid == "avatar-1" and m.get(MESSAGE_KEY) == kind]


class Harness(AtmeeDailyVideoService):
    """The service with the start call recorded instead of sent."""

    def __init__(self, transport: FakeTransport, **kwargs: Any) -> None:
        super().__init__(
            transport=transport,  # type: ignore[arg-type]
            avatar_id="a-1",
            api_key="sk_atmee_test",
            daily_api_key="",
            **kwargs,
        )
        self.started: list[dict[str, Any]] = []
        self.ended: list[str] = []

    async def _create_session(self, **kwargs: Any) -> Any:
        self.started.append(kwargs)
        from pipecat_atmee import AvatarSessionInfo

        return AvatarSessionInfo(
            session_id="s-1",
            status="initializing",
            avatar_participant_identity="Atmee avatar",
            agent_identity="bot-1",
            room_name="bot-room",
            max_duration_seconds=3600,
            billing_mode="metered",
        )

    async def _end_session(self) -> None:
        if self.session_id and self.session_id not in self.ended:
            self.ended.append(self.session_id)


def tts_audio(ms: int = 20, rate: int = 24000) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(
        audio=b"\x10\x00" * (rate * ms // 1000), sample_rate=rate, num_channels=1
    )


async def test_the_session_is_started_for_the_bots_daily_room():
    transport = FakeTransport()
    svc = Harness(transport)
    await run_test(svc, frames_to_send=[SleepFrame(0.2)])
    assert svc.started == [
        {
            "daily_room_url": "https://acme.daily.co/bot-room",
            "daily_token": None,
            "daily_participant_id": "bot-1",
            "max_duration_seconds": 3600,
            "metadata": None,
        }
    ]
    assert svc.ended == ["s-1"]
    # The avatar is told to leave when the pipeline ends.
    assert transport.to_avatar("end") == [{MESSAGE_KEY: "end"}]


async def test_speech_goes_to_the_avatar_and_on_downstream():
    transport = FakeTransport()
    svc = Harness(transport)
    down, _ = await run_test(
        svc,
        frames_to_send=[
            SleepFrame(0.2),
            TTSStartedFrame(),
            *[tts_audio() for _ in range(25)],
            TTSStoppedFrame(),
            SleepFrame(0.2),
        ],
    )
    # The bot's own transport still sees its speech (its microphone is off).
    assert sum(isinstance(f, TTSAudioRawFrame) for f in down) == 25
    chunks = transport.to_avatar("audio")
    pcm = b"".join(base64.b64decode(c["audio"]) for c in chunks)
    # 0.5 s at 24 kHz, resampled to 16 kHz 16-bit mono, in 100 ms messages
    assert abs(len(pcm) / 32000 - 0.5) < 0.05
    assert all(c["sample_rate"] == 16000 for c in chunks)
    assert len({c["segment"] for c in chunks}) == 1
    assert transport.to_avatar("audio_end") == [
        {MESSAGE_KEY: "audio_end", "segment": chunks[0]["segment"]}
    ]


async def test_speech_before_the_avatar_is_ready_waits_for_it():
    transport = FakeTransport(avatar_answers=False)
    svc = Harness(transport)

    async def avatar_joins_late() -> None:
        await asyncio.sleep(0.3)
        await transport.deliver({MESSAGE_KEY: "ready"}, "avatar-1")

    task = asyncio.create_task(avatar_joins_late())
    await run_test(
        svc,
        frames_to_send=[
            TTSStartedFrame(),
            *[tts_audio() for _ in range(10)],
            TTSStoppedFrame(),
            SleepFrame(0.6),
        ],
    )
    await task
    assert transport.to_avatar("audio") and transport.to_avatar("audio_end")


async def test_interruption_clears_a_speaking_avatar():
    transport = FakeTransport()
    svc = Harness(transport)
    await run_test(
        svc,
        frames_to_send=[
            SleepFrame(0.2),
            TTSStartedFrame(),
            tts_audio(200),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.1),
        ],
    )
    assert transport.to_avatar("clear_buffer") == [{MESSAGE_KEY: "clear_buffer"}]


async def test_interruption_of_an_idle_avatar_sends_nothing():
    transport = FakeTransport()
    svc = Harness(transport)
    await run_test(svc, frames_to_send=[SleepFrame(0.2), InterruptionFrame(), SleepFrame(0.1)])
    assert transport.to_avatar("clear_buffer") == []


async def test_messages_from_others_are_ignored():
    transport = FakeTransport(avatar_answers=False)
    svc = Harness(transport)
    finished: list[tuple[float, bool]] = []

    @svc.event_handler("on_playback_finished")
    async def on_finished(_svc: Any, position: float, interrupted: bool) -> None:
        finished.append((position, interrupted))

    async def messages() -> None:
        await asyncio.sleep(0.2)
        await transport.deliver({MESSAGE_KEY: "ready"}, "avatar-1")
        # A second "avatar" cannot take over, a user cannot fake an ack.
        await transport.deliver({MESSAGE_KEY: "ready"}, "user-1")
        await transport.deliver(
            {MESSAGE_KEY: "playback_finished", "position": 1.0, "interrupted": False}, "user-1"
        )
        await transport.deliver(
            {MESSAGE_KEY: "playback_finished", "position": 2.0, "interrupted": True}, "avatar-1"
        )

    task = asyncio.create_task(messages())
    await run_test(svc, frames_to_send=[SleepFrame(0.5)])
    await task
    assert svc._avatar_participant == "avatar-1"
    assert finished == [(2.0, True)]


async def test_the_avatar_leaving_ends_the_session():
    transport = FakeTransport()
    svc = Harness(transport)
    disconnected: list[str | None] = []

    @svc.event_handler("on_avatar_disconnected")
    async def on_left(_svc: Any, session_id: str | None) -> None:
        disconnected.append(session_id)

    async def avatar_leaves() -> None:
        await asyncio.sleep(0.2)
        await transport.handlers["on_participant_left"](transport, {"id": "avatar-1"}, "left")

    task = asyncio.create_task(avatar_leaves())
    await run_test(svc, frames_to_send=[SleepFrame(0.4)], expected_down_frames=None)
    await task
    assert disconnected == ["s-1"] and svc.ended == ["s-1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daily_room_url": "https://acme.daily.co/r"},
        {"daily_participant_id": "bot-1"},
        {
            "daily_room_url": "https://acme.daily.co/r",
            "daily_participant_id": "bot-1",
            "livekit_url": "wss://x",
            "livekit_token": "t",
        },
    ],
)
async def test_the_api_refuses_an_incomplete_daily_room(kwargs: dict[str, Any]):
    async with AtmeeAPI(api_key="sk_atmee_test", api_url="http://127.0.0.1:9") as api:
        with pytest.raises(ValueError):
            await api.create_avatar_session("a-1", **kwargs)


async def test_the_api_sends_the_daily_room(fake_api: Any):
    fake_api.respond(
        "/v1/avatars/a-1/avatar_sessions",
        (
            202,
            {
                "sessionId": "s-1",
                "status": "initializing",
                "avatarParticipantIdentity": "Atmee avatar",
                "agentIdentity": "bot-1",
                "roomName": "r",
                "maxDurationSeconds": 3600,
                "billingMode": "metered",
            },
        ),
    )
    async with AtmeeAPI(api_key="sk_atmee_test", api_url=fake_api.url) as api:
        info = await api.create_avatar_session(
            "a-1",
            daily_room_url="https://acme.daily.co/r",
            daily_token="tok",
            daily_participant_id="bot-1",
        )
    assert info.session_id == "s-1" and info.livekit_url is None
    assert fake_api.calls("/v1/avatars/a-1/avatar_sessions") == [
        {"daily": {"roomUrl": "https://acme.daily.co/r", "participantId": "bot-1", "token": "tok"}}
    ]
