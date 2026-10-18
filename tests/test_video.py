from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from livekit import api as lk_api
from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_atmee import (
    AtmeeAvatarWarmingError,
    AtmeeException,
    AtmeeVideoService,
    AvatarSessionInfo,
)


class FakeSender:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.calls: list[str] = []
        self.speaking = False

    async def write(self, pcm: bytes) -> None:
        self.audio += pcm
        self.speaking = True
        self.calls.append("write")

    async def flush(self) -> None:
        self.speaking = False
        self.calls.append("flush")

    async def clear(self) -> None:
        self.speaking = False
        self.calls.append("clear")

    def unregister_rpc_methods(self) -> None:
        self.calls.append("unregister")


class FakeAPI:
    def __init__(self, fail_first_end: Exception | None = None) -> None:
        self.ended: list[str] = []
        self._fail = fail_first_end

    async def end_avatar_session(self, session_id: str) -> dict[str, Any]:
        if self._fail is not None:
            err, self._fail = self._fail, None
            raise err
        self.ended.append(session_id)
        return {}

    async def aclose(self) -> None:
        pass


class Harness(AtmeeVideoService):
    """The service with its room and API replaced by fakes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(avatar_id="a-1", api_key="sk_atmee_test", **kwargs)
        self.fake_sender = FakeSender()
        self.fake_api = FakeAPI()
        self._api = self.fake_api  # type: ignore[assignment]

    async def _start_session(self) -> None:
        self.session_id = "s-1"
        self._sender = self.fake_sender  # type: ignore[assignment]
        self._avatar_ready.set()


def tts_audio(ms: int = 20, rate: int = 24000) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(
        audio=b"\x10\x00" * (rate * ms // 1000), sample_rate=rate, num_channels=1
    )


async def test_tts_audio_goes_to_the_avatar_not_downstream():
    svc = Harness()
    down, _ = await run_test(
        svc,
        frames_to_send=[TTSStartedFrame(), *[tts_audio() for _ in range(25)], TTSStoppedFrame()],
        expected_down_frames=[TTSStartedFrame, TTSStoppedFrame],
    )
    # resampled 24 kHz -> 16 kHz, 16-bit mono: ~0.5 s of audio
    assert abs(len(svc.fake_sender.audio) / 32000 - 0.5) < 0.05
    assert svc.fake_sender.calls[-2:] == ["flush", "unregister"]
    assert svc.fake_api.ended == ["s-1"]


async def test_interruption_clears_a_speaking_avatar():
    svc = Harness()
    await run_test(
        svc,
        frames_to_send=[TTSStartedFrame(), tts_audio(), SleepFrame(0.1), InterruptionFrame()],
    )
    assert "clear" in svc.fake_sender.calls


async def test_interruption_of_an_idle_avatar_sends_nothing():
    svc = Harness()
    await run_test(svc, frames_to_send=[InterruptionFrame()])
    assert "clear" not in svc.fake_sender.calls


async def test_failed_end_is_retried_by_a_later_teardown():
    svc = Harness()
    svc.fake_api = FakeAPI(fail_first_end=AtmeeException("down", status_code=502))
    svc._api = svc.fake_api  # type: ignore[assignment]
    await run_test(svc, frames_to_send=[])
    # stop() failed to end the session; cleanup() tried again and succeeded
    assert svc.fake_api.ended == ["s-1"]
    await svc._end_session()
    assert svc.fake_api.ended == ["s-1"]  # once confirmed, never again


async def test_unknown_session_end_is_not_retried():
    svc = Harness()
    svc.fake_api = FakeAPI(fail_first_end=AtmeeException("gone", status_code=404))
    svc._api = svc.fake_api  # type: ignore[assignment]
    await run_test(svc, frames_to_send=[])
    await svc._end_session()
    assert svc.fake_api.ended == []


def test_byo_tokens():
    svc = AtmeeVideoService(
        avatar_id="a-1",
        api_key="sk_atmee_test",
        livekit_url="wss://mine.example",
        livekit_api_key="APIkey",
        livekit_api_secret="s" * 32,
        agent_identity="my-bot",
    )
    verifier = lk_api.TokenVerifier("APIkey", "s" * 32)
    avatar = verifier.verify(svc._mint_avatar_token("room-1"))
    assert avatar.identity == "atmee-avatar-agent"
    assert avatar.video.room == "room-1" and avatar.video.room_join
    assert avatar.attributes == {"lk.publish_on_behalf": "my-bot"}
    agent = verifier.verify(svc._mint_token("room-1", "my-bot"))
    assert agent.identity == "my-bot" and agent.video.room == "room-1"


def test_byo_needs_all_three():
    with pytest.raises(AtmeeException):
        AtmeeVideoService(avatar_id="a-1", api_key="k", livekit_url="wss://x")


class WarmingAPI(FakeAPI):
    """Answers avatar_warming ``warm_for`` times, then starts the session."""

    def __init__(self, warm_for: int) -> None:
        super().__init__()
        self.starts = 0
        self._warm_for = warm_for
        self.warm_retry_after = 0.01

    async def create_avatar_session(self, avatar_id: str, **kwargs: Any) -> str:
        self.starts += 1
        if self.starts <= self._warm_for:
            raise AtmeeAvatarWarmingError("preparing", retry_after=self.warm_retry_after)
        return "info"


async def test_start_retries_while_the_avatar_warms_up():
    service = AtmeeVideoService(avatar_id="a-1", api_key="sk_atmee_test")
    service._api = WarmingAPI(warm_for=2)  # type: ignore[assignment]
    assert await service._create_session(agent_identity="bot") == "info"
    assert service._api.starts == 3


async def test_warming_retries_stop_at_the_warmup_timeout():
    service = AtmeeVideoService(
        avatar_id="a-1", api_key="sk_atmee_test", avatar_warmup_timeout=0.015
    )
    service._api = WarmingAPI(warm_for=100)  # type: ignore[assignment]
    with pytest.raises(AtmeeAvatarWarmingError):
        await service._create_session(agent_identity="bot")
    assert service._api.starts == 2


class BlockingStartAPI(FakeAPI):
    """A start call that only returns once ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.starts = 0

    async def create_avatar_session(self, avatar_id: str, **kwargs: Any) -> AvatarSessionInfo:
        self.starts += 1
        await self.release.wait()
        return AvatarSessionInfo(
            session_id="s-late",
            status="initializing",
            avatar_participant_identity="atmee-avatar-agent",
            agent_identity="bot",
            room_name="r",
            max_duration_seconds=60,
            billing_mode="reserve",
            livekit_url="wss://example",
            agent_token="t",
        )


def bare_service(**kwargs: Any) -> AtmeeVideoService:
    service = AtmeeVideoService(avatar_id="a-1", api_key="sk_atmee_test", **kwargs)
    service.cancel_task = _cancel  # type: ignore[method-assign]
    return service


async def _cancel(task: asyncio.Task, *_: Any) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_stop_during_warming_pause_starts_nothing_more():
    service = bare_service()
    api = WarmingAPI(warm_for=100)
    api.warm_retry_after = 30.0
    service._api = api  # type: ignore[assignment]
    task = asyncio.ensure_future(service._create_session(agent_identity="bot"))
    await asyncio.sleep(0.05)  # first attempt answered avatar_warming; now pausing
    service._closing = True
    service._closed.set()
    with pytest.raises(Exception) as err:
        await asyncio.wait_for(task, 1.0)
    assert type(err.value).__name__ == "_StartAborted"
    assert api.starts == 1


async def test_stop_during_start_call_ends_the_created_session():
    service = bare_service()
    api = BlockingStartAPI()
    service._api = api  # type: ignore[assignment]
    service._start_task = asyncio.ensure_future(service._run_start())
    await asyncio.sleep(0.01)
    teardown = asyncio.ensure_future(service._teardown())
    await asyncio.sleep(0.01)
    api.release.set()  # the start call returns after the pipeline stopped
    await asyncio.wait_for(teardown, 2.0)
    assert service.session_id == "s-late"
    assert api.ended == ["s-late"]
    assert service._room is None


async def test_first_speech_waits_for_the_start_before_the_join_timeout():
    service = bare_service(avatar_join_timeout=0.05)
    service._start_task = asyncio.ensure_future(asyncio.sleep(1))  # still starting
    service._sender = FakeSender()  # type: ignore[assignment]
    send = asyncio.ensure_future(service._send_audio(tts_audio()))
    await asyncio.sleep(0.2)  # longer than the join timeout, but the start is not done
    assert not send.done()
    assert not service._avatar_wait_failed
    service._start_done.set()
    service._avatar_ready.set()
    await asyncio.wait_for(send, 1.0)
    assert service._sender.calls == ["write"]  # type: ignore[union-attr]
    service._start_task.cancel()
