"""The data-stream wire protocol the avatar speaks (LiveKit Agents' remote-avatar protocol)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pipecat_atmee._datastream import (
    AUDIO_STREAM_TOPIC,
    RPC_CLEAR_BUFFER,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    DataStreamAudioSender,
)


@dataclass
class FakeWriter:
    kwargs: dict[str, Any]
    chunks: list[bytes] = field(default_factory=list)
    closed: bool = False

    async def write(self, data: bytes) -> None:
        assert not self.closed
        self.chunks.append(data)

    async def aclose(self) -> None:
        self.closed = True


class FakeParticipant:
    def __init__(self) -> None:
        self.writers: list[FakeWriter] = []
        self.rpcs: list[dict[str, Any]] = []
        self.handlers: dict[str, Any] = {}

    async def stream_bytes(self, name: str, **kwargs: Any) -> FakeWriter:
        writer = FakeWriter({"name": name, **kwargs})
        self.writers.append(writer)
        return writer

    async def perform_rpc(self, **kwargs: Any) -> str:
        self.rpcs.append(kwargs)
        return "ok"

    def register_rpc_method(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def unregister_rpc_method(self, name: str) -> None:
        self.handlers.pop(name)


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeParticipant()


@dataclass
class Invocation:
    caller_identity: str
    payload: str


def sender(room: FakeRoom, **kwargs: Any) -> DataStreamAudioSender:
    return DataStreamAudioSender(
        room,  # type: ignore[arg-type]
        destination_identity="atmee-avatar-agent",
        sample_rate=16000,
        **kwargs,
    )


async def test_one_stream_per_utterance_with_attributes():
    room = FakeRoom()
    s = sender(room)
    await s.write(b"\x00\x01" * 160)
    await s.write(b"\x00\x01" * 160)
    assert s.speaking
    assert abs(s.pushed_seconds - 0.02) < 1e-9
    await s.flush()
    await s.write(b"\x00\x02" * 160)
    await s.flush()

    first, second = room.local_participant.writers
    assert first.kwargs["topic"] == AUDIO_STREAM_TOPIC
    assert first.kwargs["destination_identities"] == ["atmee-avatar-agent"]
    assert first.kwargs["attributes"] == {"sample_rate": "16000", "num_channels": "1"}
    assert first.kwargs["name"].startswith("AUDIO_")
    assert first.kwargs["name"] != second.kwargs["name"]
    assert first.closed and second.closed
    assert b"".join(first.chunks) == b"\x00\x01" * 320
    assert not s.speaking


async def test_flush_without_audio_opens_nothing():
    room = FakeRoom()
    s = sender(room)
    await s.flush()
    await s.write(b"")
    assert room.local_participant.writers == []


async def test_clear_closes_stream_and_calls_rpc():
    room = FakeRoom()
    s = sender(room)
    await s.write(b"\x00\x00" * 10)
    await s.clear()
    assert room.local_participant.writers[0].closed
    assert room.local_participant.rpcs == [
        {"destination_identity": "atmee-avatar-agent", "method": RPC_CLEAR_BUFFER, "payload": ""}
    ]


async def test_playback_rpcs():
    room = FakeRoom()
    events: list[tuple[float, bool]] = []

    async def on_finished(position: float, interrupted: bool) -> None:
        events.append((position, interrupted))

    s = sender(room, on_playback_finished=on_finished)
    s.register_rpc_methods()
    s.register_rpc_methods()  # idempotent
    handlers = room.local_participant.handlers
    assert set(handlers) == {RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED}

    payload = json.dumps({"playback_position": 2.5, "interrupted": True})
    assert await handlers[RPC_PLAYBACK_FINISHED](Invocation("atmee-avatar-agent", payload)) == "ok"
    assert await handlers[RPC_PLAYBACK_FINISHED](Invocation("someone-else", payload)) == "reject"
    assert await handlers[RPC_PLAYBACK_FINISHED](Invocation("atmee-avatar-agent", "nope")) == "ok"
    assert await handlers[RPC_PLAYBACK_STARTED](Invocation("atmee-avatar-agent", "")) == "ok"
    assert events == [(2.5, True), (0.0, False)]

    s.unregister_rpc_methods()
    assert handlers == {}
