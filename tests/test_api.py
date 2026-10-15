from __future__ import annotations

import pytest

from pipecat_atmee import (
    AtmeeAPI,
    AtmeeAvatarNotReadyError,
    AtmeeException,
    AtmeeHostedRoomMissingError,
    AtmeeNoCapacityError,
    ConnectOptions,
)

AVATAR = "006e5b5c-a4a7-4d7d-a016-b4848e8b60af"
START = f"/v1/avatars/{AVATAR}/avatar_sessions"
FAST = ConnectOptions(max_retry=2, retry_interval=0.01, timeout=2)

SESSION = {
    "sessionId": "s-1",
    "status": "initializing",
    "avatarParticipantIdentity": "atmee-avatar-agent",
    "agentIdentity": "pipecat-agent",
    "roomName": "room-1",
    "maxDurationSeconds": 600,
    "billingMode": "reserve",
}


def api(fake) -> AtmeeAPI:
    return AtmeeAPI(api_key="sk_atmee_test", api_url=fake.url, conn_options=FAST)


async def test_hosted_room_request_and_answer(fake_api):
    fake_api.respond(
        START, (202, {**SESSION, "livekitUrl": "wss://lk.example", "agentToken": "jwt-agent"})
    )
    async with api(fake_api) as client:
        info = await client.create_avatar_session(
            AVATAR, agent_identity="pipecat-agent", max_duration_seconds=600
        )
    body = fake_api.calls(START)[0]
    assert body == {"agentIdentity": "pipecat-agent", "maxDurationSeconds": 600}
    assert info.session_id == "s-1"
    assert info.livekit_url == "wss://lk.example"
    assert info.agent_token == "jwt-agent"
    # the token is a credential: never in repr or the raw copy
    assert "jwt-agent" not in repr(info)
    assert "agentToken" not in info.raw


async def test_byo_livekit_request(fake_api):
    fake_api.respond(START, (202, SESSION))
    async with api(fake_api) as client:
        info = await client.create_avatar_session(
            AVATAR, livekit_url="wss://mine", livekit_token="tok", wait_for="avatar_joined"
        )
    assert fake_api.calls(START)[0] == {"livekitUrl": "wss://mine", "livekitToken": "tok"}
    assert fake_api.requests[0][0] == "POST"
    assert info.livekit_url is None


async def test_byo_needs_url_and_token_together(fake_api):
    async with api(fake_api) as client:
        with pytest.raises(ValueError):
            await client.create_avatar_session(AVATAR, livekit_url="wss://mine")


async def test_hosted_room_missing_carries_session(fake_api):
    fake_api.respond(START, (202, SESSION))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeHostedRoomMissingError) as err:
            await client.create_avatar_session(AVATAR)
    assert err.value.info.session_id == "s-1"


async def test_start_is_never_retried(fake_api):
    fake_api.respond(START, (500, {"error": "internal_error", "message": "boom"}))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeException) as err:
            await client.create_avatar_session(AVATAR)
    assert err.value.status_code == 500
    assert len(fake_api.calls(START)) == 1


async def test_no_capacity_is_typed_and_final(fake_api):
    fake_api.respond(START, (503, {"error": "no_capacity", "message": "busy"}))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeNoCapacityError) as err:
            await client.create_avatar_session(AVATAR)
    assert err.value.retry_after == 5.0


async def test_not_renderable_is_typed(fake_api):
    fake_api.respond(START, (409, {"error": "avatar_not_renderable", "message": "no portrait"}))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeAvatarNotReadyError):
            await client.create_avatar_session(AVATAR)


async def test_end_retries_5xx(fake_api):
    path = "/v1/avatar_sessions/s-1/end"
    fake_api.respond(path, (502, {}), (200, {"status": "completed"}))
    async with api(fake_api) as client:
        assert await client.end_avatar_session("s-1") == {"status": "completed"}
    assert len(fake_api.calls(path)) == 2


async def test_4xx_is_final(fake_api):
    path = "/v1/avatar_sessions/s-1/end"
    fake_api.respond(path, (404, {"error": "not_found", "message": "gone"}))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeException) as err:
            await client.end_avatar_session("s-1")
    assert (err.value.status_code, err.value.code) == (404, "not_found")
    assert len(fake_api.calls(path)) == 1


async def test_retries_are_bounded(fake_api):
    path = "/v1/avatar_sessions/s-1"
    fake_api.respond(path, (500, {}))
    async with api(fake_api) as client:
        with pytest.raises(AtmeeException):
            await client.get_avatar_session("s-1")
    assert len(fake_api.calls(path)) == FAST.max_retry + 1


async def test_create_avatar_from_url(fake_api):
    fake_api.respond("/v1/avatars", (201, {"avatarId": "a-1", "status": "ready"}))
    async with api(fake_api) as client:
        assert await client.create_avatar("Val", "https://example.com/val.jpg") == "a-1"
    assert fake_api.calls("/v1/avatars")[0] == {
        "schemaVersion": 1,
        "name": "Val",
        "assets": {"image": {"url": "https://example.com/val.jpg"}},
    }


async def test_create_avatar_from_bytes(fake_api):
    fake_api.respond("/v1/avatars", (201, {"avatarId": "a-2", "status": "ready"}))
    async with api(fake_api) as client:
        info = await client.create_avatar_info("Val", b"\xff\xd8jpeg", content_type="image/jpeg")
    assert info.ready and info.avatar_id == "a-2"


def test_requires_key(monkeypatch):
    monkeypatch.delenv("ATMEE_API_KEY", raising=False)
    with pytest.raises(AtmeeException, match="ATMEE_API_KEY"):
        AtmeeAPI()


def test_refuses_plain_http():
    with pytest.raises(AtmeeException, match="https"):
        AtmeeAPI(api_key="k", api_url="http://api.example.com")
    AtmeeAPI(api_key="k", api_url="http://localhost:8080")
