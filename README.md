# pipecat-atmee

Give your [Pipecat](https://github.com/pipecat-ai/pipecat) bot a face.
`AtmeeVideoService` renders an Atmee **v1 avatar**, a talking head generated
from a single portrait, lip-synced to your bot's speech. Your bot keeps its
own STT, LLM and TTS; Atmee only renders the face, and the avatar's video and
audio flow back into your pipeline to whatever transport you use (Daily,
SmallWebRTC, LiveKit, ...).

```python
from pipecat_atmee import AtmeeVideoService

avatar = AtmeeVideoService(avatar_id="...")   # ATMEE_API_KEY in the environment

pipeline = Pipeline([
    transport.input(), stt, user_aggregator, llm, tts,
    avatar,                                    # right after the TTS
    transport.output(), assistant_aggregator,
])
```

This integration is built and maintained by [Atmanity](https://atmee.ai), the
company behind Atmee.

## Installation

```bash
uv add pipecat-atmee      # or: pip install pipecat-atmee
```

Python 3.11+, `pipecat-ai` 1.8 or newer.

## Prerequisites

- An **Atmee API key** (`sk_atmee_...`) in `ATMEE_API_KEY`. Create one in the
  Atmee studio at [atmee.ai/studio/api-keys](https://www.atmee.ai/studio/api-keys).
  Keep it on your bot's side; it never belongs in a browser.
- An **avatar id**. Create an avatar from one portrait, once:

  ```python
  from pipecat_atmee import AtmeeAPI

  async with AtmeeAPI() as api:                  # reads ATMEE_API_KEY
      avatar_id = await api.create_avatar(name="Val", image="portrait.jpg")
  ```

  `image` may be a local file path, raw bytes, or an `https://` URL (`.jpg`,
  `.png`, `.webp`). A portrait-only avatar is ready immediately. Any avatar of
  your account in the [Atmee studio](https://atmee.ai) works too.

## Usage

Put `AtmeeVideoService` right after your TTS service and enable video output
on the transport at the avatar's size, **512x512 at 25 fps**:

```python
params = TransportParams(           # or DailyParams / LiveKitParams
    audio_in_enabled=True,
    audio_out_enabled=True,
    video_out_enabled=True,
    video_out_is_live=True,
    video_out_width=512,
    video_out_height=512,
    video_out_framerate=25,
)
```

What happens:

- When the pipeline starts, an Atmee avatar session starts (about a second;
  the first video follows a few seconds later).
- The TTS audio frames are consumed and streamed to the avatar. The avatar's
  lip-synced audio (`SpeechOutputAudioRawFrame`) and video
  (`OutputImageRawFrame`) are pushed downstream instead, so the user hears and
  sees the avatar speak in sync. Text frames pass through untouched, so your
  context aggregators work as usual.
- An interruption (`InterruptionFrame`, e.g. the user barging in) stops the
  avatar mid-sentence.
- When the pipeline ends or is cancelled, the session ends and billing stops.
  `max_duration_seconds` is a hard ceiling in any case.

A complete bot is in [`examples/bot.py`](examples/bot.py).

### Where the avatar is rendered

By default Atmee hosts the media room the avatar is rendered into: you need
nothing but your Atmee API key, and your transport can be anything. The
service joins that room itself and relays the avatar into your pipeline.

> **Note:** Atmee-hosted rooms are still being rolled out. Until they are
> enabled, render into your own LiveKit project as shown below; a LiveKit
> Cloud free project is enough.

If you already run a LiveKit project and want the avatar rendered there, pass
your credentials; the avatar's access token is minted locally, your LiveKit
secret never leaves your process:

```python
avatar = AtmeeVideoService(
    avatar_id="...",
    livekit_url=os.environ["LIVEKIT_URL"],
    livekit_api_key=os.environ["LIVEKIT_API_KEY"],
    livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
)
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `avatar_id` | required | The Atmee avatar to render. |
| `api_key`, `api_url` | `ATMEE_API_KEY`, `ATMEE_API_URL` | Credentials and API base (`https://api.atmanity.us`). Set `api_url` only to target another environment. |
| `max_duration_seconds` | `3600` | Hard ceiling of the session. |
| `metadata` | `None` | Free-form JSON attached to the session on the Atmee side. |
| `agent_identity` | `pipecat-agent` | Your bot's identity in the media room. |
| `livekit_url`, `livekit_api_key`, `livekit_api_secret` | `None` | Render into a room of your own LiveKit project instead of an Atmee-hosted one. |
| `livekit_room_name` | random | Room name in your own LiveKit project. |
| `avatar_participant_identity` | `atmee-avatar-agent` | The avatar's identity in your own LiveKit room. |
| `avatar_join_timeout` | `120` | How long the first speech waits for the avatar's video before it is dropped with an error. |
| `session` | `None` | An `aiohttp.ClientSession` to reuse for API calls. |

After start, `service.session_id` holds the Atmee session id.

### Events

```python
@avatar.event_handler("on_avatar_connected")
async def on_avatar_connected(service, session_id): ...

@avatar.event_handler("on_avatar_disconnected")   # the avatar left unexpectedly
async def on_avatar_disconnected(service, session_id): ...

@avatar.event_handler("on_playback_finished")
async def on_playback_finished(service, playback_position, interrupted): ...
```

### Errors

A session that cannot start (bad key, unknown avatar, no credits, no rendering
capacity) pushes an `ErrorFrame` marking the service unusable; the pipeline
worker's `ProcessorUnusablePolicy` decides whether the pipeline goes on. The
API client raises `AtmeeException` (with `status_code` and `code`),
`AtmeeNoCapacityError` (with the server's `retry_after` hint) and
`AtmeeAvatarNotReadyError`.

## Running the example

```bash
git clone https://github.com/Atmanity/pipecat-atmee && cd pipecat-atmee
cp examples/.env.example .env           # fill in the keys
uv run examples/create_avatar.py --name Val --image portrait.jpg   # once; put the id in .env
uv run examples/bot.py                  # then open http://localhost:7860 and click Connect
```

`uv run examples/bot.py --transport daily` runs the same bot on Daily
(needs `DAILY_API_KEY`).

## Compatibility

Tested with Pipecat v1.12.0; CI also runs against v1.8.0, the oldest
supported version.

## License

Apache 2.0, see [LICENSE](LICENSE).
