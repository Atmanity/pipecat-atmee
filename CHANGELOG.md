# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- `AtmeeVideoService`: renders an Atmee avatar (v1 or v2) lip-synced to the bot's TTS
  and pushes its video and audio into the pipeline, for any Pipecat transport.
  Interruptions stop the avatar mid-sentence; the session ends with the
  pipeline.
- Atmee-hosted media room by default (only an Atmee API key needed), or your
  own LiveKit project via `livekit_url` / `livekit_api_key` / `livekit_api_secret`.
- v2 avatars: the start retries while the avatar warms up on a rendering
  machine (`avatar_warmup_timeout`, `AtmeeAvatarWarmingError`), and the
  default `avatar_join_timeout` (330 s) covers a cold start.
- `AtmeeDailyVideoService` (extra `daily`): on `DailyTransport` the avatar joins
  the bot's Daily room as its own participant and publishes there directly.
- `AtmeeAPI`: create v1 avatars from a portrait, start, read and end avatar sessions.
- Examples: `examples/bot.py` (SmallWebRTC / Daily / LiveKit), `examples/create_avatar.py`.
