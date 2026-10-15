# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- `AtmeeVideoService`: renders an Atmee v1 avatar lip-synced to the bot's TTS
  and pushes its video and audio into the pipeline, for any Pipecat transport.
  Interruptions stop the avatar mid-sentence; the session ends with the
  pipeline.
- Atmee-hosted media room by default (only an Atmee API key needed), or your
  own LiveKit project via `livekit_url` / `livekit_api_key` / `livekit_api_secret`.
- `AtmeeAPI`: create avatars from a portrait, start, read and end avatar sessions.
- Examples: `examples/bot.py` (SmallWebRTC / Daily / LiveKit), `examples/create_avatar.py`.
