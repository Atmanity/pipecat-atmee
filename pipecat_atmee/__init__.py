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

"""Atmee avatars for Pipecat: give your bot a face.

``AtmeeVideoService`` renders an Atmee avatar (v1 or v2) lip-synced to your
bot's TTS, and ``AtmeeAPI`` creates v1 avatars from a single portrait.
"""

from pipecat_atmee.api import (
    AtmeeAPI,
    AtmeeAvatarNotReadyError,
    AtmeeAvatarWarmingError,
    AtmeeException,
    AtmeeHostedRoomMissingError,
    AtmeeNoCapacityError,
    AvatarInfo,
    AvatarSessionInfo,
    ConnectOptions,
)
from pipecat_atmee.version import __version__
from pipecat_atmee.video import AtmeeVideoService, AtmeeVideoSettings

__all__ = [
    "AtmeeAPI",
    "AtmeeDailyVideoService",
    "AtmeeAvatarNotReadyError",
    "AtmeeAvatarWarmingError",
    "AtmeeException",
    "AtmeeHostedRoomMissingError",
    "AtmeeNoCapacityError",
    "AtmeeVideoService",
    "AtmeeVideoSettings",
    "AvatarInfo",
    "AvatarSessionInfo",
    "ConnectOptions",
    "__version__",
]


def __getattr__(name: str):
    # Imported on first use: it needs Pipecat's optional Daily transport.
    if name == "AtmeeDailyVideoService":
        from pipecat_atmee.daily import AtmeeDailyVideoService

        return AtmeeDailyVideoService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
