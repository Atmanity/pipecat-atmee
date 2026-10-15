#
# Copyright 2026 Atmanity
#
# SPDX-License-Identifier: Apache-2.0
#

"""Create an Atmee avatar from a portrait and print its id.

    uv run examples/create_avatar.py --name Val --image portrait.jpg

``--image`` is a local .jpg/.png/.webp file or an https:// URL. Put the printed
id into ATMEE_AVATAR_ID for examples/bot.py.
"""

import argparse
import asyncio

from dotenv import load_dotenv

from pipecat_atmee import AtmeeAPI


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True, help="portrait file path or https:// URL")
    args = parser.parse_args()

    async with AtmeeAPI() as api:  # reads ATMEE_API_KEY
        avatar_id = await api.create_avatar(name=args.name, image=args.image)
    print(avatar_id)


if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(main())
