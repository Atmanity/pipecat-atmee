#
# Copyright 2026 Atmanity
#
# SPDX-License-Identifier: Apache-2.0
#

"""A voice bot with an Atmee avatar face.

Deepgram STT -> OpenAI LLM -> OpenAI TTS -> Atmee avatar -> transport.
The bot's TTS drives the avatar; the transport sends the avatar's lip-synced
video and audio to the user.

Run with the Pipecat development runner:

    uv run examples/bot.py                      # SmallWebRTC, open http://localhost:7860
    uv run examples/bot.py --transport daily    # needs DAILY_API_KEY

Environment (see examples/.env.example): ATMEE_API_KEY, ATMEE_AVATAR_ID,
DEEPGRAM_API_KEY, OPENAI_API_KEY.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_atmee import AtmeeVideoService

load_dotenv(override=True)

# Atmee v1 avatars render 512x512 at 25 fps: send them at that size.
VIDEO = {
    "video_out_enabled": True,
    "video_out_is_live": True,
    "video_out_width": 512,
    "video_out_height": 512,
    "video_out_framerate": 25,
}


def _daily_params():
    from pipecat.transports.daily.transport import DailyParams

    return DailyParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO)


def _livekit_params():
    from pipecat.transports.livekit.transport import LiveKitParams

    return LiveKitParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO)


transport_params = {
    "daily": _daily_params,
    "livekit": _livekit_params,
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO),
}

SYSTEM_PROMPT = (
    "You are a friendly assistant shown as a talking avatar. Your replies are spoken "
    "aloud, so keep them short and avoid emojis, lists or special characters."
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Build and run the pipeline on the given transport."""
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model="gpt-4.1-mini", system_instruction=SYSTEM_PROMPT),
    )
    tts = OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAITTSService.Settings(voice="ash"),
    )

    # The avatar replaces the TTS audio with its own lip-synced audio + video.
    avatar = AtmeeVideoService(avatar_id=os.environ["ATMEE_AVATAR_ID"])

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            avatar,  # right after the TTS
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Greet the user in one short sentence."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point of the Pipecat runner (and Pipecat Cloud)."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
