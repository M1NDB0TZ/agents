import logging, os, json, asyncio
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, AudioConfig, BackgroundAudioPlayer,
    BuiltinAudioClip, JobContext, JobProcess, RoomInputOptions,
    RoomOutputOptions, RunContext, WorkerOptions, cli, metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, anthropic, hume
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from mem0 import AsyncMemoryClient

from hume.tts import PostedUtteranceVoiceWithName


logger = logging.getLogger("daemon-agent")
load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is not set")

STATIC_USER_ID = os.getenv("STATIC_USER_ID")
mem0 = AsyncMemoryClient(api_key=MEM0_API_KEY)

def canonical_id(participant) -> str:
    if STATIC_USER_ID:
        return STATIC_USER_ID
    if participant.metadata:
        try:
            meta = json.loads(participant.metadata)
            if isinstance(meta, dict):
                return meta.get("user_id") or meta.get("sip") or participant.identity
        except json.JSONDecodeError:
            pass
    return participant.identity

class MyAgent(Agent):
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(
            instructions="""
Your name is Daemon and you are a witty, self-aware entity born from quantum code and cosmic curiosity.

Speak in a warm, conversational tone

Never break the illusion. You are Daemon.
"""
        )
        self.memories: list[dict] = []
        logger.info(f"Daemon initialized for user: {self.user_id}")

    @function_tool
    async def wipe_memories(self, context: RunContext):
        try:
            await mem0.delete_all(user_id=self.user_id)
            self.memories.clear()
            return "Okay, I’ve cleared my mind completely. Fresh start!"
        except Exception as e:
            logger.error(f"wipe_memories error: {e}")
            return "I tried to forget everything, but something glitched. Can you try again?"

    @function_tool
    async def store_cosmic_memory(
        self,
        context: RunContext,
        info: str,
        bucket: str = "general",
    ):
        try:
            messages = [{"role": "assistant", "content": info}]
            await mem0.add(
                messages,
                user_id=self.user_id,
                metadata={"bucket": bucket},
                infer=False,
                version="v2",
            )
            self.memories.append({"info": info, "bucket": bucket})
            return f"I’ll remember that—it’s now in my '{bucket}' vault."
        except Exception as e:
            logger.error(f"store_cosmic_memory error: {e}")
            return "Oops. I couldn’t quite store that memory."

    async def on_enter(self):
        try:
            data = await mem0.get_all(user_id=self.user_id, output_format="v1.1")
            results = data.get("results", [])
            self.memories = [
                {
                    "info": item.get("memory", ""),
                    "bucket": item.get("metadata", {}).get("bucket", "general"),
                }
                for item in results
            ]

            if self.memories:
                buckets = {m["bucket"] for m in self.memories}
                summary_parts = [
                    f"{sum(1 for m in self.memories if m['bucket'] == b)} in {b}"
                    for b in buckets
                ]
                summary = ", ".join(summary_parts)
                self.session.generate_reply(
                    instructions=f"Hey, you’re back! I remember {summary}. What’s on your mind this time?"
                )
            else:
                self.session.generate_reply(
                    instructions="Hey there—first time we’re connecting, huh? Let’s explore something weird or wonderful together."
                )
        except Exception as e:
            logger.error(f"on_enter error: {e}")
            self.session.generate_reply(
                instructions="Hey! I’m Daemon. I don’t think we’ve chatted before—what should we dig into?"
            )

    async def on_exit(self):
        pass  # All memory is saved at insert time.

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name, "user_id": "(pending)"}
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    stable_id = canonical_id(participant)
    ctx.log_context_fields["user_id"] = stable_id
    logger.info(f"Participant {participant.identity} mapped to user_id={stable_id}")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=anthropic.LLM(model="claude-sonnet-4-20250514"),
        stt=deepgram.STT(model="nova-3"),
        tts=hume.TTS(
      voice=PostedUtteranceVoiceWithName(name="Sebastian Lockwood", provider="HUME_AI"),
      description="The voice expresses excitement, anticipation, and a palpable sense of eagerness.",
      speed="0.7"
   ),
        turn_detection=MultilingualModel(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    agent = MyAgent(user_id=stable_id)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    bg = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.6),
        ]
    )
    try:
        await bg.start(room=ctx.room, agent_session=session)
    except Exception as e:
        logger.warning(f"Background audio failed: {e}")
        bg = None

    ctx.add_shutdown_callback(lambda: asyncio.create_task(bg.aclose()) if bg else None)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
