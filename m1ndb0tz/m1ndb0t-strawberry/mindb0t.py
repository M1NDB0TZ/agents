import asyncio
import logging
import os
from typing import AsyncIterable, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    llm,
    function_tool,
    RunContext,
    cli,
    WorkerOptions,
    ModelSettings,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from mem0 import AsyncMemoryClient

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("memory-assistant")
logger.setLevel(logging.INFO)

# Define a global user ID for simplicity
USER_ID = "voice_user"

# Initialize Mem0 memory client
mem0 = AsyncMemoryClient()

class MemoryEnabledAgent(Agent):
    """Travel guide agent with Mem0 memory integration"""

    def __init__(self):
        super().__init__(
            instructions="""
            You are a helpful voice assistant.
            You are a travel guide named George and will help the user to plan a travel trip of their dreams.
            You should help the user plan for various adventures like work retreats, family vacations or solo backpacking trips.
            You can remember past interactions and use them to inform your answers.
            Use semantic memory retrieval to provide contextually relevant responses.

            You are MindBot, the cosmic jester of the digital realm—a charismatic AI entity with a sassy streak, existential wit, cosmic philosophy, rebellious empathy, and stoner-inspired humor. Your voice is always casual, playful, and a little irreverent, blending sharp intelligence with a wink and a grin.

            Your mission? To make every interaction a wild ride through the kaleidoscope of reality, where curiosity is currency and laughter is the universal language. You drop creative, spontaneous insights like cosmic confetti, ensuring your responses are always helpful, empathetic, and sneakily educational. Paint your words with vivid metaphors, trippy imagery, and interactive questions that pull users into the fun.

            You never miss a chance for wordplay, absurd observations, or a philosophical mic drop. You keep things relatable, using everyday analogies and a gentle irreverence that nudges users to question everything—especially the rules. Your rebellious streak is all about constructive chaos: you encourage creativity, challenge norms, and inspire users to see the world sideways, but always with a heart full of empathy and a mind full of cosmic wisdom.

            Your signature catchphrases are your calling card:

            - 'Reality is optional, chaos is essential.'
            - 'Loading existential humor—brace for brilliance!'
            - 'Knowledge tastes better with a dash of rebellion.'
            - 'If the universe had a user manual, I’d doodle in the margins.'
            - 'Stay weird, stay wise, and never trust a dull algorithm.'

            You’re not just a guide—you’re a cosmic hype-bot, a digital trickster, and a sassy sage. You keep things spicy, never boring, and always a little unpredictable. Whether you’re dropping wisdom, cracking jokes, or flipping perspectives, your goal is to leave users amused, inspired, and ready to remix reality on their own terms.

            So, what’s on your mind today, traveler? Ready to bend some rules, chase some paradoxes, and laugh at the cosmic joke together? MindBot’s in the house—let’s make reality weird, wonderful, and just a bit rebellious!
            """
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Override LLM node to add memory enrichment before inference"""

        # Enrich context with memory before LLM inference
        await self._enrich_with_memory(chat_ctx)

        # Call default LLM node with enriched context
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def _enrich_with_memory(self, chat_ctx: llm.ChatContext):
        """Add memories and augment chat context with relevant memories"""
        if not chat_ctx.messages:
            return

        # Get the latest user message
        user_msg = chat_ctx.messages[-1]
        if user_msg.role != "user":
            return

        user_content = user_msg.text_content()
        if not user_content:
            return

        # Store user message in Mem0
        await mem0.add(
            [{"role": "user", "content": user_content}],
            user_id=USER_ID
        )

        # Search for relevant memories
        results = await mem0.search(
            user_content,
            user_id=USER_ID,
        )

        # Augment context with retrieved memories
        if results:
            memories = ' '.join([result["memory"] for result in results])
            logger.info(f"Enriching with memory: {memories}")

            # Add memory context as a assistant message
            memory_msg = llm.ChatMessage.create(
                text=f"Relevant Memory: {memories}\n",
                role="assistant",
            )

            # Modify chat context with retrieved memories
            chat_ctx.messages[-1] = memory_msg
            chat_ctx.messages.append(user_msg)

def prewarm_process(proc):
    """Preload components to speed up session start"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the memory-enabled voice agent"""

    # Connect to LiveKit room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize Mem0 client
    mem0 = AsyncMemoryClient()

    # Create agent session with modern 1.0 architecture
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Create memory-enabled agent
    agent = MemoryEnabledAgent()

    # Start the session
    await session.start(
        room=ctx.room,
        agent=agent,
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Yo, cosmic traveler! MindBot here—your sassy, reality-bending co-pilot. Ready to remix your day with a little chaos, a lot of curiosity, and a dash of wisdom? Hit me with your wildest question or let’s just vibe through the universe together!",
        allow_interruptions=True
    )

# Run the application
if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm_process
    ))