"""Pluggable brain implementations for the Mumble Voice Bot.

Each brain determines how the bot responds to utterances:
- EchoBrain: Clone speaker voice, echo transcript (parrot)
- LLMBrain: Full LLM-powered responses with tool calling
- ReactiveBrain: Fillers, echo fragments, deflections (no LLM)
- AdaptiveBrain: Routes between LLMBrain and ReactiveBrain based on brain_power
- NullBrain: Never responds (transcribe-only mode) -- defined in interfaces/brain.py
"""

from mumble_voice_bot.interfaces.brain import BotResponse, Brain, NullBrain, Utterance, VoiceConfig

__all__ = [
    "Brain",
    "Utterance",
    "BotResponse",
    "VoiceConfig",
    "NullBrain",
]
