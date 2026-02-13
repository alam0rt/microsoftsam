"""Pluggable brain implementations for the Mumble Voice Bot.

Each brain determines how the bot responds to utterances:
- LLMBrain: Unified brain with LLM intelligence and reactive fallbacks.
  brain_power controls the mix (1.0=always LLM, 0.0=pure reactive).
- EchoBrain: Clone speaker voice, echo transcript (parrot)
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
