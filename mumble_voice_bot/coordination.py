"""SharedBotServices - Coordination for multi-bot deployments.

Extracted from mumble_tts_bot.py. Handles:
- Event journal (shared timeline of events for LLM context)
- Response claim tracking (prevents multiple bots responding to same utterance)
- Speaking coordination (tracks which bots are currently speaking)
- Utterance broadcasting (bot-to-bot communication without ASR)
- Voice prompt loading and caching
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class SharedBotServices:
    """Container for shared TTS/STT/LLM services used by multiple bots.

    This allows running multiple bot instances that share expensive
    resources like neural network models, while coordinating their
    behavior to avoid conflicts.

    Attributes:
        tts: Shared TTS engine.
        stt: Shared STT engine.
        llm: Shared LLM client.
        device: Compute device being used.
        voice_prompts: Dict of persona_name -> pre-computed voice tensors.
        echo_filter: Shared echo filter so all bots know what all bots said.
    """

    def __init__(
        self,
        tts: Any = None,
        stt: Any = None,
        llm: Any = None,
        device: str = "cuda",
        echo_filter: Any = None,
    ):
        self.tts = tts
        self.stt = stt
        self.llm = llm
        self.device = device
        self.voice_prompts: dict[str, dict] = {}

        # Shared echo filter
        self.echo_filter = echo_filter

        # Speaking coordination
        self._speaking_count = 0
        self._speaking_lock = threading.Lock()

        # Event journal - shared timeline of events for LLM context
        self._event_journal: list[dict] = []
        self._journal_lock = threading.Lock()
        self._journal_max_entries = 50
        self._journal_max_age = 300.0  # 5 minutes
        self._start_time = time.time()

        # Response claim tracking
        self._response_claims: dict[tuple, float] = {}
        self._claims_lock = threading.Lock()
        self._claim_expiry = 10.0

        # Utterance listeners (for bot-to-bot communication)
        self._utterance_listeners: list[Callable] = []

    # =========================================================================
    # Response Claiming
    # =========================================================================

    def try_claim_response(self, user_id: int, text: str) -> bool:
        """Try to claim the right to respond to a user utterance.

        Prevents multiple responders from piling on to the same utterance.

        Args:
            user_id: The user who spoke.
            text: The transcribed text.

        Returns:
            True if we can respond, False if someone already started.
        """
        now = time.time()
        text_key = text[:50] if text else ""
        claim_key = (user_id, text_key)

        with self._claims_lock:
            # Clean expired claims
            expired = [k for k, v in self._response_claims.items() if now - v > self._claim_expiry]
            for k in expired:
                del self._response_claims[k]

            if claim_key in self._response_claims:
                logger.info(f"[CLAIM] REJECTED: user={user_id}, text='{text_key[:30]}...'")
                return False
            self._response_claims[claim_key] = now
            logger.info(f"[CLAIM] SUCCESS: user={user_id}, text='{text_key[:30]}...'")
            return True

    # =========================================================================
    # Speaking Coordination
    # =========================================================================

    def bot_started_speaking(self) -> None:
        """Called when a bot starts speaking."""
        with self._speaking_lock:
            self._speaking_count += 1
            logger.debug(f"Bot started speaking (count: {self._speaking_count})")

    def bot_stopped_speaking(self) -> None:
        """Called when a bot stops speaking."""
        with self._speaking_lock:
            self._speaking_count = max(0, self._speaking_count - 1)
            logger.debug(f"Bot stopped speaking (count: {self._speaking_count})")

    def any_bot_speaking(self) -> bool:
        """Check if any bot in this shared group is speaking."""
        with self._speaking_lock:
            return self._speaking_count > 0

    # =========================================================================
    # Event Journal
    # =========================================================================

    def log_event(self, event_type: str, speaker: str = None, content: str = None) -> None:
        """Log an event to the shared journal.

        Args:
            event_type: "user_message", "bot_message", "user_joined", etc.
            speaker: Who triggered the event.
            content: The message content (for message events).
        """
        now = time.time()
        entry = {
            "event": event_type,
            "speaker": speaker,
            "content": content,
            "time": now,
        }

        with self._journal_lock:
            self._event_journal.append(entry)
            # Prune old entries
            cutoff = now - self._journal_max_age
            self._event_journal = [e for e in self._event_journal if e["time"] > cutoff]
            if len(self._event_journal) > self._journal_max_entries:
                self._event_journal = self._event_journal[-self._journal_max_entries:]

    def get_journal_for_llm(self, max_events: int = 30) -> list[dict]:
        """Get the journal formatted for LLM context.

        Returns events with seconds_ago instead of absolute timestamps.
        """
        now = time.time()
        with self._journal_lock:
            events = list(self._event_journal)

        if not events:
            return []

        events = events[-max_events:]

        result = []
        for e in events:
            entry = {
                "event": e["event"],
                "speaker": e.get("speaker"),
                "seconds_ago": int(now - e["time"]),
            }
            if e.get("content"):
                entry["content"] = e["content"]
            result.append(entry)

        return result

    def get_recent_messages_for_llm(self, max_messages: int = 20, bot_name: str = None) -> list[dict]:
        """Get recent messages formatted as OpenAI-style messages.

        In multi-bot mode, only the requesting bot's own messages are "assistant".
        Other bots' messages become "user" with name prefix.
        """
        with self._journal_lock:
            events = list(self._event_journal)

        messages = []
        for e in events:
            event_type = e["event"]
            speaker = e.get('speaker', 'Unknown')
            content = e.get('content', '')

            if event_type == "user_message":
                messages.append({
                    "role": "user",
                    "content": f"{speaker}: {content}" if speaker else content,
                    "time": e["time"],
                })
            elif event_type == "text_message":
                messages.append({
                    "role": "user",
                    "content": f"{speaker} (text): {content}" if speaker else content,
                    "time": e["time"],
                })
            elif event_type == "bot_message":
                if bot_name and speaker != bot_name:
                    messages.append({
                        "role": "user",
                        "content": f"{speaker}: {content}",
                        "time": e["time"],
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "time": e["time"],
                    })

        messages = messages[-max_messages:]

        for m in messages:
            del m["time"]

        return messages

    # =========================================================================
    # Utterance Broadcasting
    # =========================================================================

    def broadcast_utterance(self, speaker_name: str, text: str) -> None:
        """Broadcast an utterance to all listening bots.

        Called by a bot when it starts speaking. Other bots receive this
        as a "perfect transcription" without needing ASR.
        """
        self.log_event("bot_message", speaker_name, text)
        logger.info(
            f"[BROADCAST] {speaker_name} speaking: '{text[:50]}...'"
            if len(text) > 50
            else f"[BROADCAST] {speaker_name} speaking: '{text}'"
        )

        for callback in self._utterance_listeners:
            try:
                callback(speaker_name, text)
            except Exception:
                pass

    def register_utterance_listener(self, callback: Callable) -> None:
        """Register a callback to receive utterances from other bots."""
        self._utterance_listeners.append(callback)

    # =========================================================================
    # Voice Loading
    # =========================================================================

    def load_voice(self, name: str, audio_path: str, voices_dir: str = "voices") -> dict:
        """Load and cache a voice prompt.

        Args:
            name: Name for this voice (cache key).
            audio_path: Path to reference audio file.
            voices_dir: Directory for cached voice tensors.

        Returns:
            Voice prompt dict with tensors.
        """
        if name in self.voice_prompts:
            return self.voice_prompts[name]

        if self.tts is None:
            raise RuntimeError("TTS not initialized")

        os.makedirs(voices_dir, exist_ok=True)
        ref_name = os.path.splitext(os.path.basename(audio_path))[0]
        cache_path = os.path.join(voices_dir, f"{ref_name}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached voice: {cache_path}")
            voice = torch.load(cache_path, weights_only=False, map_location=self.device)
        else:
            logger.info(f"Encoding reference: {audio_path}")
            voice = self.tts.encode_prompt(audio_path, rms=0.01)
            torch.save(voice, cache_path)
            logger.info(f"Cached voice as '{ref_name}'")

        # Ensure tensors on correct device
        voice = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in voice.items()
        }

        self.voice_prompts[name] = voice
        return voice
