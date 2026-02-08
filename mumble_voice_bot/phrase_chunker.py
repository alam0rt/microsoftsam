"""Buffer LLM tokens and emit speakable phrase chunks."""

import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhraseChunker:
    """
    Accumulate streamed tokens and emit when ready for TTS.

    Strategy:
    - Emit on sentence-ending punctuation (. ! ?)
    - Emit on clause punctuation (, ; :) if buffer is long enough
    - Force emit after max_chars or timeout

    This enables the TTS to start generating audio before the full
    LLM response is complete, reducing time-to-first-audio.

    Usage:
        chunker = PhraseChunker()

        async for token in llm.chat_stream(messages):
            phrase = chunker.add(token)
            if phrase:
                # Send phrase to TTS
                async for audio in tts.synthesize_streaming(phrase):
                    yield audio

        # Flush any remaining text
        remaining = chunker.flush()
        if remaining:
            async for audio in tts.synthesize_streaming(remaining):
                yield audio

    Args:
        min_chars: Minimum chars before considering emit on punctuation.
        max_chars: Force emit at this length regardless of punctuation.
        timeout_ms: Force emit after this delay with no new tokens.
    """

    min_chars: int = 30  # Minimum chars before considering emit
    max_chars: int = 150  # Force emit at this length
    timeout_ms: int = 400  # Force emit after this delay

    # Punctuation patterns
    sentence_end: str = r'[.!?]'
    clause_end: str = r'[,;:]'

    _buffer: str = ""
    _last_add_time: float = field(default_factory=time.time)

    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer.

        Args:
            text: New text (usually a token or small chunk from LLM).

        Returns:
            A phrase to send to TTS, or None if still buffering.
        """
        self._buffer += text
        self._last_add_time = time.time()

        # Check for sentence end
        if len(self._buffer) >= self.min_chars:
            if re.search(self.sentence_end + r'\s*$', self._buffer):
                return self.flush()

        # Check for clause end with longer buffer
        if len(self._buffer) >= self.min_chars * 2:
            if re.search(self.clause_end + r'\s*$', self._buffer):
                return self.flush()

        # Force flush at max length
        if len(self._buffer) >= self.max_chars:
            return self.flush()

        return None

    def check_timeout(self) -> Optional[str]:
        """
        Check if we should emit due to timeout.

        Call this periodically when no new tokens are arriving.

        Returns:
            Buffered text if timeout exceeded, None otherwise.
        """
        if self._buffer and (time.time() - self._last_add_time) * 1000 > self.timeout_ms:
            return self.flush()
        return None

    def flush(self) -> str:
        """Force flush and return accumulated text.

        Returns:
            All buffered text, stripped of leading/trailing whitespace.
        """
        text = self._buffer.strip()
        self._buffer = ""
        return text

    def has_content(self) -> bool:
        """Check if there's buffered content.

        Returns:
            True if buffer has non-whitespace content.
        """
        return bool(self._buffer.strip())

    def peek(self) -> str:
        """Peek at current buffer contents without flushing.

        Returns:
            Current buffer contents.
        """
        return self._buffer

    def reset(self):
        """Reset the chunker state."""
        self._buffer = ""
        self._last_add_time = time.time()


@dataclass
class SentenceChunker:
    """
    Simpler chunker that only emits on sentence boundaries.

    More conservative than PhraseChunker - only emits complete sentences.
    Better for TTS quality, but slightly higher latency.
    """

    max_chars: int = 200  # Force emit at this length

    _buffer: str = ""
    _sentence_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r'([.!?]+)\s*')
    )

    def add(self, text: str) -> Optional[str]:
        """Add text and return a complete sentence if available."""
        self._buffer += text

        # Look for sentence ending
        match = self._sentence_pattern.search(self._buffer)
        if match:
            # Get everything up to and including the punctuation
            end_idx = match.end()
            sentence = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:]
            return sentence

        # Force flush if too long
        if len(self._buffer) >= self.max_chars:
            return self.flush()

        return None

    def flush(self) -> str:
        """Flush remaining buffer."""
        text = self._buffer.strip()
        self._buffer = ""
        return text

    def has_content(self) -> bool:
        """Check if there's buffered content."""
        return bool(self._buffer.strip())

    def reset(self):
        """Reset the chunker."""
        self._buffer = ""
