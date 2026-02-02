"""Stabilize partial ASR results for streaming pipelines."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptStabilizer:
    """
    Track partial transcripts and emit stable prefixes.
    
    Streaming ASR models often revise their output. This component
    maintains a stable prefix that won't change, allowing the LLM
    to start processing before transcription is complete.
    
    The stabilizer uses a sliding window approach: text is considered
    "stable" when it appears consistently across multiple partial results.
    
    Usage:
        stabilizer = TranscriptStabilizer()
        
        # For each partial from streaming ASR:
        stable_delta, unstable, is_final = stabilizer.update(partial_text)
        if stable_delta:
            # Send stable_delta to LLM immediately
            llm.stream(stable_delta)
        
        # When ASR signals end of utterance:
        remaining = stabilizer.finalize(final_text)
        if remaining:
            llm.stream(remaining)
    
    Args:
        stability_window: Number of consistent partials before text is "stable".
                         Higher = more stable but higher latency.
        min_stable_chars: Minimum characters before emitting stable text.
                         Prevents sending very short fragments.
    """
    
    stability_window: int = 2  # Partials before text is "stable"
    min_stable_chars: int = 10  # Minimum chars before emitting
    _history: deque = field(default_factory=lambda: deque(maxlen=3))
    _stable_prefix: str = ""
    _emitted_length: int = 0
    
    def update(self, partial: str) -> tuple[str, str, bool]:
        """
        Process a partial transcript.
        
        Args:
            partial: The current partial transcript from streaming ASR.
            
        Returns:
            Tuple of (stable_delta, unstable_tail, is_final).
            - stable_delta: New stable text to forward to LLM (may be empty)
            - unstable_tail: Text that may still change
            - is_final: Whether this appears to be final (always False here)
        """
        self._history.append(partial)
        
        if len(self._history) < self.stability_window:
            return "", partial, False
        
        # Find common prefix across recent partials
        common = self._find_common_prefix(list(self._history))
        
        # Only emit text we haven't emitted before
        new_stable = common[self._emitted_length:]
        
        # Ensure minimum stable length to avoid tiny fragments
        if len(new_stable) < self.min_stable_chars:
            new_stable = ""
        else:
            self._emitted_length = len(common)
        
        unstable = partial[self._emitted_length:]
        
        return new_stable, unstable, False
    
    def finalize(self, final: str) -> str:
        """
        Called when ASR signals end of utterance.
        
        Returns any remaining text not yet emitted as stable.
        
        Args:
            final: The final complete transcript.
            
        Returns:
            Remaining text that wasn't emitted during streaming.
        """
        remaining = final[self._emitted_length:]
        self.reset()
        return remaining
    
    def reset(self):
        """Reset state for new utterance."""
        self._history.clear()
        self._stable_prefix = ""
        self._emitted_length = 0
    
    def _find_common_prefix(self, strings: list[str]) -> str:
        """Find the longest common prefix among strings.
        
        Uses word boundaries to avoid splitting words mid-way.
        
        Args:
            strings: List of partial transcripts to compare.
            
        Returns:
            Longest common prefix, aligned to word boundaries.
        """
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix) and prefix:
                # Back off to last word boundary
                space_idx = prefix.rfind(' ')
                if space_idx > 0:
                    prefix = prefix[:space_idx]
                else:
                    prefix = prefix[:-1]
        
        return prefix
    
    def get_stable_text(self) -> str:
        """Get all stable text emitted so far.
        
        Returns:
            All text that has been marked as stable.
        """
        if self._history:
            return self._history[-1][:self._emitted_length]
        return ""
    
    def get_full_partial(self) -> str:
        """Get the most recent full partial transcript.
        
        Returns:
            Most recent partial, or empty string if none.
        """
        if self._history:
            return self._history[-1]
        return ""


@dataclass
class StreamingTranscriptBuffer:
    """
    Higher-level buffer for accumulating streaming ASR results.
    
    Combines TranscriptStabilizer with additional buffering logic
    for use in the voice pipeline.
    """
    
    stabilizer: TranscriptStabilizer = field(default_factory=TranscriptStabilizer)
    _accumulated_stable: str = ""
    _current_unstable: str = ""
    
    def add_partial(self, partial: str) -> Optional[str]:
        """
        Add a partial transcript.
        
        Args:
            partial: Partial transcript from streaming ASR.
            
        Returns:
            Stable delta to send to LLM, or None if nothing stable yet.
        """
        stable_delta, self._current_unstable, _ = self.stabilizer.update(partial)
        if stable_delta:
            self._accumulated_stable += stable_delta
            return stable_delta
        return None
    
    def finalize(self, final: str) -> str:
        """
        Finalize with the complete transcript.
        
        Args:
            final: Complete transcript from ASR.
            
        Returns:
            Any remaining text not yet emitted.
        """
        remaining = self.stabilizer.finalize(final)
        full_text = self._accumulated_stable + remaining
        
        # Reset for next utterance
        self._accumulated_stable = ""
        self._current_unstable = ""
        
        return remaining
    
    def get_accumulated(self) -> str:
        """Get all accumulated stable text."""
        return self._accumulated_stable
    
    def get_full_text(self) -> str:
        """Get accumulated stable + current unstable."""
        return self._accumulated_stable + self._current_unstable
    
    def reset(self):
        """Reset all state."""
        self.stabilizer.reset()
        self._accumulated_stable = ""
        self._current_unstable = ""
