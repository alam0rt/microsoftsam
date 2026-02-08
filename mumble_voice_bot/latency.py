"""Latency tracking and logging for voice pipeline."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnLatency:
    """Latency breakdown for a single voice turn."""

    turn_id: str
    user_id: str

    # Timestamps (seconds since epoch)
    t0_vad_start: float = 0.0
    t_vad_end: float = 0.0
    t_asr_start: float = 0.0
    t_asr_partial1: Optional[float] = None  # First partial (streaming ASR)
    t_asr_stable1: Optional[float] = None  # First stable text (streaming ASR)
    t_asr_final: float = 0.0
    t_llm_start: float = 0.0
    t_llm_first_token: Optional[float] = None  # First LLM token
    t_llm_complete: float = 0.0
    t_tts_start: float = 0.0
    t_tts_first_audio: float = 0.0
    t_playback_start: float = 0.0
    t_playback_end: float = 0.0

    # Streaming-specific timestamps
    t_llm_early_start: Optional[float] = None  # LLM started on partial transcript
    t_overlap_start: Optional[float] = None  # When ASR/LLM overlap began

    # Metadata
    transcript_length: int = 0
    response_length: int = 0
    audio_duration_ms: float = 0.0
    llm_started_early: bool = False
    stable_chars_at_llm_start: int = 0

    def compute_metrics(self) -> dict:
        """Compute derived latency metrics."""
        metrics = {}

        # Basic timings
        if self.t_vad_end > 0 and self.t0_vad_start > 0:
            metrics["vad_duration_ms"] = (self.t_vad_end - self.t0_vad_start) * 1000

        if self.t_asr_final > 0 and self.t_asr_start > 0:
            metrics["asr_total_ms"] = (self.t_asr_final - self.t_asr_start) * 1000

        # Streaming ASR metrics
        if self.t_asr_partial1 and self.t_asr_start > 0:
            metrics["asr_ttfp_ms"] = (self.t_asr_partial1 - self.t_asr_start) * 1000

        if self.t_asr_stable1 and self.t_asr_start > 0:
            metrics["asr_ttfs_ms"] = (self.t_asr_stable1 - self.t_asr_start) * 1000

        # LLM metrics
        if self.t_llm_complete > 0 and self.t_llm_start > 0:
            metrics["llm_total_ms"] = (self.t_llm_complete - self.t_llm_start) * 1000

        first_token_time = self.t_llm_first_token or self.t_llm_complete
        if first_token_time > 0 and self.t_llm_start > 0:
            metrics["llm_ttft_ms"] = (first_token_time - self.t_llm_start) * 1000

        # TTS metrics
        if self.t_tts_first_audio > 0 and self.t_tts_start > 0:
            metrics["tts_ttfa_ms"] = (self.t_tts_first_audio - self.t_tts_start) * 1000

        # End-to-end metrics
        if self.t_tts_first_audio > 0 and self.t_vad_end > 0:
            metrics["total_ttfa_ms"] = (self.t_tts_first_audio - self.t_vad_end) * 1000

        if self.t_playback_end > 0 and self.t0_vad_start > 0:
            metrics["total_turn_ms"] = (self.t_playback_end - self.t0_vad_start) * 1000

        # Overlap/early-start metrics
        if self.t_llm_early_start and self.t_vad_end > 0:
            # How much time we saved by starting early
            # (vad_end - llm_early_start) is the overlap
            overlap = self.t_vad_end - self.t_llm_early_start
            if overlap > 0:
                metrics["llm_overlap_ms"] = overlap * 1000
                metrics["llm_started_early"] = True

        if self.llm_started_early:
            metrics["llm_started_early"] = True
            metrics["stable_chars_at_llm_start"] = self.stable_chars_at_llm_start

        return metrics

    def to_json_line(self) -> str:
        """Return JSON line for logging."""
        data = asdict(self)
        data["metrics"] = self.compute_metrics()
        return json.dumps(data)

    def log(self):
        """Log this turn's latency."""
        metrics = self.compute_metrics()

        # Build log message parts
        parts = [f"Turn {self.turn_id}:"]

        if "total_ttfa_ms" in metrics:
            parts.append(f"TTFA={metrics['total_ttfa_ms']:.0f}ms")

        breakdown = []
        if "asr_total_ms" in metrics:
            asr_str = f"ASR={metrics['asr_total_ms']:.0f}ms"
            if "asr_ttfp_ms" in metrics:
                asr_str += f" (TTFP={metrics['asr_ttfp_ms']:.0f}ms)"
            breakdown.append(asr_str)

        if "llm_total_ms" in metrics:
            llm_str = f"LLM={metrics['llm_total_ms']:.0f}ms"
            if metrics.get("llm_started_early"):
                llm_str += " [early]"
            breakdown.append(llm_str)

        if "tts_ttfa_ms" in metrics:
            breakdown.append(f"TTS={metrics['tts_ttfa_ms']:.0f}ms")

        if breakdown:
            parts.append(f"({', '.join(breakdown)})")

        if "llm_overlap_ms" in metrics:
            parts.append(f"[overlap={metrics['llm_overlap_ms']:.0f}ms]")

        logger.info(" ".join(parts))


class LatencyLogger:
    """Log latency records (in-memory only by default)."""

    def __init__(self, path: Path = None, write_to_disk: bool = False):
        self.path = path or Path("latency.jsonl")
        self.write_to_disk = write_to_disk

    def log(self, turn: TurnLatency):
        """Log a turn's latency (optionally to disk)."""
        if self.write_to_disk:
            with open(self.path, "a") as f:
                f.write(turn.to_json_line() + "\n")
        turn.log()


class LatencyTracker:
    """Helper class to track latency for a single turn."""

    _counter: int = 0

    def __init__(self, user_id: str, logger: Optional[LatencyLogger] = None):
        LatencyTracker._counter += 1
        self.turn = TurnLatency(
            turn_id=f"turn_{LatencyTracker._counter}_{int(time.time())}",
            user_id=user_id,
        )
        self._logger = logger

    def vad_start(self):
        """Mark VAD start (user began speaking)."""
        self.turn.t0_vad_start = time.time()

    def vad_end(self):
        """Mark VAD end (user stopped speaking)."""
        self.turn.t_vad_end = time.time()

    def asr_start(self):
        """Mark ASR start (beginning transcription)."""
        self.turn.t_asr_start = time.time()

    def asr_partial(self):
        """Mark first partial ASR result (streaming ASR only)."""
        if self.turn.t_asr_partial1 is None:
            self.turn.t_asr_partial1 = time.time()

    def asr_stable(self):
        """Mark first stable text from ASR (streaming ASR only)."""
        if self.turn.t_asr_stable1 is None:
            self.turn.t_asr_stable1 = time.time()

    def asr_final(self, text: str = ""):
        """Mark ASR complete (final transcription)."""
        self.turn.t_asr_final = time.time()
        self.turn.transcript_length = len(text)

    def llm_start(self, early: bool = False, stable_chars: int = 0):
        """Mark LLM start (beginning generation).

        Args:
            early: Whether LLM is starting on partial transcript.
            stable_chars: Number of stable characters at start (if early).
        """
        self.turn.t_llm_start = time.time()
        if early:
            self.turn.t_llm_early_start = time.time()
            self.turn.llm_started_early = True
            self.turn.stable_chars_at_llm_start = stable_chars

    def llm_first_token(self):
        """Mark first LLM token (streaming only)."""
        if self.turn.t_llm_first_token is None:
            self.turn.t_llm_first_token = time.time()

    def llm_complete(self, response: str = ""):
        """Mark LLM complete (full response received)."""
        self.turn.t_llm_complete = time.time()
        self.turn.response_length = len(response)

    def tts_start(self):
        """Mark TTS start (beginning synthesis)."""
        self.turn.t_tts_start = time.time()

    def tts_first_audio(self):
        """Mark first TTS audio chunk."""
        if self.turn.t_tts_first_audio == 0.0:
            self.turn.t_tts_first_audio = time.time()

    def playback_start(self):
        """Mark playback start (audio sent to output)."""
        self.turn.t_playback_start = time.time()

    def playback_end(self, audio_duration_ms: float = 0.0):
        """Mark playback end (audio finished)."""
        self.turn.t_playback_end = time.time()
        self.turn.audio_duration_ms = audio_duration_ms

    def mark_overlap_start(self):
        """Mark when ASR/LLM overlap begins."""
        if self.turn.t_overlap_start is None:
            self.turn.t_overlap_start = time.time()

    def finalize(self) -> TurnLatency:
        """Finalize and optionally log the turn."""
        if self._logger:
            self._logger.log(self.turn)
        return self.turn

    def get_ttfa(self) -> float:
        """Get time-to-first-audio in milliseconds."""
        if self.turn.t_tts_first_audio and self.turn.t_vad_end:
            return (self.turn.t_tts_first_audio - self.turn.t_vad_end) * 1000
        return 0.0

    def get_metrics(self) -> dict:
        """Get current metrics without finalizing."""
        return self.turn.compute_metrics()
