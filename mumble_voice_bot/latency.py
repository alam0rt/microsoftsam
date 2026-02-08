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
    t_asr_partial1: Optional[float] = None  # First partial (future streaming ASR)
    t_asr_final: float = 0.0
    t_llm_start: float = 0.0
    t_llm_first_token: Optional[float] = None  # Future: when streaming
    t_llm_complete: float = 0.0
    t_tts_start: float = 0.0
    t_tts_first_audio: float = 0.0
    t_playback_start: float = 0.0
    t_playback_end: float = 0.0

    # Metadata
    transcript_length: int = 0
    response_length: int = 0
    audio_duration_ms: float = 0.0

    def compute_metrics(self) -> dict:
        """Compute derived latency metrics."""
        return {
            "vad_duration_ms": (self.t_vad_end - self.t0_vad_start) * 1000,
            "asr_ms": (self.t_asr_final - self.t_asr_start) * 1000,
            "llm_ttft_ms": ((self.t_llm_first_token or self.t_llm_complete) - self.t_llm_start) * 1000,
            "llm_total_ms": (self.t_llm_complete - self.t_llm_start) * 1000,
            "tts_ttfa_ms": (self.t_tts_first_audio - self.t_tts_start) * 1000,
            "total_ttfa_ms": (self.t_tts_first_audio - self.t_vad_end) * 1000,
            "total_turn_ms": (self.t_playback_end - self.t0_vad_start) * 1000,
        }

    def to_json_line(self) -> str:
        """Return JSON line for logging."""
        data = asdict(self)
        data["metrics"] = self.compute_metrics()
        return json.dumps(data)

    def log(self):
        """Log this turn's latency."""
        metrics = self.compute_metrics()
        logger.info(
            f"Turn {self.turn_id}: TTFA={metrics['total_ttfa_ms']:.0f}ms "
            f"(ASR={metrics['asr_ms']:.0f}ms, LLM={metrics['llm_total_ms']:.0f}ms, "
            f"TTS={metrics['tts_ttfa_ms']:.0f}ms)"
        )


class LatencyLogger:
    """Append latency records to a JSONL file."""

    def __init__(self, path: Path = Path("latency.jsonl")):
        self.path = path

    def log(self, turn: TurnLatency):
        """Append a turn's latency to the log file."""
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

    def asr_final(self, text: str = ""):
        """Mark ASR complete (final transcription)."""
        self.turn.t_asr_final = time.time()
        self.turn.transcript_length = len(text)

    def llm_start(self):
        """Mark LLM start (beginning generation)."""
        self.turn.t_llm_start = time.time()

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
