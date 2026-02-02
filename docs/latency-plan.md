# Latency Optimization Plan for microsoftsam Voice Bot

This document outlines a detailed implementation plan for achieving low-latency speech-to-speech interaction on the T1000 GPU (Turing, CC 7.5, 8GB VRAM).

---

## Current Implementation Status

### ✅ Already Implemented

| Component | File(s) | Status |
|-----------|---------|--------|
| **Main Pipeline Entrypoint** | `mumble_tts_bot.py`, `mumble_voice_bot/pipeline.py` | Complete |
| **Wyoming STT Client** | `mumble_voice_bot/providers/wyoming_stt.py` | Complete (async + sync wrappers) |
| **Wyoming TTS Client** | `mumble_voice_bot/providers/wyoming_tts.py` | Complete (streaming support) |
| **vLLM/OpenAI-compatible LLM Client** | `mumble_voice_bot/providers/openai_llm.py` | Complete (non-streaming only) |
| **TTS (LuxTTS)** | `mumble_tts_bot.py#StreamingLuxTTS` | Sentence-level streaming |
| **VAD (RMS-based)** | `mumble_tts_bot.py#on_sound_received` | Basic threshold + hold timer |
| **Basic Latency Logging** | `mumble_tts_bot.py` | First chunk timing only |
| **Pluggable Interfaces** | `mumble_voice_bot/interfaces/{llm,stt,tts}.py` | Complete abstractions |

### ❌ Not Yet Implemented

| Component | Status |
|-----------|--------|
| **Structured Latency KPIs (JSON logs)** | Missing |
| **VAD-driven barge-in / cancellation** | Missing |
| **Transcript Stabilizer (for streaming ASR)** | Missing |
| **PhraseChunker for LLM→TTS** | Missing |
| **LLM Streaming Responses** | Not implemented |
| **GPU STT via containerized Wyoming** | Missing |
| **Alternative streaming ASR (sherpa-onnx)** | Missing |

---

## Key Latency Metric: Time-to-First-Audio (TTFA)

The metric users feel most is **TTFA**: time from when they stop speaking until they hear the bot's first audio.

```
TTFA = t_asr + t_llm_first_token + t_tts_first_chunk

Where:
  t_asr            = speech_end → transcript_ready
  t_llm_first_token = transcript_ready → first LLM token
  t_tts_first_chunk = first complete phrase → first audio chunk
```

### Current Bottleneck Analysis

Based on your NixOS config:
- **STT (CPU Whisper)**: Likely **1-3 seconds** for typical utterances — **BIGGEST BOTTLENECK**
- **LLM (vLLM, non-streaming)**: ~200-500ms for short responses
- **TTS (LuxTTS GPU)**: ~200-400ms to first audio chunk

---

## Phase 0 — Latency Instrumentation

**Goal:** Add structured logging to measure and optimize each stage.

### Deliverable: `LatencyTracker` class

Create `mumble_voice_bot/latency.py`:

```python
"""Latency tracking and logging for voice pipeline."""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

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
```

### Deliverable: Latency analysis script

Create `scripts/analyze_latency.py`:

```python
#!/usr/bin/env python3
"""Analyze latency logs and compute percentiles."""

import json
import sys
from pathlib import Path
from statistics import median, quantiles

def analyze(path: Path):
    ttfa_values = []
    asr_values = []
    llm_values = []
    tts_values = []
    
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            metrics = data.get("metrics", {})
            if metrics.get("total_ttfa_ms"):
                ttfa_values.append(metrics["total_ttfa_ms"])
                asr_values.append(metrics.get("asr_ms", 0))
                llm_values.append(metrics.get("llm_total_ms", 0))
                tts_values.append(metrics.get("tts_ttfa_ms", 0))
    
    if not ttfa_values:
        print("No latency data found")
        return
    
    def stats(values, name):
        if len(values) < 2:
            return
        q = quantiles(values, n=100)
        print(f"{name}:")
        print(f"  p50: {q[49]:.0f}ms")
        print(f"  p90: {q[89]:.0f}ms")
        print(f"  p99: {q[98]:.0f}ms")
        print()
    
    print(f"Analyzed {len(ttfa_values)} turns\n")
    stats(ttfa_values, "Total TTFA")
    stats(asr_values, "ASR")
    stats(llm_values, "LLM")
    stats(tts_values, "TTS TTFA")

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("latency.jsonl")
    analyze(path)
```

### Files to modify

- `mumble_tts_bot.py`: Integrate `TurnLatency` tracking in:
  - `on_sound_received()`: Set `t0_vad_start`, `t_vad_end`
  - `_process_speech()`: Set `t_asr_start`, `t_asr_final`
  - `_maybe_respond()`: Set `t_llm_start`, `t_llm_complete`
  - `_speak_sync()`: Set `t_tts_start`, `t_tts_first_audio`, `t_playback_end`

---

## Phase 1 — Make the Pipeline Actually Streaming

### Phase 1.1 — VAD-driven Barge-in Controller

**Problem:** Currently the bot doesn't stop talking when the user interrupts.

**Solution:** Add cancellation support.

Create `mumble_voice_bot/turn_controller.py`:

```python
"""Turn-taking and barge-in control."""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TurnState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class TurnController:
    """Manages turn-taking with barge-in support."""
    
    state: TurnState = TurnState.IDLE
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _current_user: Optional[str] = None
    
    def start_listening(self, user_id: str):
        """User started speaking."""
        self._current_user = user_id
        self.state = TurnState.LISTENING
        self._cancel_event.clear()
    
    def start_processing(self):
        """Processing user's speech."""
        self.state = TurnState.PROCESSING
    
    def start_speaking(self):
        """Bot is now speaking."""
        self.state = TurnState.SPEAKING
    
    def request_barge_in(self) -> bool:
        """
        Called when user speaks while bot is outputting.
        Returns True if barge-in was triggered.
        """
        if self.state == TurnState.SPEAKING:
            self._cancel_event.set()
            self.state = TurnState.LISTENING
            return True
        return False
    
    def is_cancelled(self) -> bool:
        """Check if current generation should be cancelled."""
        return self._cancel_event.is_set()
    
    def reset(self):
        """Reset to idle state."""
        self._cancel_event.clear()
        self._current_user = None
        self.state = TurnState.IDLE
```

### Modifications to `mumble_tts_bot.py`

```python
# In MumbleVoiceBot.__init__:
self.turn_controller = TurnController()

# In on_sound_received(), add barge-in detection:
if self._speaking.is_set() and rms > self.asr_threshold:
    if self.turn_controller.request_barge_in():
        print(f"[Barge-in] User {user_name} interrupted bot")
        # Clear the Mumble output buffer
        self.mumble.sound_output.clear()

# In _speak_sync(), check for cancellation:
for wav_chunk in self.tts.generate_speech_streaming(...):
    if self.turn_controller.is_cancelled():
        print("[TTS] Cancelled due to barge-in")
        break
    # ... rest of playback
```

---

### Phase 1.2 — Transcript Stabilizer (Prep for Streaming ASR)

This component is preparation for Phase 2B (streaming ASR).

Create `mumble_voice_bot/transcript_stabilizer.py`:

```python
"""Stabilize partial ASR results for streaming pipelines."""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TranscriptStabilizer:
    """
    Track partial transcripts and emit stable prefixes.
    
    Streaming ASR models often revise their output. This component
    maintains a stable prefix that won't change, allowing the LLM
    to start processing before transcription is complete.
    """
    
    stability_window: int = 2  # Partials before text is "stable"
    _history: deque = field(default_factory=lambda: deque(maxlen=3))
    _stable_prefix: str = ""
    _emitted_length: int = 0
    
    def update(self, partial: str) -> tuple[str, str, bool]:
        """
        Process a partial transcript.
        
        Args:
            partial: The current partial transcript.
            
        Returns:
            Tuple of (stable_delta, unstable_tail, is_final).
            - stable_delta: New stable text to forward to LLM
            - unstable_tail: Text that may still change
            - is_final: Whether this appears to be final
        """
        self._history.append(partial)
        
        if len(self._history) < self.stability_window:
            return "", partial, False
        
        # Find common prefix across recent partials
        common = self._find_common_prefix(list(self._history))
        
        # Only emit text we haven't emitted before
        new_stable = common[self._emitted_length:]
        self._emitted_length = len(common)
        
        unstable = partial[len(common):]
        
        return new_stable, unstable, False
    
    def finalize(self, final: str) -> str:
        """
        Called when ASR signals end of utterance.
        Returns any remaining text not yet emitted.
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
        """Find the longest common prefix among strings."""
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
```

---

### Phase 1.3 — LLM Streaming + PhraseChunker

**Problem:** Current LLM client waits for complete response before TTS can start.

#### Deliverable 1: Add streaming to `OpenAIChatLLM`

Add to `mumble_voice_bot/providers/openai_llm.py`:

```python
import json
from typing import AsyncIterator

# Add this method to the OpenAIChatLLM class:

async def chat_stream(
    self,
    messages: list[dict],
    context: dict | None = None
) -> AsyncIterator[str]:
    """Stream chat completion tokens.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        context: Optional context dict (unused).
        
    Yields:
        Text chunks as they arrive from the API.
    """
    headers = self._build_headers()
    body = self._build_request_body(messages)
    body["stream"] = True
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            self.endpoint,
            headers=headers,
            json=body,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                    
                data = line[6:]  # Strip "data: " prefix
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        # Filter out <think> tags for models like Qwen3
                        if "<think>" not in content and "</think>" not in content:
                            yield content
                except json.JSONDecodeError:
                    continue
```

Also update the interface in `mumble_voice_bot/interfaces/llm.py`:

```python
from typing import AsyncIterator

# Add to LLMProvider ABC:

async def chat_stream(
    self,
    messages: list[dict],
    context: dict | None = None
) -> AsyncIterator[str]:
    """Stream chat completion tokens.
    
    Default implementation falls back to non-streaming.
    """
    response = await self.chat(messages, context)
    yield response.content
```

#### Deliverable 2: PhraseChunker

Create `mumble_voice_bot/phrase_chunker.py`:

```python
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
    - Force emit after max_tokens or timeout
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
            text: New text (usually a token or small chunk).
            
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
        """Force flush and return accumulated text."""
        text = self._buffer.strip()
        self._buffer = ""
        return text
    
    def has_content(self) -> bool:
        """Check if there's buffered content."""
        return bool(self._buffer.strip())
```

#### Deliverable 3: Streaming pipeline

Add to `mumble_voice_bot/pipeline.py`:

```python
async def process_audio_streaming(
    self,
    audio,
    sample_rate: int,
    user_id: str = "default",
    voice_prompt: dict = None,
    num_steps: int = 4,
) -> AsyncIterator[tuple[str, any]]:
    """
    Process audio with streaming LLM and TTS for minimal latency.
    
    Yields:
        Tuples of (event_type, data) where event_type is one of:
        - "transcription": TranscriptionResult
        - "llm_chunk": str (partial LLM response)
        - "tts_audio": audio tensor
        - "complete": PipelineResult
    """
    latency = {}
    
    # Step 1: Transcribe (still blocking for now)
    start = time.time()
    transcription = await self.transcribe(audio, sample_rate)
    latency["transcription"] = time.time() - start
    
    should_respond, cleaned_text = self._should_respond(transcription.text)
    if not should_respond:
        return
    
    transcription.text = cleaned_text
    yield ("transcription", transcription)
    
    # Step 2: Stream LLM response
    self._add_to_history(user_id, "user", cleaned_text)
    history = self._get_history(user_id)
    
    start = time.time()
    chunker = PhraseChunker()
    full_response = ""
    first_token_time = None
    
    async for token in self.llm.chat_stream(history):
        if first_token_time is None:
            first_token_time = time.time()
            latency["llm_ttft"] = first_token_time - start
        
        full_response += token
        yield ("llm_chunk", token)
        
        # Check if we have a phrase ready for TTS
        phrase = chunker.add(token)
        if phrase and voice_prompt is not None:
            async for audio_chunk in self.synthesize_streaming(
                phrase, voice_prompt, num_steps
            ):
                yield ("tts_audio", audio_chunk)
    
    # Flush any remaining text
    remaining = chunker.flush()
    if remaining and voice_prompt is not None:
        async for audio_chunk in self.synthesize_streaming(
            remaining, voice_prompt, num_steps
        ):
            yield ("tts_audio", audio_chunk)
    
    latency["llm_total"] = time.time() - start
    
    # Add to history
    self._add_to_history(user_id, "assistant", full_response)
    
    yield ("complete", PipelineResult(
        transcription=transcription,
        llm_response=LLMResponse(content=full_response),
        audio=None,  # Streamed, not collected
        latency=latency,
    ))
```

---

## Phase 2 — Fix STT Latency

**This is the biggest latency bottleneck.** CPU Whisper can take 1-3+ seconds.

### Phase 2A (Recommended) — GPU Wyoming via Container

Your NixOS config has Wyoming on CPU due to CUDA build issues. Solution: Run GPU-enabled Wyoming in a container.

#### Option 1: Podman/Docker container

```nix
# Add to your NixOS configuration

virtualisation.oci-containers.backend = "podman";  # or "docker"

virtualisation.oci-containers.containers.wyoming-whisper-gpu = {
  image = "rhasspy/wyoming-whisper:latest";
  ports = [ "10300:10300" ];
  
  environment = {
    # Aggressive speed settings
    WHISPER_MODEL = "tiny-int8";  # Fastest model
    WHISPER_LANGUAGE = "en";  # Skip language detection
    WHISPER_BEAM_SIZE = "1";  # Greedy decoding
  };
  
  extraOptions = [
    "--device=nvidia.com/gpu=all"  # For podman with CDI
    # Or for docker: "--gpus=all"
  ];
};

# Update your Wyoming service to connect to container
# Or just point your bot's --wyoming-stt-host to localhost:10300
```

#### Option 2: Nix overlay for proper CUDA build

If you want native NixOS, you may need an overlay that properly builds `faster-whisper` with CUDA. This is more complex but avoids containers.

### Phase 2B (Future) — Streaming ASR with sherpa-onnx

For even lower latency, consider a true streaming ASR that emits partial results.

Create `mumble_voice_bot/providers/sherpa_stt.py`:

```python
"""Streaming ASR using sherpa-onnx."""

# This is a stub for future implementation
# sherpa-onnx provides streaming models like:
# - Zipformer transducer models
# - Paraformer models
# - Whisper in streaming mode

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.transcript_stabilizer import TranscriptStabilizer


class SherpaSpeechRecognizer(STTProvider):
    """
    Streaming ASR using sherpa-onnx.
    
    Provides partial results with ~100-200ms latency instead of
    waiting for complete utterance.
    """
    
    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        provider: str = "cuda",  # or "cpu"
    ):
        # TODO: Initialize sherpa-onnx recognizer
        self.stabilizer = TranscriptStabilizer()
        raise NotImplementedError("sherpa-onnx integration not yet implemented")
    
    async def transcribe_streaming(
        self,
        audio_stream,
        sample_rate: int = 16000,
    ):
        """
        Transcribe audio stream, yielding partial results.
        
        Yields:
            Tuples of (stable_text, is_final)
        """
        # TODO: Feed audio to sherpa-onnx, get partials
        # Use self.stabilizer to emit stable prefixes
        raise NotImplementedError()
```

---

## Phase 3 — vLLM Tuning for Voice

### Voice-optimized configuration

Update `config.yaml`:

```yaml
llm:
  endpoint: "http://localhost:8002/v1/chat/completions"
  model: "LiquidAI/LFM2.5-1.2B-Instruct"
  
  # Voice-optimized settings
  max_tokens: 80  # Keep responses short (1-2 sentences)
  temperature: 0.3  # More consistent output
  
  system_prompt: |
    You are a voice assistant in a Mumble chat. Follow these rules strictly:
    
    1. Respond in 1-2 short sentences maximum
    2. Never use bullet points, lists, or formatting
    3. Never use emojis or special characters
    4. Speak naturally as if in conversation
    5. If uncertain, give a brief acknowledgment
    6. Avoid technical jargon unless asked
```

### NixOS vLLM tuning

Test with vs without CUDA graphs:

```nix
# Current (eager mode - safer but potentially slower)
services.vllm.settings.serverArgs = {
  "enforce-eager" = true;
};

# Alternative (CUDA graphs - faster after warmup, needs more VRAM)
# Comment out enforce-eager and benchmark:
# services.vllm.settings.serverArgs = {
#   "disable-log-stats" = true;
# };
```

For voice workloads on T1000:
- Keep `maxNumSeqs = 2` (single user focus)
- Keep `maxModelLen = 2048` (short context is fine for voice)
- `chunkedPrefill = true` is good for memory

---

## Phase 4 — TTS Improvements

### Improve sentence splitting for earlier chunks

Update `split_into_sentences()` in `mumble_tts_bot.py`:

```python
def split_into_sentences(text: str, max_chars: int = 120) -> List[str]:
    """
    Split text into speakable chunks optimized for streaming TTS.
    
    Strategy:
    - Split on sentence boundaries first
    - Split long sentences on clause boundaries
    - Ensure minimum chunk size for natural speech
    """
    MIN_CHUNK = 20  # Don't create tiny chunks
    
    # First pass: split on sentence endings
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split long sentences on clause boundaries
            clause_pattern = r'(?<=[,;:])\s+'
            clauses = re.split(clause_pattern, sentence)
            
            # Merge tiny clauses
            current = ""
            for clause in clauses:
                if len(current) + len(clause) < MIN_CHUNK:
                    current += (" " if current else "") + clause
                else:
                    if current:
                        chunks.append(current)
                    current = clause
            if current:
                chunks.append(current)
    
    return [c.strip() for c in chunks if c.strip()]
```

### Add playback jitter buffer (optional)

For smoother audio when chunks arrive irregularly:

```python
# In _speak_sync(), add a small pre-buffer:

import collections

def _speak_sync_buffered(self, text: str, voice_prompt: dict, pipeline_start: float = None):
    """Generate and play speech with jitter buffer."""
    BUFFER_CHUNKS = 2  # Pre-buffer this many chunks
    
    chunk_queue = collections.deque()
    generator = self.tts.generate_speech_streaming(text, voice_prompt, num_steps=self.num_steps)
    
    # Pre-fill buffer
    for _ in range(BUFFER_CHUNKS):
        try:
            chunk = next(generator)
            chunk_queue.append(chunk)
        except StopIteration:
            break
    
    # Play while generating
    first_chunk = True
    while chunk_queue:
        wav_chunk = chunk_queue.popleft()
        
        if first_chunk:
            # Log first audio timing here
            first_chunk = False
        
        # Convert and play
        wav_float = wav_chunk.numpy().squeeze()
        wav_float = np.clip(wav_float, -1.0, 1.0)
        pcm = (wav_float * 32767).astype(np.int16)
        self.mumble.sound_output.add_sound(pcm.tobytes())
        
        # Refill buffer
        try:
            chunk_queue.append(next(generator))
        except StopIteration:
            pass
```

---

## Implementation Priority

| Priority | Phase | Task | Impact | Effort |
|----------|-------|------|--------|--------|
| 🔴 1 | 2A | GPU STT (container) | **Very High** | Medium |
| 🔴 2 | 1.3 | LLM streaming | **High** | Medium |
| 🟡 3 | 1.1 | Barge-in support | Medium | Low |
| 🟡 4 | 0 | Latency instrumentation | Medium | Low |
| 🟢 5 | 3 | vLLM/prompt tuning | Low-Medium | Low |
| 🟢 6 | 4 | TTS chunking | Low | Low |
| ⚪ 7 | 1.2, 2B | Streaming ASR | Future | High |

---

## Expected Latency Improvements

| Stage | Current (est.) | After Phase 2A | After All Phases |
|-------|----------------|----------------|------------------|
| ASR | 1500-3000ms | 200-500ms | 150-400ms |
| LLM TTFT | 300-500ms | 300-500ms | 150-300ms (streaming) |
| TTS TTFA | 300-500ms | 300-500ms | 200-400ms |
| **Total TTFA** | **2100-4000ms** | **800-1500ms** | **500-1100ms** |

The biggest single improvement comes from getting STT onto GPU (Phase 2A).

---

## Files to Create

```
mumble_voice_bot/
├── latency.py           # Phase 0: Latency tracking
├── turn_controller.py   # Phase 1.1: Barge-in support
├── transcript_stabilizer.py  # Phase 1.2: For streaming ASR
├── phrase_chunker.py    # Phase 1.3: LLM→TTS chunking
└── providers/
    └── sherpa_stt.py    # Phase 2B: Streaming ASR (future)

scripts/
└── analyze_latency.py   # Phase 0: Latency analysis

docs/
└── latency-plan.md      # This document
```

## Files to Modify

- `mumble_tts_bot.py`: Integrate all components
- `mumble_voice_bot/pipeline.py`: Add streaming pipeline
- `mumble_voice_bot/providers/openai_llm.py`: Add `chat_stream()`
- `mumble_voice_bot/interfaces/llm.py`: Add streaming interface
- NixOS configuration: Add GPU Wyoming container
