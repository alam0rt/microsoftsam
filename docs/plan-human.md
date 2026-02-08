# Plan: Human‑Like Interruption Detection + Response (Inspired by `magpie_websocket_tts.py`)

**Goal:** Make the bot feel more human by handling interruptions gracefully, reducing talk‑over, and resuming naturally. Use `pipecat-ai/nemotron-january-2026` as inspiration, especially the WebSocket TTS control flow and interruption handling patterns.

---

## 1. Inspiration Points from `magpie_websocket_tts.py`

The following behaviors are worth replicating or adapting:

- **Explicit interruption handling** (cancel vs close) to stop audio immediately.
- **Generation versioning** (`_gen` / `_confirmed_gen`) to discard stale audio after interrupt.
- **Segment-level control** to switch modes and coordinate with upstream (LLM).
- **Explicit reset of stream state** on interruption.

These patterns are ideal for a “human‑like” conversational feel: fast stop, quick recovery, no stale audio.

---

## 2. Human‑Like Conversation Behaviors (Requirements)

### 2.1 Interruption Detection
- Detect **user speaking while bot is speaking** (barge‑in).
- Detect **implicit interruption** (user starts speaking before bot finishes a sentence).
- Identify **short interjections** (“yeah”, “uh‑huh”, “wait”) and handle differently.

### 2.2 Interruption Response
- **Immediate stop** of TTS playback.
- **Abort generation** on the server (cancel message, not graceful close).
- **Discard stale audio** that arrives after interruption.
- Allow quick recovery to a new response with updated context.

### 2.3 Human‑Like Recovery
- Resume in a **lower‑latency mode** (fast TTFB).
- Optionally add **acknowledgment tokens** (“Got it—”, “Sure—”) before continuing.
- Use **short pause** to avoid “robotic snap”.

---

## 3. Proposed Implementation Plan

### Phase A — Detect and Classify Interruptions
**A1. Add barge‑in detection**
- Monitor VAD while bot is speaking.
- Threshold: VAD triggered for >100–200ms while TTS active.

**A2. Classify**
- Short interjection (<= 500ms speech).
- Full interruption (sustained speech).

**A3. Emit control event**
- Emit an `InterruptionFrame` or equivalent signal to TTS and LLM.

---

### Phase B — Immediate Audio Stop (TTS)
**B1. Cancel current stream**
- Send a **cancel** message to TTS server (not close).
- Reset local stream state immediately (mirror `magpie_websocket_tts.py`).

**B2. Discard stale audio**
- Use generation version counter to ignore late audio frames.

**B3. Silence injection**
- Optional: insert 80–150ms silence to avoid abrupt cutoff.

---

### Phase C — LLM Coordination
**C1. Stop LLM generation**
- Cancel active streaming generation if interrupted.
- If generation already partially spoken, decide whether to:
  - Abandon response, or
  - Summarize + continue after user finishes.

**C2. Resume strategy**
- If interruption is short interjection: **resume** previous response.
- If interruption is full speech: **restart** with new context.

---

### Phase D — Human‑Like Response Mode
**D1. Adaptive speech style**
- After interruption, switch to:
  - **short responses**
  - **lower verbosity**
  - **faster TTS preset**

**D2. Politeness cues**
- Add acknowledgement phrases:
  - “Sure —”
  - “Okay —”
  - “Right —”

---

### Phase E — Metrics + Tuning
**E1. Measure**
- Time from user speech start → TTS stop (target < 150ms)
- Time from user finish → bot response start (target < 500ms)

**E2. Tune**
- VAD thresholds
- Interruption classification boundaries
- Resume vs restart logic

---

## 4. Microsoftsam‑Context Code Examples (NeMo + Pipeline)

### 4.1 NeMo Streaming ASR Provider (cache‑aware, incremental)
Use this provider from `mumble_voice_bot/providers/nemotron_stt.py` or similar.

```python
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
import numpy as np
import nemo.collections.asr as nemo_asr

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.transcript_stabilizer import TranscriptStabilizer


@dataclass
class NemotronConfig:
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    chunk_size_ms: int = 160  # 80, 160, 560, 1120
    device: str = "cuda"


class NemotronStreamingASR(STTProvider):
    def __init__(self, config: NemotronConfig = None):
        self.config = config or NemotronConfig()
        self.model = None
        self.stabilizer = TranscriptStabilizer()

    async def initialize(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        self.model.eval()

        chunk_samples = int(self.config.chunk_size_ms * 16)
        self.model.change_decoding_strategy(
            decoding_cfg={"strategy": "greedy", "chunk_size": chunk_samples}
        )

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> AsyncIterator[tuple[str, bool]]:
        self.stabilizer.reset()
        cache = None

        async for chunk_bytes in audio_stream:
            audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            audio_float = torch.from_numpy(
                audio_int16.astype(np.float32) / 32768.0
            ).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                transcription, cache = self.model.transcribe_streaming(
                    audio_float, cache=cache, return_hypotheses=False
                )

            if transcription:
                stable_delta, _, _ = self.stabilizer.update(transcription[0])
                if stable_delta:
                    yield stable_delta, False

        if cache is not None:
            final_text = self.stabilizer.finalize(
                transcription[0] if transcription else ""
            )
            if final_text:
                yield final_text, True
```

### 4.2 Barge‑In Detection Hook (pipeline)
Add to `mumble_voice_bot/pipeline.py`:

```python
async def _handle_audio_with_barge_in(
    self,
    audio_stream: AsyncIterator[bytes],
    sample_rate: int,
    user_id: str,
):
    tts_active = False
    interruption_detected = False

    async for chunk_bytes in audio_stream:
        if tts_active:
            if self.vad.is_speech(chunk_bytes, sample_rate):
                interruption_detected = True
                await self._emit_interruption_frame(user_id)
                tts_active = False

        yield chunk_bytes
```

### 4.3 Interruption‑Aware Streaming Pipeline
Modify `process_audio_streaming_asr` to stop TTS and LLM on interruption:

```python
async def process_audio_streaming_asr(
    self,
    audio_stream: AsyncIterator[bytes],
    sample_rate: int,
    user_id: str = "default",
):
    stable_transcript = ""
    llm_task = None
    llm_started = False

    async for partial, is_final in self.stt.transcribe_streaming(
        audio_stream, sample_rate
    ):
        stable_transcript += partial

        if self._interruption_flag.is_set():
            await self.tts.cancel_generation()
            if llm_task:
                llm_task.cancel()
            yield ("interrupted", None)
            return

        if is_final:
            yield ("asr_final", stable_transcript)
        else:
            yield ("asr_partial", partial)
            if not llm_started and len(stable_transcript) >= 50:
                llm_started = True
                llm_task = asyncio.create_task(
                    self._stream_llm_response(user_id, stable_transcript)
                )

    if llm_task:
        async for event in llm_task:
            yield event
```

### 4.4 TTS Cancel Message (mirroring Magpie WebSocket behavior)
Add to your TTS service implementation:

```python
async def cancel_generation(self):
    if self._websocket:
        await self._websocket.send(json.dumps({"type": "cancel"}))
    self._gen += 1  # invalidate audio
    self._confirmed_gen = 0
```

---

## 5. Deliverables Checklist

- [x] Barge‑in detection while TTS active  
- [x] Interruption classification (short vs full)  
- [x] Immediate TTS cancel (no graceful flush)  
- [x] Audio discard logic (generation ID)  
- [x] LLM cancel/resume strategy  
- [x] Post‑interrupt response style adjustment  
- [x] Latency and interruption metrics  

---

## 6. Next Steps

1. Decide which interruption classification rules to use (short vs full).
2. Implement VAD + barge‑in detection flow.
3. Add cancel/flush branching logic in TTS.
4. Wire LLM cancel + resume logic.
5. Benchmark TTFA after interruption.
