# Streaming ASR Latency Plan (Adopting Nemotron Streaming Improvements)

**Goal:** Port the streaming-first practices and architectural patterns from `pipecat-ai/nemotron-january-2026` into `alam0rt/microsoftsam` to achieve minimal ASR latency and enable early LLM start.

---

## 1. Reference: Streaming Improvements in `nemotron-january-2026`

Key design patterns and artifacts to mirror:

- **Streaming ASR client over WebSocket** with frame ordering fixes and partial result handling.
- **Cache-aware streaming ASR** (NeMo streaming utilities).
- **Pipeline overlap**: start LLM on stable ASR prefixes before final transcript.
- **Latency instrumentation**: track VAD/ASR/LLM/TTS milestones.
- **Streaming pipeline architecture documentation** to guide integration.

Relevant docs and code (source repo):
- `docs/streaming-pipeline-architecture.md`
- `pipecat_bots/nvidia_stt.py`
- `tests/test_streaming_blackwell.py`

---

## 2. Target Architecture for `microsoftsam`

### 2.1 Streaming ASR Pipeline (primary change)
Implement the ASR path as true streaming:

1. **Chunked audio ingestion** (80–160ms windows).
2. **Partial transcript emission** every chunk.
3. **Stable prefix tracking** to detect reliable transcription segments.
4. **LLM kickoff** once prefix exceeds threshold (e.g., 50 chars).

This aligns with your existing `process_audio_streaming_asr()` scaffolding in `mumble_voice_bot/pipeline.py`.

---

## 3. Step-by-Step Plan

### Phase A — Streaming ASR Integration

**A1. Add streaming ASR provider compatible with Nemotron-style streaming**
- Prefer **WebSocket streaming ASR** if available (to match `nvidia_stt.py`).
- Otherwise, implement **cache-aware streaming** locally via NeMo streaming APIs.

**A2. Stabilizer logic**
- Track stable prefixes; only start LLM when stable text length >= threshold.
- Use stabilization pattern similar to streaming ASR logic from nemotron.

**A3. Partial vs final events**
- Emit `asr_partial` events continuously.
- Emit `asr_final` when stream closes or VAD triggers end.

---

### Phase B — Pipeline Overlap (LLM early start)

**B1. Kick off LLM early**
- Start LLM task as soon as stable prefix threshold met.
- Continue feeding ASR into context until end of utterance.

**B2. Reconcile partial LLM**
- If transcript changes substantially after LLM start, either:
  - Abort & restart LLM, or
  - Allow LLM to finish and treat later ASR deltas as follow-up.

---

### Phase C — Performance Instrumentation

**C1. Expand latency markers**
- Use `LatencyTracker` to record:
  - VAD start/end
  - ASR start
  - ASR first partial
  - ASR final
  - LLM start
  - LLM first token

**C2. Build a latency report**
- Track TTFT and TTFA for each stage.
- Output metrics on completion of each user turn.

---

### Phase D — Testing & Validation

**D1. Cache-aware streaming test**
- Mirror `tests/test_streaming_blackwell.py` to ensure incremental inference works.

**D2. End-to-end latency test**
- Simulate utterances and measure TTFA.
- Ensure target: **< 800ms TTFA** for typical utterances.

---

## 4. Concrete Deliverables (Checklist)

- [x] Add streaming ASR provider (WebSocket or NeMo cache-aware).
- [x] Implement stable prefix tracking + LLM early start.
- [x] Expand latency instrumentation for streaming metrics.
- [x] Write integration test for streaming ASR pipeline.
- [x] Add a documented config profile for minimum latency.

---

## 5. Target Metrics

| Metric | Current | Target |
|--------|---------|--------|
| ASR TTFT | 1500–3000ms | **24–80ms** |
| ASR total | 2000–3000ms | **~200ms streaming** |
| Total TTFA | 2100–4000ms | **400–800ms** |

---

## 6. Notes

- The largest latency win is **overlap**, not just faster ASR.
- True streaming requires that partials and stable prefixes are emitted **without waiting for end of utterance**.
- ASR must operate in chunked streaming mode (not just chunked batch).

---

## 7. Next Actions (Suggested)

1. Decide on ASR provider path:
   - WebSocket streaming (closest to nemotron)
   - NeMo cache-aware streaming locally
2. Implement streaming provider + stable prefix logic.
3. Add latency instrumentation + benchmark.
4. Iterate on chunk size / prefix threshold.
