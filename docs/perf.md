# Plan: Improve TTS latency, turn-taking, and overall responsiveness

## Goals
- Minimize time-to-first-audio (TTFA).
- Prevent stale responses from being spoken.
- Keep interactions responsive under load (multiple users, long replies).
- Make latency regressions visible and testable.

---

## Current pain points (repo-specific)
- Single TTS worker runs `_speak_sync` synchronously, which can create backlog.
- `_tts_queue` is unbounded; backlog grows when latency spikes.
- Stale responses are only dropped late (after queued) and do not prevent queue buildup.
- Stats are lifetime averages, masking regressions.
- LLM and TTS are tightly coupled to a single pipeline and share state without explicit turn coordination.

---

## Proposed architecture changes

### 1) Turn coordinator (authoritative turn ID)
**What:** Introduce a `turn_id` for each user utterance.  
**Why:** Ensures only latest valid response can be spoken.

**Details:**
- Add `self._turn_counter` (monotonic int).
- When a new user utterance is finalized, assign `turn_id`.
- Store `self._latest_turn_id[user_id]`.
- Attach `turn_id` to LLM response and TTS queue item.
- When TTS dequeues, drop if `turn_id < latest_turn_id`.

**Acceptance criteria:**
- Older responses never speak after a newer user utterance.

---

### 2) Bounded queues with drop policy
**What:** Replace `_tts_queue` with `queue.Queue(maxsize=N)` and add a drop policy.  
**Why:** Prevents backlog and progressive slowdown.

**Drop policies to consider:**
- **Drop-oldest**: discard queue head when full.
- **Drop-stale**: discard any item not matching latest `turn_id`.
- **Drop-nonpriority**: keep only latest per user.

**Acceptance criteria:**
- Under heavy load, queue size stays bounded and TTFA remains stable.

---

### 3) Split TTS synthesis and playback
**What:** Decouple generation from playback with separate queues.  
**Why:** Prevent synthesis time from blocking playback and reduce tail latency.

**Details:**
- `tts_synthesis_worker`: generates audio chunks (producer).
- `tts_playback_worker`: streams PCM to Mumble (consumer).
- Playback can proceed while synthesis continues for next chunk.

**Acceptance criteria:**
- TTFA improves; playback feels smoother with fewer long pauses.

---

### 4) Streaming LLM → sentence chunking → TTS
**What:** Stream partial LLM tokens and feed TTS sentence chunks.  
**Why:** Reduce TTFA and improve perceived responsiveness.

**Details:**
- Collect LLM tokens into sentence buffers.
- Push each sentence chunk to TTS queue with same `turn_id`.
- On barge-in, cancel pending chunks for that `turn_id`.

**Acceptance criteria:**
- First audio begins within 1–2 seconds for typical prompts.

---

### 5) Rolling latency metrics + percentiles
**What:** Track latencies in rolling windows (last N samples).  
**Why:** Detect regressions over time.

**Details:**
- Keep deques for ASR, LLM, TTS.
- Log p50/p95/p99 every 30s.
- Track queue length and drop count.

**Acceptance criteria:**
- Logs show percentile latency and drop rates.

---

### 6) Adaptive pacing (avoid fixed sleeps)
**What:** Replace fixed sleep during playback with buffer-aware pacing.  
**Why:** Avoid unnecessary delays when backlog is high.

**Details:**
- If playback queue is long, reduce sleep.
- If queue is empty, allow longer pauses for natural cadence.

**Acceptance criteria:**
- Pacing adapts under load; backlog drains faster.

---

## Test plan (QoS + regression)

### A) Unit tests
1. **Turn ID ordering**
   - Simulate two turns; ensure older turn is dropped when newer exists.
   - Validate TTS queue discards stale `turn_id`.

2. **Bounded queue behavior**
   - Fill queue beyond maxsize.
   - Verify drop policy triggers and logs correctly.

3. **Sentence chunking logic**
   - Feed token stream; ensure proper sentence segmentation.
   - Ensure each chunk inherits correct `turn_id`.

---

### B) Integration tests (local simulation)
1. **TTFA under load**
   - Simulate 10 rapid requests.
   - Assert TTFA < target threshold (e.g., 2.0s).

2. **Barge-in cancellation**
   - Start TTS, then simulate loud input.
   - Ensure TTS cancels and newer turn speaks.

3. **Backlog recovery**
   - Artificially delay TTS generation.
   - Ensure queue doesn’t grow beyond max and newer turns are prioritized.

---

### C) Performance benchmarks
1. **Latency distribution**
   - Collect 5–10 minutes of synthetic traffic.
   - Track p50/p95/p99 for ASR, LLM, TTS.

2. **Queue stability**
   - Measure queue length over time under steady load.
   - Verify maxsize + drop policy prevent runaway.

---

## Rollout phases
1. **Phase 1:** Turn ID + bounded queues + rolling metrics.
2. **Phase 2:** Streaming LLM + sentence chunking.
3. **Phase 3:** Split synthesis/playback workers + adaptive pacing.

---

## Success metrics
- TTFA ≤ 1.5–2.0s for typical prompts.
- p95 TTS latency stable (no upward drift).
- Queue length bounded (no unbounded backlog).
- Stale responses never spoken after newer input.
