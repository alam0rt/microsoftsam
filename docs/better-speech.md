Given your goal — a simple Python voice→voice bot where the stack itself is secondary and you care about human-like turn handling, interruptions, and latency — I’d strongly suggest focusing on a small, high-leverage subset rather than wading through all the full-duplex literature.

Below is a tight “essential set”: 2 papers + 2 repos + 1 concept that will give you 80% of the practical insight with minimal complexity.

🥇 Absolute Essentials (Read / Study First)
1. Voice Activity Projection (VAP) / Turn Prediction

📄 “Continuous Turn-Taking Prediction Using Voice Activity Projection”

(often referenced as VAP or VAP-style models)

Why this is essential

This is the key idea behind human-like turn taking.

Humans don’t wait for silence — they predict when someone is finishing.

You can approximate this without training a model using heuristics.

What to steal for your Python bot

Predict “user is done” before silence.

Start LLM generation early.

Cancel generation if speech resumes.

Minimal implementation idea

Audio stream →
  VAD (frame-level) →
    rolling probability(user continues) →
      threshold → “commit turn”


Even heuristic VAP beats silence-based EoT.

2. FlexDuo (State-Based Full-Duplex Control)

📄 FlexDuo: A Pluggable Full-Duplex Control Module

Why it’s essential

It shows that you do NOT need end-to-end speech models.

Natural interaction comes from explicit control states, not magic models.

Core idea
Use a tiny finite-state machine:

LISTENING
SPEAKING
INTERRUPTED
THINKING


Key insight

Interruptions are state transitions, not errors.

TTS must be cancelable at any frame boundary.

This paper maps directly to Python async code.

🥈 Best Practical Git Repos (Simple & Hackable)
3. RealtimeVoiceChat (Python, Minimal)

🔗 https://github.com/KoljaB/RealtimeVoiceChat

Why

Clean reference for:

streaming ASR

interruptible TTS

async audio loops

Easy to rip apart and rebuild.

What to study

How audio threads are decoupled from LLM inference.

How playback cancellation is handled.

This is ideal as a base skeleton, not a finished product.

4. TEN Framework (for ideas, not adoption)

🔗 https://github.com/TEN-framework/ten-framework

Why

Shows industry-style turn detection modules.

Has explicit:

VAD

barge-in

EoT detection

Overkill to use directly, but excellent for architecture ideas.

Steal concepts, not code.

🧠 One Concept You MUST Implement (Even Simply)
Incremental, Interruptible Generation

This matters more than model choice.

Golden rules

Generate LLM output incrementally

Stream TTS sentence or clause-level

Always allow:

cancel_current_tts()


Human-like behavior emerges when

The bot starts speaking early

Stops instantly when interrupted

Resumes with context (“sorry—go ahead”)

🧪 Minimal “Human-Like” Python Architecture

Here’s the simplest architecture that actually works:

Mic →
  VAD →
    partial ASR →
      turn predictor →
        LLM (streaming tokens) →
          chunker →
            TTS (interruptible)

Key tricks

ASR runs continuously, even during TTS

If ASR detects speech while speaking → interrupt

LLM generation runs in a cancellable asyncio task

🧭 What I Would Skip (For Now)

❌ End-to-end speech-to-speech models
❌ Heavy multimodal transformers
❌ Large dialogue management frameworks
❌ Reinforcement-learning dialogue policies

They add complexity without helping your current goal.

📌 If You Want a “Next Step”

After you implement:

interruptible TTS

predictive turn-taking

incremental generation

Then look at:

Backchannel generation (“mm-hmm”, “right”)

Prosodic cues (TTS speed / pitch during interruptions)

Confidence-based early starts

---

# Implementation Plan

Based on the research above and our current issues (echo detection, context drops, fragmented speech handling), here's the detailed implementation plan.

## Current Problems Identified

1. **Echo Detection**: Bot hears fragments of its own speech (e.g., "you" from "Thank you") and tries to respond
2. **Fragmented Speech**: Short utterances like "you" or "okay" trigger unnecessary responses  
3. **Context Loss on Soul Switch**: Conversation history resets when switching souls
4. **Turn Overlap**: Bot starts responding before user finishes speaking
5. **No State Machine**: Current code doesn't have clear states for LISTENING/SPEAKING/THINKING/INTERRUPTED

## Phase 1: Echo Filtering (HIGH PRIORITY)

### Problem
The bot's own TTS output is picked up by STT, causing self-responses. Example: Bot says "Thank you for asking" → STT hears "you" → Bot responds to "you".

### Solution: Recent Output Buffer

```python
# In handlers.py or pipeline.py
class EchoFilter:
    def __init__(self, decay_time: float = 3.0):
        self.recent_outputs: list[tuple[str, float]] = []  # (text, timestamp)
        self.decay_time = decay_time
    
    def add_output(self, text: str) -> None:
        """Record what the bot just said."""
        self.recent_outputs.append((text.lower(), time.time()))
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove old entries."""
        cutoff = time.time() - self.decay_time
        self.recent_outputs = [(t, ts) for t, ts in self.recent_outputs if ts > cutoff]
    
    def is_echo(self, transcript: str, threshold: float = 0.8) -> bool:
        """Check if transcript matches recent bot output."""
        self._cleanup()
        transcript_lower = transcript.lower().strip()
        
        # Check for exact substring match
        for output, _ in self.recent_outputs:
            if transcript_lower in output:
                return True
            # Also check if output words appear in transcript
            output_words = set(output.split())
            transcript_words = set(transcript_lower.split())
            if transcript_words and transcript_words.issubset(output_words):
                return True
        
        return False
```

### Integration Points
- `_process_speech()` in `handlers.py` - check before sending to LLM
- `_speak_response()` - record output after TTS completes
- Add to `AppConfig` - `echo_filter_decay: float = 3.0`

### Files to Modify
- `mumble_voice_bot/handlers.py` - Add EchoFilter class and integration
- `mumble_voice_bot/config.py` - Add echo filter config options

## Phase 2: Minimum Meaningful Utterance Filter

### Problem
Very short transcriptions ("okay", "um", "you") shouldn't trigger full LLM responses.

### Solution: Utterance Classifier

```python
class UtteranceClassifier:
    # Filler words that shouldn't trigger responses alone
    FILLERS = {"um", "uh", "hmm", "okay", "ok", "yeah", "yep", "nope", "mhm", "ah"}
    
    # Minimum word count for meaningful utterance
    MIN_WORDS = 2
    
    # Minimum character count
    MIN_CHARS = 5
    
    @classmethod
    def is_meaningful(cls, text: str) -> bool:
        """Determine if utterance warrants a response."""
        text = text.strip().lower()
        words = text.split()
        
        # Too short
        if len(text) < cls.MIN_CHARS:
            return False
        
        # Just filler words
        if all(w in cls.FILLERS for w in words):
            return False
        
        # Single word that's not a question/command
        if len(words) < cls.MIN_WORDS:
            # Allow single-word questions
            if text.endswith("?") or text in {"what", "why", "how", "when", "where", "who"}:
                return True
            return False
        
        return True
```

### Integration Points
- `_process_speech()` - filter before echo check
- Configurable via `min_utterance_words`, `min_utterance_chars`

## Phase 3: State Machine for Turn Handling

### Problem  
No clear state management leads to overlapping speech, interrupted responses, and confusion.

### Solution: ConversationStateMachine

```python
from enum import Enum, auto
from dataclasses import dataclass
import asyncio

class ConversationState(Enum):
    IDLE = auto()        # Not in conversation
    LISTENING = auto()   # Actively listening to user
    THINKING = auto()    # Processing/generating response  
    SPEAKING = auto()    # TTS playing
    INTERRUPTED = auto() # User interrupted bot speech
    COOLDOWN = auto()    # Brief pause after speaking

@dataclass
class StateTransition:
    from_state: ConversationState
    to_state: ConversationState
    timestamp: float
    reason: str

class ConversationStateMachine:
    def __init__(self):
        self.state = ConversationState.IDLE
        self.state_entered_at: float = time.time()
        self.transitions: list[StateTransition] = []
        self._lock = asyncio.Lock()
    
    async def transition(self, new_state: ConversationState, reason: str = "") -> bool:
        """Attempt state transition. Returns True if successful."""
        async with self._lock:
            # Define valid transitions
            valid_transitions = {
                ConversationState.IDLE: {ConversationState.LISTENING},
                ConversationState.LISTENING: {ConversationState.THINKING, ConversationState.IDLE},
                ConversationState.THINKING: {ConversationState.SPEAKING, ConversationState.LISTENING},
                ConversationState.SPEAKING: {ConversationState.COOLDOWN, ConversationState.INTERRUPTED},
                ConversationState.INTERRUPTED: {ConversationState.LISTENING},
                ConversationState.COOLDOWN: {ConversationState.LISTENING, ConversationState.IDLE},
            }
            
            if new_state not in valid_transitions.get(self.state, set()):
                logger.warning(f"Invalid transition: {self.state} -> {new_state}")
                return False
            
            self.transitions.append(StateTransition(
                from_state=self.state,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason
            ))
            
            self.state = new_state
            self.state_entered_at = time.time()
            return True
    
    def time_in_state(self) -> float:
        """How long we've been in current state."""
        return time.time() - self.state_entered_at
    
    @property
    def can_respond(self) -> bool:
        """Whether bot is allowed to start a response."""
        return self.state in {ConversationState.LISTENING}
    
    @property
    def is_speaking(self) -> bool:
        return self.state == ConversationState.SPEAKING
```

### Integration Points
- Replace `_is_bot_speaking` flag with state machine
- `_process_speech()` checks `can_respond` before processing
- `_speak_response()` manages THINKING → SPEAKING → COOLDOWN transitions
- Interruption detection triggers SPEAKING → INTERRUPTED → LISTENING

## Phase 4: Context Preservation on Soul Switch

### Problem
When switching souls, conversation context is lost because messages aren't preserved.

### Solution: Context-Preserving Soul Switch

```python
async def _switch_soul(self, soul_name: str, preserve_context: bool = True) -> str:
    """Switch to a different soul, optionally preserving conversation context."""
    
    # Store current context before switch
    preserved_messages = []
    if preserve_context and hasattr(self, '_context_messages'):
        # Keep user messages and bot responses, but not system prompts
        preserved_messages = [
            msg for msg in self._context_messages
            if msg.get('role') in ('user', 'assistant')
        ][-self.config.llm.context_messages:]  # Keep last N messages
    
    # Perform soul switch (loads new personality, voice, etc.)
    # ... existing switch logic ...
    
    # Restore context with new system prompt
    if preserved_messages:
        # New system prompt is already set by soul switch
        # Re-inject preserved conversation
        self._context_messages.extend(preserved_messages)
        logger.info(f"Preserved {len(preserved_messages)} messages across soul switch")
    
    return f"Switched to {soul_name}"
```

### Config Option
```yaml
souls:
  preserve_context_on_switch: true
  max_preserved_messages: 10
```

## Phase 5: ASR During TTS (Continuous Listening)

### Problem
Bot stops listening while speaking, missing interruptions and follow-ups.

### Solution: Keep ASR running, buffer results

```python
class ContinuousASRBuffer:
    """Buffer ASR results while bot is speaking."""
    
    def __init__(self):
        self.buffer: list[tuple[str, float]] = []  # (transcript, confidence)
        self.is_buffering = False
    
    def start_buffering(self):
        """Call when TTS starts."""
        self.is_buffering = True
        self.buffer.clear()
    
    def add_transcript(self, text: str, confidence: float = 1.0):
        """Add ASR result to buffer (if buffering)."""
        if self.is_buffering:
            self.buffer.append((text, confidence))
    
    def stop_buffering(self) -> list[tuple[str, float]]:
        """Call when TTS ends. Returns buffered transcripts."""
        self.is_buffering = False
        results = self.buffer.copy()
        self.buffer.clear()
        return results
    
    def check_for_interruption(self, threshold: float = 0.7) -> bool:
        """Check if buffered speech indicates interruption."""
        if not self.buffer:
            return False
        
        # High confidence speech during TTS = likely interruption
        max_confidence = max(conf for _, conf in self.buffer)
        total_words = sum(len(text.split()) for text, _ in self.buffer)
        
        return max_confidence > threshold and total_words >= 2
```

## Phase 6: Turn Prediction Heuristics

### Problem
Bot responds too quickly (interrupting) or too slowly (awkward pauses).

### Solution: Simple turn prediction without ML

```python
class TurnPredictor:
    """Predict when user has finished their turn."""
    
    # Indicators that user is done speaking
    TURN_END_MARKERS = {
        "?": 0.9,   # Questions strongly indicate turn end
        ".": 0.7,   # Statements somewhat indicate turn end
        "!": 0.8,   # Exclamations indicate turn end
    }
    
    # Phrases that indicate more coming
    CONTINUATION_PHRASES = [
        "and", "but", "so", "because", "however", "also",
        "first", "second", "then", "next", "finally"
    ]
    
    def __init__(self, base_delay: float = 0.3, max_delay: float = 1.5):
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def predict_turn_complete(self, transcript: str, silence_duration: float) -> float:
        """
        Returns confidence (0-1) that user is done speaking.
        
        Factors:
        - Punctuation at end
        - Silence duration
        - Trailing continuation words
        """
        text = transcript.strip()
        confidence = 0.5  # Base confidence
        
        # Check punctuation
        if text:
            last_char = text[-1]
            confidence += self.TURN_END_MARKERS.get(last_char, 0)
        
        # Check for continuation words
        words = text.lower().split()
        if words and words[-1] in self.CONTINUATION_PHRASES:
            confidence -= 0.4  # Likely more coming
        
        # Factor in silence
        silence_factor = min(silence_duration / self.max_delay, 1.0)
        confidence = confidence * 0.6 + silence_factor * 0.4
        
        return min(max(confidence, 0), 1)
    
    def get_response_delay(self, transcript: str) -> float:
        """Calculate how long to wait before responding."""
        text = transcript.strip()
        
        # Questions get fast responses
        if text.endswith("?"):
            return self.base_delay
        
        # Check for continuation indicators
        words = text.lower().split()
        if words and words[-1] in self.CONTINUATION_PHRASES:
            return self.max_delay
        
        # Default: moderate delay
        return self.base_delay + 0.2
```

---

# Testing Requirements

## Unit Tests

### test_echo_filter.py

```python
import pytest
import time
from mumble_voice_bot.handlers import EchoFilter

class TestEchoFilter:
    def test_detects_exact_echo(self):
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Thank you for asking about that")
        assert ef.is_echo("you") is True
        assert ef.is_echo("thank") is True
        assert ef.is_echo("Thank you") is True
    
    def test_ignores_non_echo(self):
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Hello there")
        assert ef.is_echo("goodbye") is False
        assert ef.is_echo("what time is it") is False
    
    def test_decay_removes_old_outputs(self):
        ef = EchoFilter(decay_time=0.1)
        ef.add_output("test phrase")
        time.sleep(0.2)
        assert ef.is_echo("test") is False
    
    def test_multiple_outputs(self):
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("First response")
        ef.add_output("Second response")
        assert ef.is_echo("first") is True
        assert ef.is_echo("second") is True
```

### test_utterance_classifier.py

```python
import pytest
from mumble_voice_bot.handlers import UtteranceClassifier

class TestUtteranceClassifier:
    def test_rejects_short_utterances(self):
        assert UtteranceClassifier.is_meaningful("ok") is False
        assert UtteranceClassifier.is_meaningful("um") is False
        assert UtteranceClassifier.is_meaningful("") is False
    
    def test_rejects_filler_words(self):
        assert UtteranceClassifier.is_meaningful("um hmm") is False
        assert UtteranceClassifier.is_meaningful("okay yeah") is False
    
    def test_accepts_questions(self):
        assert UtteranceClassifier.is_meaningful("what?") is True
        assert UtteranceClassifier.is_meaningful("why") is True
    
    def test_accepts_meaningful_speech(self):
        assert UtteranceClassifier.is_meaningful("tell me a joke") is True
        assert UtteranceClassifier.is_meaningful("what is the weather") is True
```

### test_conversation_state_machine.py

```python
import pytest
import asyncio
from mumble_voice_bot.handlers import ConversationStateMachine, ConversationState

class TestConversationStateMachine:
    @pytest.fixture
    def sm(self):
        return ConversationStateMachine()
    
    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self, sm):
        assert sm.state == ConversationState.IDLE
    
    @pytest.mark.asyncio
    async def test_valid_transition(self, sm):
        result = await sm.transition(ConversationState.LISTENING, "user started speaking")
        assert result is True
        assert sm.state == ConversationState.LISTENING
    
    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self, sm):
        # Can't go directly from IDLE to SPEAKING
        result = await sm.transition(ConversationState.SPEAKING, "invalid")
        assert result is False
        assert sm.state == ConversationState.IDLE
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, sm):
        # IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> LISTENING
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.THINKING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.COOLDOWN)
        await sm.transition(ConversationState.LISTENING)
        assert sm.state == ConversationState.LISTENING
        assert len(sm.transitions) == 5
    
    @pytest.mark.asyncio
    async def test_interruption_flow(self, sm):
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.THINKING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.INTERRUPTED, "user interrupted")
        await sm.transition(ConversationState.LISTENING)
        assert sm.state == ConversationState.LISTENING
```

### test_turn_predictor.py

```python
import pytest
from mumble_voice_bot.handlers import TurnPredictor

class TestTurnPredictor:
    @pytest.fixture
    def predictor(self):
        return TurnPredictor(base_delay=0.3, max_delay=1.5)
    
    def test_question_high_confidence(self, predictor):
        conf = predictor.predict_turn_complete("What time is it?", silence_duration=0.0)
        assert conf > 0.8
    
    def test_continuation_word_low_confidence(self, predictor):
        conf = predictor.predict_turn_complete("I want to tell you about and", silence_duration=0.0)
        assert conf < 0.5
    
    def test_silence_increases_confidence(self, predictor):
        conf_no_silence = predictor.predict_turn_complete("Hello there", silence_duration=0.0)
        conf_with_silence = predictor.predict_turn_complete("Hello there", silence_duration=1.5)
        assert conf_with_silence > conf_no_silence
    
    def test_response_delay_for_question(self, predictor):
        delay = predictor.get_response_delay("What?")
        assert delay == predictor.base_delay
    
    def test_response_delay_for_continuation(self, predictor):
        delay = predictor.get_response_delay("I think because")
        assert delay == predictor.max_delay
```

### test_context_preservation.py

```python
import pytest
from mumble_voice_bot.handlers import MumbleVoiceBot

class TestContextPreservation:
    @pytest.mark.asyncio
    async def test_soul_switch_preserves_context(self, bot_with_context):
        # Add some conversation
        bot_with_context._context_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken..."},
        ]
        
        original_user_messages = [
            m for m in bot_with_context._context_messages
            if m["role"] == "user"
        ]
        
        await bot_with_context._switch_soul("zapp", preserve_context=True)
        
        # User messages should be preserved
        new_user_messages = [
            m for m in bot_with_context._context_messages
            if m["role"] == "user"
        ]
        
        assert len(new_user_messages) == len(original_user_messages)
```

## Integration Tests

### test_echo_integration.py

```python
import pytest
import asyncio

class TestEchoFilterIntegration:
    @pytest.mark.asyncio
    async def test_bot_ignores_own_speech(self, running_bot, mock_mumble):
        """Bot shouldn't respond to transcriptions of its own output."""
        # Bot says something
        await running_bot._speak_response("Thank you for your question")
        
        # Simulate STT picking up fragment
        await asyncio.sleep(0.1)
        await running_bot._process_speech("you")
        
        # Should not trigger new response
        assert running_bot._llm_call_count == 1  # Only the original
    
    @pytest.mark.asyncio
    async def test_bot_responds_to_new_speech(self, running_bot, mock_mumble):
        """Bot should respond to genuinely new user speech."""
        await running_bot._speak_response("Hello")
        await asyncio.sleep(0.1)
        
        # User says something completely different
        await running_bot._process_speech("What is the weather like?")
        
        # Should trigger response
        assert running_bot._llm_call_count == 2
```

---

# Implementation Checklist

## Phase 1: Echo Filtering ✅
- [x] Create `EchoFilter` class in `speech_filter.py`
- [x] Add `echo_filter_decay` to `PipelineBotConfig` in `config.py`
- [x] Integrate echo filter in `_process_speech()`
- [x] Call `echo_filter.add_output()` in `_speak_sync()`
- [x] Add unit tests in `tests/test_speech_filter.py`
- [ ] Add integration tests
- [x] Update config.example.yaml with echo filter options

## Phase 2: Utterance Filtering ✅
- [x] Create `UtteranceClassifier` class
- [x] Add configurable thresholds to `PipelineBotConfig`
- [x] Integrate in `_process_speech()` after echo check
- [x] Add unit tests in `tests/test_speech_filter.py`
- [x] Document filler words list (in class docstring)

## Phase 3: State Machine ✅
- [x] Create `ConversationState` enum
- [x] Create `ConversationStateMachine` class
- [ ] Replace `_is_bot_speaking` flag (still uses `_speaking` Event)
- [x] Add state transition logging
- [x] Integrate state checks in speech processing
- [x] Add unit tests in `tests/test_conversation_state.py`
- [ ] Add state to debug/metrics output

## Phase 4: Context Preservation ❌
- [ ] Modify `_switch_soul()` to accept `preserve_context` parameter
- [ ] Add `preserve_context_on_switch` config option
- [ ] Filter system prompts when preserving
- [ ] Add logging for preserved message count
- [ ] Add unit tests in `tests/test_context_preservation.py`

## Phase 5: Continuous ASR (Future) - Partial
- [x] Create `ContinuousASRBuffer` class (in conversation_state.py)
- [ ] Keep STT running during TTS
- [ ] Detect interruptions from buffered transcripts
- [ ] Integrate with state machine (SPEAKING → INTERRUPTED)
- [x] Add tests for interruption detection (basic tests added)

## Phase 6: Turn Prediction ✅
- [x] Create `TurnPredictor` class
- [x] Add configurable delays to config
- [x] Integrate with response timing
- [x] Add unit tests in `tests/test_speech_filter.py`

## Documentation
- [ ] Update IMPLEMENTATION.md with new components
- [ ] Add architecture diagram for state machine
- [x] Document configuration options (in config.example.yaml)
- [ ] Add troubleshooting guide for echo issues

## Performance
- [ ] Benchmark echo filter impact
- [ ] Profile state machine overhead
- [ ] Test with various STT backends
- [ ] Measure latency impact of filtering

---

# Priority Order

1. **Echo Filtering** (Phase 1) - ✅ DONE
2. **Utterance Filtering** (Phase 2) - ✅ DONE
3. **State Machine** (Phase 3) - ✅ DONE (mostly)
4. **Context Preservation** (Phase 4) - ❌ NOT STARTED
5. **Turn Prediction** (Phase 6) - ✅ DONE
6. **Continuous ASR** (Phase 5) - Partial (class created, not integrated)


