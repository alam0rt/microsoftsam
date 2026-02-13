"""Tests for the Brain protocol and brain implementations.

Tests cover:
- Brain protocol compliance
- NullBrain (always silent)
- EchoBrain (voice cloning + echo)
- ReactiveBrain (fillers, echo fragments, deflections)
- AdaptiveBrain (brain_power routing)
- LLMBrain (basic construction)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mumble_voice_bot.interfaces.brain import (
    BotResponse,
    Brain,
    NullBrain,
    Utterance,
    VoiceConfig,
)


# =============================================================================
# Utterance / BotResponse data types
# =============================================================================


class TestUtterance:
    """Tests for the Utterance dataclass."""

    def test_basic_construction(self):
        u = Utterance(text="hello", user_id=1, user_name="sam")
        assert u.text == "hello"
        assert u.user_id == 1
        assert u.user_name == "sam"
        assert u.audio_chunks == []
        assert u.duration == 0.0
        assert u.rms == 0.0
        assert not u.is_question
        assert not u.is_directed
        assert not u.is_first_speech

    def test_with_audio(self):
        chunk = b'\x00' * 960
        u = Utterance(
            text="test", user_id=1, user_name="sam",
            audio_chunks=[chunk], duration=1.5, rms=3000.0,
        )
        assert len(u.audio_chunks) == 1
        assert u.duration == 1.5
        assert u.rms == 3000.0

    def test_question_flag(self):
        u = Utterance(text="what time is it?", user_id=1, user_name="sam", is_question=True)
        assert u.is_question


class TestBotResponse:
    """Tests for the BotResponse dataclass."""

    def test_basic(self):
        r = BotResponse(text="hello there")
        assert r.text == "hello there"
        assert r.voice is None
        assert r.speed == 1.0
        assert not r.skip_broadcast
        assert not r.is_filler

    def test_filler(self):
        r = BotResponse(text="mmhm", is_filler=True, skip_broadcast=True)
        assert r.is_filler
        assert r.skip_broadcast

    def test_with_voice(self):
        vc = VoiceConfig(voice_prompt={"test": "tensor"}, speed=1.2)
        r = BotResponse(text="hello", voice=vc)
        assert r.voice.speed == 1.2


# =============================================================================
# NullBrain
# =============================================================================


class TestNullBrain:
    """Tests for NullBrain (transcribe-only mode)."""

    def test_process_returns_none(self):
        brain = NullBrain()
        u = Utterance(text="hello", user_id=1, user_name="sam")
        assert brain.process(u) is None

    def test_on_bot_utterance_returns_none(self):
        brain = NullBrain()
        assert brain.on_bot_utterance("other_bot", "hello") is None

    def test_on_text_message_returns_none(self):
        brain = NullBrain()
        assert brain.on_text_message("sender", "hello") is None

    def test_implements_brain_protocol(self):
        assert isinstance(NullBrain(), Brain)


# =============================================================================
# EchoBrain
# =============================================================================


class TestEchoBrain:
    """Tests for EchoBrain (voice cloning + echo)."""

    def test_returns_none_for_empty_text(self):
        from mumble_voice_bot.brains.echo import EchoBrain
        tts = MagicMock()
        brain = EchoBrain(tts=tts)

        u = Utterance(text="", user_id=1, user_name="sam")
        assert brain.process(u) is None

    def test_returns_none_without_audio(self):
        from mumble_voice_bot.brains.echo import EchoBrain
        tts = MagicMock()
        brain = EchoBrain(tts=tts)

        u = Utterance(text="hello world", user_id=1, user_name="sam")
        assert brain.process(u) is None

    def test_echoes_text_with_cloned_voice(self):
        from mumble_voice_bot.brains.echo import EchoBrain
        tts = MagicMock()
        tts.encode_prompt.return_value = {"tensor": "data"}
        brain = EchoBrain(tts=tts)

        # Create a PCM audio chunk (16-bit, 48kHz, 0.5s)
        samples = np.zeros(24000, dtype=np.int16)
        chunk = samples.tobytes()

        u = Utterance(
            text="hello world", user_id=1, user_name="sam",
            audio_chunks=[chunk],
        )

        with patch('soundfile.write'):
            result = brain.process(u)

        assert result is not None
        assert result.text == "hello world"
        assert result.voice is not None

    def test_ignores_other_bots(self):
        from mumble_voice_bot.brains.echo import EchoBrain
        brain = EchoBrain(tts=MagicMock())
        assert brain.on_bot_utterance("other", "hello") is None

    def test_ignores_text_messages(self):
        from mumble_voice_bot.brains.echo import EchoBrain
        brain = EchoBrain(tts=MagicMock())
        assert brain.on_text_message("sender", "hello") is None


# =============================================================================
# ReactiveBrain
# =============================================================================


class TestReactiveBrain:
    """Tests for ReactiveBrain (no LLM)."""

    def test_sometimes_responds(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain(silence_weight=0.0)  # Never silent
        u = Utterance(text="hello world", user_id=1, user_name="sam")

        # Should always respond when silence_weight is 0
        result = brain.process(u)
        assert result is not None
        assert result.is_filler
        assert result.skip_broadcast

    def test_silence_at_high_weight(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain(silence_weight=1.0)  # Always silent
        u = Utterance(text="hello world", user_id=1, user_name="sam")

        result = brain.process(u)
        assert result is None

    def test_question_gets_special_treatment(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain(silence_weight=0.0)
        u = Utterance(text="what time is it?", user_id=1, user_name="sam", is_question=True)

        # Should respond to questions
        result = brain.process(u)
        assert result is not None

    def test_echo_fragment_extraction(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        fragment = brain._echo_fragment("I went to the store yesterday")
        assert fragment is not None
        assert "yesterday" in fragment or "store" in fragment

    def test_echo_fragment_too_short(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        fragment = brain._echo_fragment("hi")
        assert fragment is None

    def test_stalling_echo(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        stall = brain._stalling_echo("how does this thing work anyway")
        assert stall is not None
        assert "how does" in stall

    def test_get_thinking_stall(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        stall = brain.get_thinking_stall()
        assert isinstance(stall, str)
        assert len(stall) > 0

    def test_ignores_bots(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        assert brain.on_bot_utterance("other", "hello") is None

    def test_ignores_text(self):
        from mumble_voice_bot.brains.reactive import ReactiveBrain

        brain = ReactiveBrain()
        assert brain.on_text_message("sender", "hello") is None


# =============================================================================
# AdaptiveBrain
# =============================================================================


class TestAdaptiveBrain:
    """Tests for AdaptiveBrain (brain_power routing)."""

    def test_always_thinks_at_power_1(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        llm = MagicMock()
        llm.process.return_value = BotResponse(text="thought response")
        reactive = MagicMock()
        reactive.process.return_value = BotResponse(text="reactive response")

        brain = AdaptiveBrain(llm_brain=llm, reactive_brain=reactive, brain_power=1.0)
        u = Utterance(text="hello", user_id=1, user_name="sam")

        result = brain.process(u)
        assert result is not None
        llm.process.assert_called_once()
        reactive.process.assert_not_called()

    def test_never_thinks_at_power_0(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        llm = MagicMock()
        reactive = MagicMock()
        reactive.process.return_value = BotResponse(text="reactive", is_filler=True)

        brain = AdaptiveBrain(llm_brain=llm, reactive_brain=reactive, brain_power=0.0)
        u = Utterance(text="hello", user_id=1, user_name="sam")

        brain.process(u)
        llm.process.assert_not_called()

    def test_force_think_on_direct_question(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        llm = MagicMock()
        llm.process.return_value = BotResponse(text="answer")
        reactive = MagicMock()

        brain = AdaptiveBrain(
            llm_brain=llm, reactive_brain=reactive,
            brain_power=0.1,  # Very low
            bot_name="TestBot",
        )
        u = Utterance(
            text="TestBot what time is it?", user_id=1, user_name="sam",
            is_question=True, is_directed=True,
        )

        result = brain.process(u)
        llm.process.assert_called_once()

    def test_override_brain_power(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        llm = MagicMock()
        reactive = MagicMock()
        reactive.process.return_value = BotResponse(text="reactive", is_filler=True)

        brain = AdaptiveBrain(llm_brain=llm, reactive_brain=reactive, brain_power=1.0)
        assert brain.effective_brain_power == 1.0

        brain.set_override(0.0)
        assert brain.effective_brain_power == 0.0

        u = Utterance(text="hello", user_id=1, user_name="sam")
        brain.process(u)
        llm.process.assert_not_called()

        brain.set_override(None)
        assert brain.effective_brain_power == 1.0

    def test_urgency_scoring(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        brain = AdaptiveBrain(
            llm_brain=MagicMock(), reactive_brain=MagicMock(),
            brain_power=0.5,
        )

        # Low urgency
        u_low = Utterance(text="yeah ok", user_id=1, user_name="sam")
        score_low = brain._score_utterance(u_low)

        # High urgency
        u_high = Utterance(
            text="hey bot", user_id=1, user_name="sam",
            is_directed=True, is_question=True, is_first_speech=True,
            rms=6000.0,
        )
        score_high = brain._score_utterance(u_high)

        assert score_high > score_low
        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0

    def test_text_messages_go_to_llm(self):
        from mumble_voice_bot.brains.adaptive import AdaptiveBrain

        llm = MagicMock()
        llm.on_text_message.return_value = BotResponse(text="response")
        reactive = MagicMock()

        brain = AdaptiveBrain(llm_brain=llm, reactive_brain=reactive, brain_power=0.0)
        result = brain.on_text_message("sender", "hello")
        llm.on_text_message.assert_called_once()


# =============================================================================
# Audio utilities
# =============================================================================


class TestAudioUtilities:
    """Tests for extracted audio utilities."""

    def test_pcm_rms_empty(self):
        from mumble_voice_bot.audio import pcm_rms
        assert pcm_rms(b'') == 0
        assert pcm_rms(b'\x00') == 0

    def test_pcm_rms_silence(self):
        from mumble_voice_bot.audio import pcm_rms
        silence = np.zeros(1000, dtype=np.int16).tobytes()
        assert pcm_rms(silence) == 0

    def test_pcm_rms_signal(self):
        from mumble_voice_bot.audio import pcm_rms
        # A signal with known amplitude
        signal = np.full(1000, 1000, dtype=np.int16).tobytes()
        assert pcm_rms(signal) == 1000

    def test_pcm_bytes_roundtrip(self):
        from mumble_voice_bot.audio import float_to_pcm_bytes, pcm_bytes_to_float
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = float_to_pcm_bytes(original)
        restored = pcm_bytes_to_float(pcm)
        np.testing.assert_allclose(restored, original, atol=1e-4)

    def test_pcm_duration(self):
        from mumble_voice_bot.audio import pcm_duration
        # 1 second at 48kHz, 16-bit = 96000 bytes
        assert pcm_duration(b'\x00' * 96000) == 1.0
        assert pcm_duration(b'\x00' * 48000) == 0.5

    def test_normalize_for_stt(self):
        from mumble_voice_bot.audio import normalize_for_stt
        loud = np.full(1000, 0.9, dtype=np.float32)
        normalized = normalize_for_stt(loud, target_rms=0.1)
        rms = np.sqrt(np.mean(normalized ** 2))
        assert abs(rms - 0.1) < 0.01


# =============================================================================
# Text processing
# =============================================================================


class TestTextProcessing:
    """Tests for extracted text processing utilities."""

    def test_split_into_sentences(self):
        from mumble_voice_bot.text_processing import split_into_sentences
        result = split_into_sentences("Hello world. How are you? I'm fine!")
        assert len(result) == 3

    def test_pad_tts_text_short(self):
        from mumble_voice_bot.text_processing import pad_tts_text
        result = pad_tts_text("hi")
        assert len(result) >= 20
        assert "..." in result

    def test_pad_tts_text_long_enough(self):
        from mumble_voice_bot.text_processing import pad_tts_text
        text = "This is already long enough for TTS."
        result = pad_tts_text(text)
        assert result == text

    def test_pad_tts_text_empty(self):
        from mumble_voice_bot.text_processing import pad_tts_text
        assert pad_tts_text("") == ""
        assert pad_tts_text("   ") == ""

    def test_is_question(self):
        from mumble_voice_bot.text_processing import is_question
        assert is_question("what time is it?")
        assert is_question("How does this work")
        assert is_question("Is it working?")
        assert not is_question("Hello there.")
        assert not is_question("I like pizza.")


# =============================================================================
# Events
# =============================================================================


class TestEventResponder:
    """Tests for EventResponder."""

    def test_no_soul_config(self):
        from mumble_voice_bot.events import EventResponder
        responder = EventResponder()
        assert responder.get_filler("thinking") is None
        assert responder.get_event_response("user_joined") is None

    def test_with_mock_soul(self):
        from mumble_voice_bot.events import EventResponder
        soul = MagicMock()
        soul.events.thinking = ["Hmm...", "Let me think..."]
        soul.events.user_first_speech = ["Hey {user}!", "Welcome {user}!"]
        soul.fallbacks = None

        responder = EventResponder(soul_config=soul)
        filler = responder.get_filler("thinking")
        assert filler in ["Hmm...", "Let me think..."]

        greeting = responder.get_event_response("user_first_speech", user="sam")
        assert "sam" in greeting


class TestChannelActivityTracker:
    """Tests for ChannelActivityTracker."""

    def test_first_time_speaker(self):
        from mumble_voice_bot.events import ChannelActivityTracker
        tracker = ChannelActivityTracker()
        assert tracker.check_first_time_speaker("sam") is True
        assert tracker.check_first_time_speaker("sam") is False
        assert tracker.check_first_time_speaker("other") is True

    def test_long_speech(self):
        from mumble_voice_bot.events import ChannelActivityTracker
        tracker = ChannelActivityTracker(long_speech_threshold=5.0)
        assert not tracker.check_long_speech(1, "sam", 3.0)
        assert tracker.check_long_speech(1, "sam", 3.0)  # Now >5.0 total
        # Counter should have reset
        assert not tracker.check_long_speech(1, "sam", 1.0)
