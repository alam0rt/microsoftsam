"""Tests for speech filtering components."""

import time

from mumble_voice_bot.speech_filter import EchoFilter, TurnPredictor, UtteranceClassifier


class TestEchoFilter:
    """Tests for EchoFilter class."""

    def test_init_defaults(self):
        """Test default initialization."""
        ef = EchoFilter()
        assert ef.decay_time == 3.0
        assert ef.recent_outputs == []

    def test_detects_exact_echo(self):
        """Test that exact substrings are detected as echoes."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Thank you for asking about that")
        assert ef.is_echo("you") is True
        assert ef.is_echo("thank") is True
        assert ef.is_echo("Thank you") is True
        assert ef.is_echo("asking") is True

    def test_detects_word_subset(self):
        """Test that word subsets are detected as echoes."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Hello there my friend")
        assert ef.is_echo("hello") is True
        assert ef.is_echo("there") is True
        assert ef.is_echo("hello there") is True

    def test_ignores_non_echo(self):
        """Test that unrelated text is not detected as echo."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Hello there")
        assert ef.is_echo("goodbye") is False
        assert ef.is_echo("what time is it") is False
        assert ef.is_echo("something completely different") is False

    def test_decay_removes_old_outputs(self):
        """Test that old outputs are removed after decay time."""
        ef = EchoFilter(decay_time=0.1)
        ef.add_output("test phrase")
        time.sleep(0.2)
        assert ef.is_echo("test") is False

    def test_multiple_outputs(self):
        """Test tracking multiple recent outputs."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("First response")
        ef.add_output("Second response")
        assert ef.is_echo("first") is True
        assert ef.is_echo("second") is True
        assert ef.is_echo("third") is False

    def test_empty_text(self):
        """Test handling of empty text."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Some text")
        assert ef.is_echo("") is False
        assert ef.is_echo("   ") is False

    def test_add_empty_output(self):
        """Test that empty outputs are not added."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("")
        ef.add_output("   ")
        assert len(ef.recent_outputs) == 0

    def test_clear(self):
        """Test clearing all outputs."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("test")
        ef.clear()
        assert ef.is_echo("test") is False

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("Hello World")
        assert ef.is_echo("hello") is True
        assert ef.is_echo("HELLO") is True
        assert ef.is_echo("HeLLo WoRLd") is True

    def test_word_overlap_detection(self):
        """Test detection of significant word overlap."""
        ef = EchoFilter(decay_time=3.0)
        ef.add_output("The quick brown fox jumps over the lazy dog")
        # 80% of transcript words in output should trigger
        assert ef.is_echo("quick fox") is True
        assert ef.is_echo("brown fox") is True


class TestUtteranceClassifier:
    """Tests for UtteranceClassifier class."""

    def test_rejects_short_utterances(self):
        """Test rejection of very short utterances."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("ok") is False
        assert uc.is_meaningful("um") is False
        assert uc.is_meaningful("") is False
        assert uc.is_meaningful("a") is False

    def test_rejects_filler_words(self):
        """Test rejection of utterances with only filler words."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("um hmm") is False
        assert uc.is_meaningful("okay yeah") is False
        assert uc.is_meaningful("uh huh") is False
        assert uc.is_meaningful("like yeah") is False

    def test_accepts_questions(self):
        """Test acceptance of question words."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("what?") is True
        assert uc.is_meaningful("why") is True
        assert uc.is_meaningful("how") is True
        assert uc.is_meaningful("when") is True
        assert uc.is_meaningful("where") is True

    def test_accepts_meaningful_speech(self):
        """Test acceptance of meaningful utterances."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("tell me a joke") is True
        assert uc.is_meaningful("what is the weather") is True
        assert uc.is_meaningful("hello there") is True
        assert uc.is_meaningful("can you help me") is True

    def test_accepts_greetings(self):
        """Test acceptance of single-word greetings."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("hello") is True
        assert uc.is_meaningful("hi") is True
        assert uc.is_meaningful("hey") is True
        assert uc.is_meaningful("bye") is True

    def test_accepts_commands(self):
        """Test acceptance of single-word commands."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("stop") is True
        assert uc.is_meaningful("help") is True
        assert uc.is_meaningful("start") is True

    def test_rejects_single_non_meaningful_word(self):
        """Test rejection of single non-meaningful words."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("dog") is False
        assert uc.is_meaningful("the") is False
        assert uc.is_meaningful("stuff") is False

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        uc = UtteranceClassifier(min_words=3, min_chars=10)
        assert uc.is_meaningful("hi there") is False  # Only 2 words
        assert uc.is_meaningful("hello world friend") is True  # 3 words

    def test_empty_input(self):
        """Test handling of empty input."""
        uc = UtteranceClassifier()
        assert uc.is_meaningful("") is False
        assert uc.is_meaningful(None) is False


class TestTurnPredictor:
    """Tests for TurnPredictor class."""

    def test_init_defaults(self):
        """Test default initialization."""
        tp = TurnPredictor()
        assert tp.base_delay == 0.3
        assert tp.max_delay == 1.5

    def test_question_high_confidence(self):
        """Test that questions have high completion confidence."""
        tp = TurnPredictor()
        conf = tp.predict_turn_complete("What time is it?", silence_duration=0.0)
        assert conf > 0.8

    def test_continuation_word_low_confidence(self):
        """Test that continuation words lower confidence."""
        tp = TurnPredictor()
        conf = tp.predict_turn_complete("I want to tell you about and", silence_duration=0.0)
        assert conf < 0.5

    def test_silence_increases_confidence(self):
        """Test that silence increases completion confidence."""
        tp = TurnPredictor()
        conf_no_silence = tp.predict_turn_complete("Hello there", silence_duration=0.0)
        conf_with_silence = tp.predict_turn_complete("Hello there", silence_duration=1.5)
        assert conf_with_silence > conf_no_silence

    def test_response_delay_for_question(self):
        """Test response delay for questions."""
        tp = TurnPredictor(base_delay=0.3, max_delay=1.5)
        delay = tp.get_response_delay("What?")
        assert delay == 0.3

    def test_response_delay_for_continuation(self):
        """Test response delay for continuation words."""
        tp = TurnPredictor(base_delay=0.3, max_delay=1.5)
        delay = tp.get_response_delay("I think because")
        assert delay == 1.5

    def test_response_delay_normal(self):
        """Test normal response delay."""
        tp = TurnPredictor(base_delay=0.3, max_delay=1.5)
        delay = tp.get_response_delay("Hello there.")
        assert 0.3 < delay < 1.5

    def test_should_respond_with_question(self):
        """Test should_respond returns True for questions."""
        tp = TurnPredictor()
        assert tp.should_respond("What is that?", silence_duration=0.5, threshold=0.7) is True

    def test_should_respond_with_silence(self):
        """Test should_respond returns True with enough silence."""
        tp = TurnPredictor()
        assert tp.should_respond("Hello there", silence_duration=2.0, threshold=0.7) is True

    def test_should_not_respond_continuation(self):
        """Test should_respond returns False for continuations."""
        tp = TurnPredictor()
        assert tp.should_respond("I was thinking and", silence_duration=0.1, threshold=0.7) is False

    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        tp = TurnPredictor()
        assert tp.predict_turn_complete("", silence_duration=1.0) == 0.0
        assert tp.get_response_delay("") == tp.max_delay
        assert tp.should_respond("", silence_duration=1.0) is False

    def test_punctuation_effects(self):
        """Test different punctuation effects on confidence."""
        tp = TurnPredictor()
        conf_question = tp.predict_turn_complete("Really?", silence_duration=0.0)
        conf_exclaim = tp.predict_turn_complete("Really!", silence_duration=0.0)
        conf_period = tp.predict_turn_complete("Really.", silence_duration=0.0)
        conf_none = tp.predict_turn_complete("Really", silence_duration=0.0)

        # Questions should have highest confidence
        assert conf_question > conf_exclaim
        assert conf_exclaim > conf_period
        assert conf_period > conf_none
