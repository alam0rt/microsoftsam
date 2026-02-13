"""Tests for the MumbleBot base class.

Tests cover:
- Construction with NullBrain
- Construction with mock Brain
- Utterance building and brain routing
- TTS response pipeline
- Bot utterance handling
- Lifecycle
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mumble_voice_bot.interfaces.brain import BotResponse, NullBrain, Utterance


class TestMumbleBotConstruction:
    """Tests for MumbleBot initialization."""

    def test_init_with_null_brain(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost",
            user="TestBot",
            brain=NullBrain(),
            tts=MagicMock(),
            stt=MagicMock(),
        )
        assert bot.user == "TestBot"
        assert isinstance(bot.brain, NullBrain)
        assert bot.mumble is None  # Not connected yet

    def test_init_defaults_to_null_brain(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost",
            user="TestBot",
            tts=MagicMock(),
            stt=MagicMock(),
        )
        assert isinstance(bot.brain, NullBrain)

    def test_init_with_shared_services(self):
        from mumble_voice_bot.bot import MumbleBot
        from mumble_voice_bot.coordination import SharedBotServices

        shared = SharedBotServices(tts=MagicMock(), stt=MagicMock())
        bot = MumbleBot(
            host="localhost",
            user="TestBot",
            tts=MagicMock(),
            stt=MagicMock(),
            shared_services=shared,
        )
        assert bot._shared_services is shared


class TestMumbleBotVAD:
    """Tests for VAD and audio buffering."""

    def test_get_buffer_duration_empty(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(host="localhost", user="TestBot", tts=MagicMock(), stt=MagicMock())
        assert bot._get_buffer_duration(999) == 0

    def test_get_buffer_duration_with_data(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(host="localhost", user="TestBot", tts=MagicMock(), stt=MagicMock())
        # 1 second at 48kHz, 16-bit = 96000 bytes
        bot.audio_buffers[1] = [b'\x00' * 96000]
        assert abs(bot._get_buffer_duration(1) - 1.0) < 0.01


class TestMumbleBotBrainRouting:
    """Tests for brain routing logic."""

    def test_maybe_respond_no_pending(self):
        from mumble_voice_bot.bot import MumbleBot

        brain = MagicMock()
        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=brain, tts=MagicMock(), stt=MagicMock(),
        )
        # No pending text → brain should not be called
        bot._maybe_respond(user_id=1, user_name="sam", force=True)
        brain.process.assert_not_called()

    def test_maybe_respond_with_pending(self):
        from mumble_voice_bot.bot import MumbleBot

        brain = MagicMock()
        brain.process.return_value = None  # Silent brain

        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=brain, tts=MagicMock(), stt=MagicMock(),
        )
        bot.pending_text[1] = "hello world"
        bot.pending_text_time[1] = 0  # Old enough to trigger

        bot._maybe_respond(user_id=1, user_name="sam", force=True)
        brain.process.assert_called_once()

        # Verify the Utterance
        utterance = brain.process.call_args[0][0]
        assert isinstance(utterance, Utterance)
        assert utterance.text == "hello world"
        assert utterance.user_name == "sam"

    def test_maybe_respond_speaks_response(self):
        from mumble_voice_bot.bot import MumbleBot

        brain = MagicMock()
        brain.process.return_value = BotResponse(text="hi there")

        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=brain, tts=MagicMock(), stt=MagicMock(),
        )
        bot.pending_text[1] = "hello"
        bot.pending_text_time[1] = 0

        with patch.object(bot, '_speak_response') as mock_speak:
            bot._maybe_respond(user_id=1, user_name="sam", force=True)
            mock_speak.assert_called_once()
            response = mock_speak.call_args[0][0]
            assert response.text == "hi there"

    def test_null_brain_stays_silent(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=NullBrain(), tts=MagicMock(), stt=MagicMock(),
        )
        bot.pending_text[1] = "hello world"
        bot.pending_text_time[1] = 0

        with patch.object(bot, '_speak_response') as mock_speak:
            bot._maybe_respond(user_id=1, user_name="sam", force=True)
            mock_speak.assert_not_called()


class TestMumbleBotTTS:
    """Tests for TTS output pipeline."""

    def test_speak_response_queues_text(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            tts=MagicMock(), stt=MagicMock(),
        )
        response = BotResponse(text="hello world")
        bot._speak_response(response)

        assert not bot._tts_queue.empty()
        item = bot._tts_queue.get()
        text, voice_prompt, skip_broadcast = item
        assert text == "hello world"
        assert not skip_broadcast

    def test_speak_response_skip_broadcast_for_fillers(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            tts=MagicMock(), stt=MagicMock(),
        )
        response = BotResponse(text="mmhm", is_filler=True, skip_broadcast=True)
        bot._speak_response(response)

        item = bot._tts_queue.get()
        _, _, skip_broadcast = item
        assert skip_broadcast

    def test_speak_response_empty_after_sanitize_is_skipped(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            tts=MagicMock(), stt=MagicMock(),
        )
        # All emoji text gets sanitized to empty
        response = BotResponse(text="🎉🎊🎈")
        bot._speak_response(response)
        assert bot._tts_queue.empty()


class TestMumbleBotBotUtterance:
    """Tests for bot-to-bot communication."""

    def test_ignores_own_utterances(self):
        from mumble_voice_bot.bot import MumbleBot

        brain = MagicMock()
        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=brain, tts=MagicMock(), stt=MagicMock(),
        )
        bot._on_bot_utterance("TestBot", "hello")
        brain.on_bot_utterance.assert_not_called()

    def test_routes_other_bot_utterances_to_brain(self):
        from mumble_voice_bot.bot import MumbleBot

        brain = MagicMock()
        brain.on_bot_utterance.return_value = None

        bot = MumbleBot(
            host="localhost", user="TestBot",
            brain=brain, tts=MagicMock(), stt=MagicMock(),
        )
        bot._on_bot_utterance("OtherBot", "hello there friend")
        brain.on_bot_utterance.assert_called_once_with("OtherBot", "hello there friend")


class TestMumbleBotLifecycle:
    """Tests for bot lifecycle."""

    def test_shutdown_sets_flags(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            tts=MagicMock(), stt=MagicMock(),
        )
        bot._running = True
        bot.shutdown()
        assert not bot._running
        assert bot._shutdown.is_set()

    def test_speak_public_api(self):
        from mumble_voice_bot.bot import MumbleBot

        bot = MumbleBot(
            host="localhost", user="TestBot",
            tts=MagicMock(), stt=MagicMock(),
        )
        bot.speak("hello")
        assert not bot._tts_queue.empty()
