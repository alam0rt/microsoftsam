"""Tests for shared bot services and multi-persona bot initialization.

Tests cover:
- SharedBotServices initialization
- Voice prompt loading and caching
- Multi-persona bot creation
- MumbleVoiceBot with shared services
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# --- SharedBotServices Tests ---


class TestSharedBotServices:
    """Tests for SharedBotServices container."""

    def test_init_with_services(self):
        """Test initialization with provided services."""
        from mumble_voice_bot.coordination import SharedBotServices

        mock_tts = MagicMock()
        mock_stt = MagicMock()
        mock_llm = MagicMock()

        services = SharedBotServices(
            tts=mock_tts,
            stt=mock_stt,
            llm=mock_llm,
            device="cuda",
        )

        assert services.tts is mock_tts
        assert services.stt is mock_stt
        assert services.llm is mock_llm
        assert services.device == "cuda"
        assert services.voice_prompts == {}

    def test_init_empty(self):
        """Test initialization with no services."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()

        assert services.tts is None
        assert services.stt is None
        assert services.llm is None
        assert services.device == "cuda"

    def test_load_voice_caches(self, tmp_path):
        """Test that load_voice caches voice prompts."""
        from mumble_voice_bot.coordination import SharedBotServices

        # Create a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        voices_dir = tmp_path / "voices"

        mock_tts = MagicMock()
        mock_tts.encode_prompt.return_value = {
            "embedding": torch.randn(1, 256),
        }

        services = SharedBotServices(tts=mock_tts, device="cpu")

        # First load
        voice1 = services.load_voice(
            name="test_voice",
            audio_path=str(audio_file),
            voices_dir=str(voices_dir),
        )

        assert voice1 is not None
        assert "test_voice" in services.voice_prompts

        # Second load should use cache
        voice2 = services.load_voice(
            name="test_voice",
            audio_path=str(audio_file),
            voices_dir=str(voices_dir),
        )

        assert voice2 is voice1  # Same object
        mock_tts.encode_prompt.assert_called_once()  # Only called once

    def test_load_voice_creates_cache_file(self, tmp_path):
        """Test that voice loading creates a cache .pt file."""
        from mumble_voice_bot.coordination import SharedBotServices

        audio_file = tmp_path / "myvoice.wav"
        audio_file.write_bytes(b"fake audio data")

        voices_dir = tmp_path / "voices"

        mock_tts = MagicMock()
        mock_tts.encode_prompt.return_value = {
            "embedding": torch.randn(1, 256),
        }

        services = SharedBotServices(tts=mock_tts, device="cpu")
        services.load_voice(
            name="test",
            audio_path=str(audio_file),
            voices_dir=str(voices_dir),
        )

        # Check cache file was created
        cache_file = voices_dir / "myvoice.pt"
        assert cache_file.exists()

    def test_load_voice_uses_existing_cache(self, tmp_path):
        """Test that voice loading uses existing cache file."""
        from mumble_voice_bot.coordination import SharedBotServices

        # Create cache file directly
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        cache_file = voices_dir / "cached.pt"
        cached_voice = {"embedding": torch.randn(1, 256)}
        torch.save(cached_voice, cache_file)

        # Create audio file (won't be used)
        audio_file = tmp_path / "cached.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_tts = MagicMock()
        services = SharedBotServices(tts=mock_tts, device="cpu")

        voice = services.load_voice(
            name="test",
            audio_path=str(audio_file),
            voices_dir=str(voices_dir),
        )

        # TTS encode should NOT be called - cache was used
        mock_tts.encode_prompt.assert_not_called()
        assert voice is not None

    def test_load_voice_no_tts_raises(self):
        """Test that loading voice without TTS raises error."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices(tts=None, device="cpu")

        with pytest.raises(RuntimeError, match="TTS not initialized"):
            services.load_voice(
                name="test",
                audio_path="/nonexistent.wav",
            )


# --- MumbleVoiceBot Shared Services Tests ---


class TestMumbleVoiceBotSharedServices:
    """Tests for MumbleVoiceBot with shared services."""

    def test_bot_accepts_shared_tts(self):
        """Test that bot accepts shared TTS."""
        # We can't fully test without a real Mumble server,
        # but we can verify the parameter is accepted
        from mumble_tts_bot import MumbleVoiceBot

        mock_tts = MagicMock()
        mock_tts.generate_speech_streaming = MagicMock(return_value=iter([]))
        mock_stt = MagicMock()

        # This should not raise
        with patch('mumble_tts_bot.pymumble.Mumble'):
            with patch.object(MumbleVoiceBot, '_load_reference_voice'):
                with patch.object(MumbleVoiceBot, '_init_speech_filters'):
                    with patch.object(MumbleVoiceBot, '_init_tools'):
                        with patch.object(MumbleVoiceBot, '_init_event_system'):
                            bot = MumbleVoiceBot(
                                host="localhost",
                                user="TestBot",
                                shared_tts=mock_tts,
                                shared_stt=mock_stt,
                                voice_prompt={"embedding": torch.randn(1, 256)},
                            )

                            assert bot.tts is mock_tts
                            assert bot._owns_tts is False

    def test_bot_accepts_shared_llm(self):
        """Test that bot accepts shared LLM."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_llm = MagicMock()
        mock_stt = MagicMock()

        with patch('mumble_tts_bot.pymumble.Mumble'):
            with patch.object(MumbleVoiceBot, '_load_reference_voice'):
                with patch.object(MumbleVoiceBot, '_init_speech_filters'):
                    with patch.object(MumbleVoiceBot, '_init_tools'):
                        with patch.object(MumbleVoiceBot, '_init_event_system'):
                            with patch('mumble_tts_bot.StreamingLuxTTS'):
                                bot = MumbleVoiceBot(
                                    host="localhost",
                                    user="TestBot",
                                    shared_llm=mock_llm,
                                    shared_stt=mock_stt,
                                )

                                assert bot.llm is mock_llm
                                assert bot._owns_llm is False


# --- Multi-Persona Config Tests ---


class TestMultiPersonaConfigIntegration:
    """Tests for multi-persona configuration loading."""

    def test_is_multi_persona_config_true(self, tmp_path):
        """Test detection of multi-persona config."""
        from mumble_voice_bot.multi_persona_config import is_multi_persona_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
personas:
  - name: bot1
    display_name: Bot One
  - name: bot2
    display_name: Bot Two
mumble:
  host: localhost
""")

        assert is_multi_persona_config(config_file) is True

    def test_is_multi_persona_config_false(self, tmp_path):
        """Test detection of single-persona config."""
        from mumble_voice_bot.multi_persona_config import is_multi_persona_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mumble:
  host: localhost
  user: SingleBot
llm:
  model: llama3.2:3b
""")

        assert is_multi_persona_config(config_file) is False

    def test_is_multi_persona_config_nonexistent(self, tmp_path):
        """Test detection of non-existent config."""
        from mumble_voice_bot.multi_persona_config import is_multi_persona_config

        config_file = tmp_path / "nonexistent.yaml"

        assert is_multi_persona_config(config_file) is False


# --- Event Journal Tests ---


class TestEventJournal:
    """Tests for SharedBotServices event journal."""

    def test_log_event_basic(self):
        """Test logging a basic event."""

        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "hello world")

        journal = services.get_journal_for_llm()
        assert len(journal) == 1
        assert journal[0]["event"] == "user_message"
        assert journal[0]["speaker"] == "sam"
        assert journal[0]["content"] == "hello world"
        assert "seconds_ago" in journal[0]
        assert journal[0]["seconds_ago"] >= 0

    def test_log_event_types(self):
        """Test different event types are stored correctly."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "hello")
        services.log_event("bot_message", "Zapp", "Greetings!")
        services.log_event("user_joined", "bob")
        services.log_event("user_left", "alice")
        services.log_event("text_message", "charlie", "hi from chat")

        journal = services.get_journal_for_llm()
        assert len(journal) == 5

        events = [e["event"] for e in journal]
        assert events == ["user_message", "bot_message", "user_joined", "user_left", "text_message"]

    def test_journal_max_entries(self):
        """Test journal respects max entries limit."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        # Default max is 50, add 60 events
        for i in range(60):
            services.log_event("user_message", "sam", f"message {i}")

        journal = services.get_journal_for_llm()
        assert len(journal) <= 50

    def test_get_recent_messages_for_llm(self):
        """Test formatting messages for LLM context."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "hello")
        services.log_event("bot_message", "Zapp", "Greetings, puny human!")
        services.log_event("user_message", "sam", "how are you")
        services.log_event("bot_message", "Zapp", "I am magnificent!")

        messages = services.get_recent_messages_for_llm()

        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "sam: hello"}
        assert messages[1] == {"role": "assistant", "content": "Greetings, puny human!"}
        assert messages[2] == {"role": "user", "content": "sam: how are you"}
        assert messages[3] == {"role": "assistant", "content": "I am magnificent!"}

    def test_get_recent_messages_includes_text_messages(self):
        """Test text chat messages are included in LLM context."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "voice hello")
        services.log_event("text_message", "bob", "text hello")
        services.log_event("bot_message", "Zapp", "Hello both!")

        messages = services.get_recent_messages_for_llm()

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "sam: voice hello"}
        assert messages[1] == {"role": "user", "content": "bob (text): text hello"}
        assert messages[2] == {"role": "assistant", "content": "Hello both!"}

    def test_get_recent_messages_excludes_join_leave(self):
        """Test join/leave events are not in message list (but are in full journal)."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_joined", "sam")
        services.log_event("user_message", "sam", "hello")
        services.log_event("user_left", "bob")

        messages = services.get_recent_messages_for_llm()
        assert len(messages) == 1  # Only the user_message

        journal = services.get_journal_for_llm()
        assert len(journal) == 3  # All events in full journal

    def test_get_recent_messages_max_limit(self):
        """Test message limit is respected."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        for i in range(30):
            services.log_event("user_message", "sam", f"message {i}")

        messages = services.get_recent_messages_for_llm(max_messages=10)
        assert len(messages) == 10
        # Should be the most recent 10
        assert messages[-1]["content"] == "sam: message 29"

    def test_journal_seconds_ago(self):
        """Test seconds_ago is calculated correctly."""
        import time

        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "first")
        time.sleep(0.1)  # 100ms
        services.log_event("user_message", "sam", "second")

        journal = services.get_journal_for_llm()
        assert len(journal) == 2
        # Second message should have smaller seconds_ago than first
        assert journal[1]["seconds_ago"] <= journal[0]["seconds_ago"]

    def test_journal_empty(self):
        """Test empty journal returns empty lists."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()

        assert services.get_journal_for_llm() == []
        assert services.get_recent_messages_for_llm() == []

    def test_journal_thread_safety(self):
        """Test journal is thread-safe."""
        import threading

        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        errors = []

        def writer():
            try:
                for i in range(100):
                    services.log_event("user_message", "writer", f"msg {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    services.get_journal_for_llm()
                    services.get_recent_messages_for_llm()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestBotUtteranceHandling:
    """Tests for bot-to-bot utterance handling and talks_to_bots feature."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot with minimal required attributes."""
        import threading
        from unittest.mock import MagicMock

        from mumble_voice_bot.coordination import SharedBotServices

        bot = MagicMock()
        bot.user = "TestBot"
        bot.logger = MagicMock()
        bot._speaking = threading.Event()
        bot._shared_services = SharedBotServices()
        bot.pending_text = {}
        bot.pending_text_time = {}
        bot.soul_config = None
        return bot

    @pytest.fixture
    def soul_config_talks_to_bots(self):
        """Create a SoulConfig with talks_to_bots enabled."""
        from mumble_voice_bot.config import SoulConfig
        return SoulConfig(name="TalkativeBot", talks_to_bots=True)

    @pytest.fixture
    def soul_config_no_talk(self):
        """Create a SoulConfig with talks_to_bots disabled (default)."""
        from mumble_voice_bot.config import SoulConfig
        return SoulConfig(name="QuietBot", talks_to_bots=False)

    def test_ignores_own_utterances(self, mock_bot):
        """Test that bot ignores its own broadcast utterances."""
        from mumble_tts_bot import MumbleVoiceBot

        # Call the method directly on a mock
        mock_bot.user = "Zapp"
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello there friend!")

        # Should not log anything or process (ignores own utterances)
        assert "Zapp" not in str(mock_bot.logger.info.call_args_list)

    def test_ignores_short_utterances(self, mock_bot):
        """Test that bot ignores very short utterances (fillers)."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.user = "Raf"
        # Short utterances (< 3 words) should be ignored as likely fillers
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hmm...")
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello!")
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Let me...")

        # Should NOT log [BOT-HEARD] with info level (only debug)
        call_str = str(mock_bot.logger.info.call_args_list)
        assert "[BOT-HEARD]" not in call_str or "ignoring" not in call_str
        # Should log with debug level instead
        assert mock_bot.logger.debug.called

    def test_logs_heard_utterance(self, mock_bot):
        """Test that hearing another bot is logged."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.user = "Raf"
        # Note: Utterance must be 3+ words to pass the short utterance filter
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Greetings my friend!")

        mock_bot.logger.info.assert_called()
        call_str = str(mock_bot.logger.info.call_args_list)
        assert "BOT-HEARD" in call_str
        assert "Zapp" in call_str

    def test_no_response_without_talks_to_bots(self, mock_bot, soul_config_no_talk):
        """Test that bot does not respond when talks_to_bots is False."""

        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.soul_config = soul_config_no_talk
        mock_bot.user = "Raf"
        # Note: Utterance must be 3+ words to pass the short utterance filter
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello there Raf!")

        # Should log that we heard it
        mock_bot.logger.info.assert_called()

        # Should NOT add to pending_text (no response queued)
        assert len(mock_bot.pending_text) == 0

    def test_no_response_without_soul_config(self, mock_bot):
        """Test that bot does not respond when soul_config is None."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.soul_config = None
        mock_bot.user = "Raf"
        # Note: Utterance must be 3+ words to pass the short utterance filter
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello there friend!")

        # Should NOT add to pending_text
        assert len(mock_bot.pending_text) == 0

    def test_queues_response_with_talks_to_bots(self, mock_bot, soul_config_talks_to_bots):
        """Test that bot queues response when talks_to_bots is True."""

        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.soul_config = soul_config_talks_to_bots
        mock_bot.user = "Raf"
        mock_bot._speaking.clear()  # Not speaking
        # Note: Utterance must be 3+ words to pass the short utterance filter
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello there Raf!")

        # Should add to pending_text (response will be attempted)
        # Note: The actual response happens in a background thread after a delay
        # so we can't immediately check pending_text. Instead we verify the
        # method doesn't crash and logs appropriately.
        # The pending_text is set in the delayed_respond thread.
        import time as time_mod
        time_mod.sleep(0.1)  # Give thread time to start
        # The thread is running - just verify it was started
        mock_bot.logger.info.assert_called()

    def test_no_response_while_speaking(self, mock_bot, soul_config_talks_to_bots):
        """Test that bot does not queue response while speaking."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.soul_config = soul_config_talks_to_bots
        mock_bot.user = "Raf"
        mock_bot._speaking.set()  # Currently speaking
        # Note: Utterance must be 3+ words to pass the short utterance filter
        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", "Hello there friend!")

        # Should NOT add to pending_text (we're speaking)
        assert len(mock_bot.pending_text) == 0

    def test_long_text_truncated_in_log(self, mock_bot):
        """Test that long utterances are truncated in log messages."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.user = "Raf"
        # Note: Text must be 3+ words to pass short utterance filter
        # Use multiple words to make it long enough
        long_text = "This is a very long message " * 10  # ~280 characters, 70 words

        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", long_text)

        call_str = str(mock_bot.logger.info.call_args_list)
        assert "..." in call_str  # Should be truncated
        assert "This is a very long message" in call_str  # First part should be there

    def test_short_text_not_truncated_in_log(self, mock_bot):
        """Test that short utterances (3+ words) are not truncated in log."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_bot.user = "Raf"
        # Note: Must be 3+ words to pass filter, but short enough to not truncate
        short_text = "Hello there friend!"

        MumbleVoiceBot._on_bot_utterance(mock_bot, "Zapp", short_text)

        call_str = str(mock_bot.logger.info.call_args_list)
        # Should NOT have truncation ellipsis (text is short enough)
        # But we still check the text appears
        assert "Hello there friend!" in call_str


class TestSoulConfigTalksToBots:
    """Tests for talks_to_bots configuration in SoulConfig."""

    def test_default_is_false(self):
        """Test that talks_to_bots defaults to False."""
        from mumble_voice_bot.config import SoulConfig

        config = SoulConfig()
        assert config.talks_to_bots is False

    def test_can_set_true(self):
        """Test that talks_to_bots can be set to True."""
        from mumble_voice_bot.config import SoulConfig

        config = SoulConfig(talks_to_bots=True)
        assert config.talks_to_bots is True

    def test_load_from_yaml_default(self, tmp_path):
        """Test loading soul without talks_to_bots defaults to False."""
        from mumble_voice_bot.config import load_soul_config

        soul_dir = tmp_path / "souls" / "test_soul"
        soul_dir.mkdir(parents=True)

        soul_yaml = """
name: "Test Soul"
description: "A test soul"
"""
        (soul_dir / "soul.yaml").write_text(soul_yaml)

        config = load_soul_config("test_soul", tmp_path / "souls")
        assert config.talks_to_bots is False

    def test_load_from_yaml_true(self, tmp_path):
        """Test loading soul with talks_to_bots: true."""
        from mumble_voice_bot.config import load_soul_config

        soul_dir = tmp_path / "souls" / "chatty_soul"
        soul_dir.mkdir(parents=True)

        soul_yaml = """
name: "Chatty Soul"
description: "A soul that talks to other bots"
talks_to_bots: true
"""
        (soul_dir / "soul.yaml").write_text(soul_yaml)

        config = load_soul_config("chatty_soul", tmp_path / "souls")
        assert config.talks_to_bots is True

    def test_load_from_yaml_false(self, tmp_path):
        """Test loading soul with talks_to_bots: false."""
        from mumble_voice_bot.config import load_soul_config

        soul_dir = tmp_path / "souls" / "quiet_soul"
        soul_dir.mkdir(parents=True)

        soul_yaml = """
name: "Quiet Soul"
description: "A soul that does not talk to other bots"
talks_to_bots: false
"""
        (soul_dir / "soul.yaml").write_text(soul_yaml)

        config = load_soul_config("quiet_soul", tmp_path / "souls")
        assert config.talks_to_bots is False


class TestBroadcastUtterance:
    """Tests for utterance broadcasting between bots."""

    def test_broadcast_logs_to_journal(self):
        """Test that broadcast_utterance logs to the event journal."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.broadcast_utterance("Zapp", "Victory is mine!")

        journal = services.get_journal_for_llm()
        assert len(journal) == 1
        assert journal[0]["event"] == "bot_message"
        assert journal[0]["speaker"] == "Zapp"
        assert journal[0]["content"] == "Victory is mine!"

    def test_broadcast_notifies_listeners(self):
        """Test that broadcast_utterance notifies registered listeners."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        received = []

        def listener(speaker, text):
            received.append((speaker, text))

        services.register_utterance_listener(listener)
        services.broadcast_utterance("Raf", "Hey man")

        assert len(received) == 1
        assert received[0] == ("Raf", "Hey man")

    def test_broadcast_multiple_listeners(self):
        """Test broadcast notifies multiple listeners."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        received1 = []
        received2 = []

        services.register_utterance_listener(lambda s, t: received1.append((s, t)))
        services.register_utterance_listener(lambda s, t: received2.append((s, t)))
        services.broadcast_utterance("Zapp", "Kif!")

        assert received1 == [("Zapp", "Kif!")]
        assert received2 == [("Zapp", "Kif!")]

    def test_broadcast_listener_error_doesnt_break_others(self):
        """Test that a failing listener doesn't prevent others from receiving."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        received = []

        def bad_listener(speaker, text):
            raise RuntimeError("I'm broken!")

        def good_listener(speaker, text):
            received.append((speaker, text))

        services.register_utterance_listener(bad_listener)
        services.register_utterance_listener(good_listener)

        # Should not raise, and good listener should still receive
        services.broadcast_utterance("Zapp", "For glory!")

        assert received == [("Zapp", "For glory!")]

    def test_bot_messages_in_llm_context(self):
        """Test bot messages appear correctly in LLM context without bot_name."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "Hello bots!")
        services.broadcast_utterance("Raf", "Hey sam!")
        services.broadcast_utterance("Zapp", "Greetings, citizen!")

        # Without bot_name, all bot messages are "assistant"
        messages = services.get_recent_messages_for_llm()

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "sam: Hello bots!"}
        assert messages[1] == {"role": "assistant", "content": "Hey sam!"}
        assert messages[2] == {"role": "assistant", "content": "Greetings, citizen!"}

    def test_bot_messages_multi_bot_context(self):
        """Test that bots see each other's messages as user messages, not assistant."""
        from mumble_voice_bot.coordination import SharedBotServices

        services = SharedBotServices()
        services.log_event("user_message", "sam", "Hello bots!")
        services.broadcast_utterance("Raf", "Hey sam!")
        services.broadcast_utterance("Zapp", "Greetings, citizen!")

        # From Raf's perspective: only Raf's messages are "assistant"
        # Zapp's messages should be "user" with name prefix
        messages_raf = services.get_recent_messages_for_llm(bot_name="Raf")

        assert len(messages_raf) == 3
        assert messages_raf[0] == {"role": "user", "content": "sam: Hello bots!"}
        assert messages_raf[1] == {"role": "assistant", "content": "Hey sam!"}  # Raf's own message
        assert messages_raf[2] == {"role": "user", "content": "Zapp: Greetings, citizen!"}  # Zapp is another user

        # From Zapp's perspective: only Zapp's messages are "assistant"
        messages_zapp = services.get_recent_messages_for_llm(bot_name="Zapp")

        assert len(messages_zapp) == 3
        assert messages_zapp[0] == {"role": "user", "content": "sam: Hello bots!"}
        assert messages_zapp[1] == {"role": "user", "content": "Raf: Hey sam!"}  # Raf is another user
        assert messages_zapp[2] == {"role": "assistant", "content": "Greetings, citizen!"}  # Zapp's own message
