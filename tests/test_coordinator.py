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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

        services = SharedBotServices()

        assert services.tts is None
        assert services.stt is None
        assert services.llm is None
        assert services.device == "cuda"

    def test_load_voice_caches(self, tmp_path):
        """Test that load_voice caches voice prompts."""
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

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
                                voice_prompt={"embedding": torch.randn(1, 256)},
                            )

                            assert bot.tts is mock_tts
                            assert bot._owns_tts is False

    def test_bot_accepts_shared_llm(self):
        """Test that bot accepts shared LLM."""
        from mumble_tts_bot import MumbleVoiceBot

        mock_llm = MagicMock()

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
        from mumble_tts_bot import SharedBotServices
        import time

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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

        services = SharedBotServices()
        # Default max is 50, add 60 events
        for i in range(60):
            services.log_event("user_message", "sam", f"message {i}")

        journal = services.get_journal_for_llm()
        assert len(journal) <= 50

    def test_get_recent_messages_for_llm(self):
        """Test formatting messages for LLM context."""
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

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
        from mumble_tts_bot import SharedBotServices

        services = SharedBotServices()
        for i in range(30):
            services.log_event("user_message", "sam", f"message {i}")

        messages = services.get_recent_messages_for_llm(max_messages=10)
        assert len(messages) == 10
        # Should be the most recent 10
        assert messages[-1]["content"] == "sam: message 29"

    def test_journal_seconds_ago(self):
        """Test seconds_ago is calculated correctly."""
        from mumble_tts_bot import SharedBotServices
        import time

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
        from mumble_tts_bot import SharedBotServices

        services = SharedBotServices()
        
        assert services.get_journal_for_llm() == []
        assert services.get_recent_messages_for_llm() == []

    def test_journal_thread_safety(self):
        """Test journal is thread-safe."""
        from mumble_tts_bot import SharedBotServices
        import threading

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
