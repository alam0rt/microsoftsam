"""Tests for shared bot services and multi-persona bot initialization.

Tests cover:
- SharedBotServices initialization
- Voice prompt loading and caching
- Multi-persona bot creation
- MumbleVoiceBot with shared services
"""

import os
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
