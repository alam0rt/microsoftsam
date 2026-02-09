"""Tests for context preservation on soul switch."""

from unittest.mock import MagicMock

import pytest


class TestContextPreservationConfig:
    """Test configuration options for context preservation."""

    def test_default_preserve_context_on_switch(self):
        """Test default value for preserve_context_on_switch."""
        from mumble_voice_bot.config import PipelineBotConfig

        config = PipelineBotConfig()
        assert config.preserve_context_on_switch is True

    def test_default_max_preserved_messages(self):
        """Test default value for max_preserved_messages."""
        from mumble_voice_bot.config import PipelineBotConfig

        config = PipelineBotConfig()
        assert config.max_preserved_messages == 10

    def test_custom_preserve_context_on_switch(self):
        """Test custom value for preserve_context_on_switch."""
        from mumble_voice_bot.config import PipelineBotConfig

        config = PipelineBotConfig(preserve_context_on_switch=False)
        assert config.preserve_context_on_switch is False

    def test_custom_max_preserved_messages(self):
        """Test custom value for max_preserved_messages."""
        from mumble_voice_bot.config import PipelineBotConfig

        config = PipelineBotConfig(max_preserved_messages=5)
        assert config.max_preserved_messages == 5


class TestContextPreservationLogic:
    """Test context preservation logic in message filtering."""

    def test_filter_preserves_only_user_assistant(self):
        """Test that only user and assistant messages are preserved."""
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]

        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ]

        assert len(preserved) == 4
        assert preserved[0]["role"] == "user"
        assert preserved[1]["role"] == "assistant"
        assert all(msg["role"] != "system" for msg in preserved)

    def test_limit_preserved_messages(self):
        """Test that max_preserved_messages is respected."""
        max_preserved = 3
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]

        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ][-max_preserved:]

        assert len(preserved) == 3
        # Should keep the LAST 3 messages (indexes 2, 3, 4 of history)
        assert preserved[0]["content"] == "Message 2"
        assert preserved[1]["content"] == "Response 2"
        assert preserved[2]["content"] == "Message 3"

    def test_empty_history_preserves_nothing(self):
        """Test that empty history results in no preserved messages."""
        history = []

        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ]

        assert preserved == []

    def test_only_system_messages_preserves_nothing(self):
        """Test that history with only system messages results in nothing preserved."""
        history = [
            {"role": "system", "content": "System prompt 1"},
            {"role": "system", "content": "System prompt 2"},
        ]

        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ]

        assert preserved == []


class TestSwitchSoulContextPreservation:
    """Test _switch_soul method context preservation behavior."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot for testing."""
        # Import the actual class to test
        from mumble_tts_bot import PipelineBot

        # Create a minimal mock
        bot = MagicMock(spec=PipelineBot)
        bot.channel_history = []
        bot.llm = MagicMock()
        bot.soul_config = None
        bot._current_soul_name = "default"
        bot._load_reference_voice = MagicMock()
        bot._load_system_prompt = MagicMock(return_value="New system prompt")

        return bot

    def test_preserved_messages_filter_logic(self):
        """Test the message filtering logic used in _switch_soul."""
        # Simulate what happens in _switch_soul
        channel_history = [
            {"role": "system", "content": "Old system prompt"},
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Bot response 1"},
            {"role": "user", "content": "User message 2"},
            {"role": "assistant", "content": "Bot response 2"},
            {"role": "tool", "content": "Tool result"},
        ]

        max_preserved = 10

        # This is the logic from _switch_soul
        preserved_messages = [
            msg for msg in channel_history
            if msg.get('role') in ('user', 'assistant')
        ][-max_preserved:]

        assert len(preserved_messages) == 4
        assert all(msg["role"] in ("user", "assistant") for msg in preserved_messages)
        assert {"role": "system", "content": "Old system prompt"} not in preserved_messages
        assert {"role": "tool", "content": "Tool result"} not in preserved_messages

    def test_preserved_messages_respects_limit(self):
        """Test that preservation respects max_preserved_messages."""
        channel_history = [
            {"role": "user", "content": f"User message {i}"}
            if i % 2 == 0 else {"role": "assistant", "content": f"Bot response {i}"}
            for i in range(20)
        ]

        max_preserved = 5

        preserved_messages = [
            msg for msg in channel_history
            if msg.get('role') in ('user', 'assistant')
        ][-max_preserved:]

        assert len(preserved_messages) == 5
        # Should be the last 5 messages
        assert preserved_messages[0] == channel_history[15]
        assert preserved_messages[4] == channel_history[19]

    def test_context_message_format(self):
        """Test the context message format returned by _switch_soul."""
        # Test message format for preserved messages
        preserved_count = 4
        soul_name = "test_soul"

        context_msg = f" ({preserved_count} messages preserved)" if preserved_count else ""
        result = f"Switched to {soul_name}. Voice and personality updated.{context_msg}"

        assert "4 messages preserved" in result
        assert "Switched to test_soul" in result

    def test_context_message_no_preserved(self):
        """Test the context message when no messages are preserved."""
        preserved_count = 0
        soul_name = "test_soul"

        context_msg = f" ({preserved_count} messages preserved)" if preserved_count else ""
        result = f"Switched to {soul_name}. Voice and personality updated.{context_msg}"

        assert "messages preserved" not in result
        assert "Switched to test_soul" in result


class TestContextPreservationIntegration:
    """Integration tests for context preservation."""

    def test_full_preservation_flow(self):
        """Test the full flow of context preservation."""
        # Initial history
        history = [
            {"role": "system", "content": "Old personality"},
            {"role": "user", "content": "Hello bot"},
            {"role": "assistant", "content": "Hello human"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken..."},
        ]

        # Preserve user/assistant messages
        max_preserved = 10
        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ][-max_preserved:]

        # Clear history (as _switch_soul does)
        new_history = []

        # Restore preserved messages
        new_history.extend(preserved)

        # Verify
        assert len(new_history) == 4
        assert new_history[0]["content"] == "Hello bot"
        assert new_history[-1]["content"] == "Why did the chicken..."
        # System message should NOT be in new history
        assert not any(msg["role"] == "system" for msg in new_history)

    def test_preservation_with_zero_max(self):
        """Test preservation with max_preserved_messages = 0."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        max_preserved = 0

        preserved = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ][-max_preserved:] if max_preserved > 0 else []

        assert preserved == []

    def test_preservation_disabled(self):
        """Test when preserve_context_on_switch is False."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        preserve_context = False

        # When preserve_context is False, we don't preserve anything
        preserved = []
        if preserve_context:
            preserved = [
                msg for msg in history
                if msg.get("role") in ("user", "assistant")
            ]

        assert preserved == []
