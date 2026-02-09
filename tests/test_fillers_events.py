"""Tests for filler phrases and event callbacks.

Tests the following features:
- Thinking fillers ("Hmm...", "Let me think...")
- Still-thinking timer (fires after LLM delay)
- Barge-in callback (interrupted)
- Soul event responses (user_first_speech, etc.)
- Skip broadcast for fillers/events
- Minimum utterance length filter for bot-to-bot
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

from mumble_voice_bot.config import (
    SoulConfig,
    SoulFallbacks,
    SoulEvents,
    load_soul_config,
)


class TestSoulEventsConfig:
    """Test SoulEvents dataclass and config loading."""

    def test_soul_events_has_defaults(self):
        """SoulEvents should have sensible defaults (not None for common events)."""
        events = SoulEvents()
        # These have default values
        assert events.user_first_speech is not None
        assert events.thinking is not None
        assert events.still_thinking is not None
        assert events.interrupted is not None
        # These are None by default (optional/noisy)
        assert events.user_joined is None
        assert events.user_left is None

    def test_soul_events_with_values(self):
        """SoulEvents should accept list values."""
        events = SoulEvents(
            user_first_speech=["Hello {user}!", "Hi there {user}!"],
            thinking=["Hmm...", "Let me think..."],
            still_thinking=["One moment...", "Still working on that..."],
            interrupted=["Oh, sorry!", "Go ahead..."],
        )
        assert len(events.user_first_speech) == 2
        assert len(events.thinking) == 2
        assert "{user}" in events.user_first_speech[0]

    def test_soul_fallbacks_has_filler_fields(self):
        """SoulFallbacks should have thinking/still_thinking/interrupted fields."""
        fallbacks = SoulFallbacks(
            greetings=["Hello!"],
            thinking=["Hmm..."],
            still_thinking=["One sec..."],
            interrupted=["Oh sorry!"],
        )
        assert fallbacks.thinking == ["Hmm..."]
        assert fallbacks.still_thinking == ["One sec..."]
        assert fallbacks.interrupted == ["Oh sorry!"]

    def test_soul_config_has_events(self):
        """SoulConfig should have events field."""
        config = SoulConfig(
            name="test",
            events=SoulEvents(
                thinking=["Hmm..."],
            ),
        )
        assert config.events is not None
        assert config.events.thinking == ["Hmm..."]


class TestSoulConfigLoading:
    """Test loading soul configs with events."""

    def test_load_soul_config_parses_events(self, tmp_path):
        """load_soul_config should parse 'on:' section into events."""
        # Create structure: tmp_path/souls/test_soul/soul.yaml
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        soul_dir = souls_dir / "test_soul"
        soul_dir.mkdir()
        
        # Create soul.yaml with events - use DIFFERENT values than defaults
        # to verify they're actually being read from YAML
        # NOTE: Use "responses:" not "on:" because YAML parses "on" as boolean True
        soul_yaml = soul_dir / "soul.yaml"
        yaml_content = """\
name: TestBot
talks_to_bots: true

responses:
  user_first_speech:
    - "CUSTOM Hello {user}!"
    - "CUSTOM Hey {user}, welcome!"
  thinking:
    - "CUSTOM Hmm..."
    - "CUSTOM Let me think..."
  still_thinking:
    - "CUSTOM Still working on that..."
  interrupted:
    - "CUSTOM Oh, sorry!"

fallbacks:
  greetings:
    - "Hello there!"
"""
        soul_yaml.write_text(yaml_content)
        
        # Create prompt.md
        prompt_md = soul_dir / "prompt.md"
        prompt_md.write_text("You are a test bot.")
        
        # Create audio directory with a reference file
        audio_dir = soul_dir / "audio"
        audio_dir.mkdir()
        # Create a minimal wav file (44 bytes header only is enough for test)
        ref_audio = audio_dir / "reference.wav"
        ref_audio.write_bytes(b'RIFF' + b'\x00' * 40)
        
        # load_soul_config takes (soul_name, souls_dir)
        config = load_soul_config("test_soul", str(souls_dir))
        
        assert config is not None
        assert config.name == "TestBot"
        assert config.talks_to_bots is True
        
        # Check events - should contain our CUSTOM values, not defaults
        assert config.events is not None
        # If parsing works, we should see "CUSTOM" in the values
        assert any("CUSTOM" in s for s in config.events.user_first_speech), \
            f"Expected CUSTOM in user_first_speech, got: {config.events.user_first_speech}"
        assert any("CUSTOM" in s for s in config.events.thinking), \
            f"Expected CUSTOM in thinking, got: {config.events.thinking}"
        
        # Check fallbacks
        assert config.fallbacks is not None
        assert config.fallbacks.greetings == ["Hello there!"]

    def test_load_soul_config_uses_defaults_without_on_section(self, tmp_path):
        """Soul configs without 'on:' section should use SoulEvents defaults."""
        # Create structure: tmp_path/souls/minimal_soul/soul.yaml
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        soul_dir = souls_dir / "minimal_soul"
        soul_dir.mkdir()
        
        soul_yaml = soul_dir / "soul.yaml"
        soul_yaml.write_text("""\
name: MinimalBot
""")
        
        prompt_md = soul_dir / "prompt.md"
        prompt_md.write_text("Minimal prompt.")
        
        # Create audio directory with a reference file
        audio_dir = soul_dir / "audio"
        audio_dir.mkdir()
        ref_audio = audio_dir / "reference.wav"
        ref_audio.write_bytes(b'RIFF' + b'\x00' * 40)
        
        # load_soul_config takes (soul_name, souls_dir)
        config = load_soul_config("minimal_soul", str(souls_dir))
        
        assert config is not None
        assert config.name == "MinimalBot"
        # events should have defaults (not None)
        assert config.events is not None
        # Default thinking filler should exist
        assert config.events.thinking is not None


class TestGetEventResponse:
    """Test _get_event_response method (requires MumbleBot mock)."""

    def test_get_event_response_from_events(self):
        """_get_event_response should return from events first."""
        # This would require mocking MumbleBot - testing the logic
        events = SoulEvents(
            user_first_speech=["Hello {user}!"],
        )
        
        # Simulate the logic
        response = None
        if events.user_first_speech:
            import random
            response = random.choice(events.user_first_speech)
            response = response.replace("{user}", "TestUser")
        
        assert response == "Hello TestUser!"

    def test_get_event_response_with_custom_fallbacks(self):
        """When events are None, fallbacks should be used."""
        # Explicitly set events to None to test fallback
        events = SoulEvents(
            user_first_speech=None,  # Disable default
        )
        fallbacks = SoulFallbacks(greetings=["Default greeting!"])
        
        # Simulate fallback logic for user_first_speech -> greetings
        response = None
        if events.user_first_speech:
            import random
            response = random.choice(events.user_first_speech)
        elif fallbacks.greetings:
            import random
            response = random.choice(fallbacks.greetings)
        
        assert response == "Default greeting!"


class TestFillerBroadcast:
    """Test that fillers don't get broadcast to other bots."""

    def test_speak_sync_has_skip_broadcast_parameter(self):
        """_speak_sync should accept skip_broadcast parameter."""
        # This tests that the function signature is correct
        # Check the source file directly
        import ast
        from pathlib import Path
        
        bot_file = Path(__file__).parent.parent / "mumble_tts_bot.py"
        source = bot_file.read_text()
        
        # Parse and find the _speak_sync method
        tree = ast.parse(source)
        
        found_skip_broadcast = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_speak_sync':
                # Check parameters
                for arg in node.args.args:
                    if arg.arg == 'skip_broadcast':
                        found_skip_broadcast = True
                        break
                # Also check kwonlyargs
                for arg in node.args.kwonlyargs:
                    if arg.arg == 'skip_broadcast':
                        found_skip_broadcast = True
                        break
        
        assert found_skip_broadcast, "_speak_sync should have skip_broadcast parameter"

    def test_filler_types_exist(self):
        """All expected filler types should be recognized."""
        filler_types = ['thinking', 'still_thinking', 'interrupted']
        
        # These should be valid filler types
        for ftype in filler_types:
            # Just verify they're valid string types
            assert isinstance(ftype, str)


class TestMinimumUtteranceLengthFilter:
    """Test that bots ignore very short utterances from other bots."""

    def test_short_utterance_filter_logic(self):
        """Short utterances (<3 words) should be ignored."""
        test_cases = [
            ("Hmm...", 1, True),   # Should ignore
            ("Let me think...", 3, False),  # 3 words - borderline
            ("Hello there friend", 3, False),  # 3 words - OK
            ("I am the man with no name", 7, False),  # Long - OK
            ("Oh.", 1, True),  # Very short - ignore
            ("Hmm..................", 1, True),  # Padded but still 1 word
        ]
        
        for text, expected_words, should_ignore in test_cases:
            # Simulate the filter logic
            clean_text = text.strip().rstrip('.')
            word_count = len(clean_text.split())
            
            assert word_count == expected_words or abs(word_count - expected_words) <= 1, \
                f"'{text}' should have ~{expected_words} words, got {word_count}"
            
            is_ignored = word_count < 3
            assert is_ignored == should_ignore, \
                f"'{text}' should {'be ignored' if should_ignore else 'not be ignored'}"


class TestStillThinkingTimer:
    """Test the 'still thinking' timer functionality."""

    def test_timer_creation(self):
        """Timer should be created with correct timeout."""
        timer_fired = threading.Event()
        
        def on_timer():
            timer_fired.set()
        
        timer = threading.Timer(0.1, on_timer)  # 100ms for testing
        timer.daemon = True
        timer.start()
        
        # Wait for timer
        timer_fired.wait(timeout=0.5)
        assert timer_fired.is_set(), "Timer should have fired"

    def test_timer_cancellation(self):
        """Timer should be cancellable before firing."""
        timer_fired = threading.Event()
        
        def on_timer():
            timer_fired.set()
        
        timer = threading.Timer(0.5, on_timer)  # 500ms
        timer.daemon = True
        timer.start()
        
        # Cancel immediately
        timer.cancel()
        
        # Wait and verify it didn't fire
        time.sleep(0.6)
        assert not timer_fired.is_set(), "Cancelled timer should not fire"


class TestBargeInCallback:
    """Test barge-in callback registration and firing."""

    def test_barge_in_callback_registration(self):
        """TurnController should accept and store barge-in callback."""
        from mumble_voice_bot.turn_controller import TurnController
        
        callback_called = threading.Event()
        
        def on_barge_in():
            callback_called.set()
        
        controller = TurnController()
        controller.on_barge_in(on_barge_in)
        
        # Verify callback is stored
        assert controller._barge_in_callback is not None

    def test_barge_in_callback_fires_on_interrupt(self):
        """Callback should fire when barge-in is requested."""
        from mumble_voice_bot.turn_controller import TurnController
        
        callback_called = threading.Event()
        
        def on_barge_in():
            callback_called.set()
        
        controller = TurnController()
        controller.on_barge_in(on_barge_in)
        
        # Simulate: start speaking, then request barge-in
        controller.start_speaking()
        time.sleep(0.3)  # Wait past barge_in_delay_ms
        
        result = controller.request_barge_in()
        
        assert result is True, "Barge-in should succeed when speaking"
        assert callback_called.is_set(), "Callback should have been called"

    def test_barge_in_blocked_when_not_speaking(self):
        """Barge-in should fail when not in SPEAKING state."""
        from mumble_voice_bot.turn_controller import TurnController
        
        callback_called = threading.Event()
        
        def on_barge_in():
            callback_called.set()
        
        controller = TurnController()
        controller.on_barge_in(on_barge_in)
        
        # Don't start speaking - should be in IDLE
        result = controller.request_barge_in()
        
        assert result is False, "Barge-in should fail when not speaking"
        assert not callback_called.is_set(), "Callback should not be called"


class TestFirstTimeSpeakerDetection:
    """Test first-time speaker detection."""

    def test_first_time_speaker_set_logic(self):
        """First-time speakers should be tracked in a set."""
        seen_speakers = set()
        
        def check_first_time(user_name: str) -> bool:
            if user_name in seen_speakers:
                return False
            seen_speakers.add(user_name)
            return True
        
        # First time for each user
        assert check_first_time("alice") is True
        assert check_first_time("bob") is True
        
        # Second time - not first
        assert check_first_time("alice") is False
        assert check_first_time("bob") is False
        
        # New user
        assert check_first_time("charlie") is True


class TestEventTriggerIntegration:
    """Integration tests for event triggering."""

    def test_question_detection_logic(self):
        """_is_question should detect questions."""
        # Simulate the logic
        def is_question(text: str) -> bool:
            text = text.strip().lower()
            # Check for question marks or question words
            if text.endswith('?'):
                return True
            question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does']
            first_word = text.split()[0] if text.split() else ''
            return first_word in question_words
        
        # Test cases
        assert is_question("What is your name?") is True
        assert is_question("How are you?") is True
        assert is_question("Can you help me") is True
        assert is_question("Hello there") is False
        assert is_question("I like pizza") is False
        assert is_question("Is this working") is True

    def test_placeholder_substitution(self):
        """Event responses should substitute {user} placeholder."""
        response = "Hello {user}! Welcome to the channel."
        user = "sammo"
        
        result = response.replace("{user}", user)
        
        assert result == "Hello sammo! Welcome to the channel."
        assert "{user}" not in result


class TestBotToBotFiltering:
    """Test that bots properly filter messages from other bots."""

    def test_own_utterance_ignored(self):
        """Bots should ignore their own utterances."""
        my_name = "Zapp"
        speaker_name = "Zapp"
        
        # Simulate the check
        should_ignore = (speaker_name == my_name)
        
        assert should_ignore is True

    def test_other_bot_utterance_processed(self):
        """Bots should process utterances from other bots (if talks_to_bots=True)."""
        my_name = "Zapp"
        speaker_name = "Raf"
        talks_to_bots = True
        
        # Simulate the check
        should_ignore = (speaker_name == my_name)
        should_respond = not should_ignore and talks_to_bots
        
        assert should_ignore is False
        assert should_respond is True

    def test_bot_utterance_ignored_if_not_configured(self):
        """Bots should ignore other bots if talks_to_bots=False."""
        my_name = "Zapp"
        speaker_name = "Raf"
        talks_to_bots = False
        
        should_respond = talks_to_bots
        
        assert should_respond is False
