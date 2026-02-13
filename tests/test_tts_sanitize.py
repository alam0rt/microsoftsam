"""Tests for TTS text sanitization.

Tests the _sanitize_for_tts function that removes emojis and
non-speakable characters from LLM output before TTS synthesis.
"""

from mumble_voice_bot.text_processing import sanitize_for_tts as _sanitize_for_tts


class TestSanitizeForTTS:
    """Tests for the _sanitize_for_tts function."""

    def test_removes_common_emojis(self):
        """Test that common emojis are removed."""
        # Actual examples from LLM logs
        assert _sanitize_for_tts("😄 Okay, let's see...") == "Okay, let's see..."
        assert _sanitize_for_tts("👍 Got it!") == "Got it!"
        assert _sanitize_for_tts("🎉 Sure thing!") == "Sure thing!"
        assert _sanitize_for_tts("🔥 You got it!") == "You got it!"
        assert _sanitize_for_tts("🤔 Hmm...") == "Hmm..."

    def test_removes_complex_emojis(self):
        """Test that complex/composite emojis are removed."""
        # Detective emoji with gender modifier
        assert _sanitize_for_tts("🕵️‍♂️ Alright, let's dig in") == "Alright, let's dig in"
        # Sun emoji
        assert _sanitize_for_tts("See you later! 🌞") == "See you later!"
        # Musical note
        assert _sanitize_for_tts("🎶 Playing a sound") == "Playing a sound"
        assert _sanitize_for_tts("🎵 Victory fanfare!") == "Victory fanfare!"

    def test_removes_multiple_emojis(self):
        """Test that multiple emojis in one string are all removed."""
        text = "😄 Great! 🎉 Let's go! 👍"
        assert _sanitize_for_tts(text) == "Great! Let's go!"

    def test_removes_emoji_at_end(self):
        """Test emojis at end of sentences."""
        assert _sanitize_for_tts("Morning's still early 🌞") == "Morning's still early"
        assert _sanitize_for_tts("Let's go! 🎵") == "Let's go!"

    def test_removes_formatting_asterisks(self):
        """Test that markdown asterisks are removed."""
        assert _sanitize_for_tts("This is *important*") == "This is important"
        assert _sanitize_for_tts("**Bold text**") == "Bold text"
        assert _sanitize_for_tts("***Really bold***") == "Really bold"

    def test_removes_underscores(self):
        """Test that markdown underscores are removed."""
        assert _sanitize_for_tts("This is _italic_") == "This is italic"
        assert _sanitize_for_tts("__underlined__") == "underlined"

    def test_removes_backticks(self):
        """Test that code backticks are removed."""
        assert _sanitize_for_tts("Run `command`") == "Run command"
        assert _sanitize_for_tts("```code block```") == "code block"

    def test_removes_hash_symbols(self):
        """Test that header hash symbols are removed."""
        assert _sanitize_for_tts("# Heading") == "Heading"
        assert _sanitize_for_tts("## Subheading") == "Subheading"

    def test_handles_em_dash(self):
        """Test em-dash handling (common in LLM output)."""
        # Em-dash should become space
        result = _sanitize_for_tts("mystery—what do you think we're looking for?")
        assert "—" not in result
        # Should still be readable
        assert "mystery" in result
        assert "what" in result

    def test_handles_double_dash(self):
        """Test that double dashes become spaces."""
        assert _sanitize_for_tts("one -- two") == "one two"
        assert _sanitize_for_tts("one---two") == "one two"

    def test_handles_spaced_dash(self):
        """Test that spaced dashes are cleaned up."""
        assert _sanitize_for_tts("one - two") == "one two"

    def test_preserves_contractions(self):
        """Test that contractions are preserved."""
        assert _sanitize_for_tts("I'm having a blast!") == "I'm having a blast!"
        assert _sanitize_for_tts("Let's go") == "Let's go"
        assert _sanitize_for_tts("Don't do that") == "Don't do that"
        assert _sanitize_for_tts("It's really cool") == "It's really cool"

    def test_preserves_basic_punctuation(self):
        """Test that basic punctuation is preserved."""
        assert _sanitize_for_tts("Hello, world!") == "Hello, world!"
        assert _sanitize_for_tts("Really? Yes.") == "Really? Yes."
        assert _sanitize_for_tts("Wait... okay.") == "Wait... okay."

    def test_preserves_numbers(self):
        """Test that numbers are preserved."""
        assert _sanitize_for_tts("It's about 5 bucks") == "It's about 5 bucks"
        assert _sanitize_for_tts("Call me at 555-1234") == "Call me at 555-1234"

    def test_cleans_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        assert _sanitize_for_tts("Hello    world") == "Hello world"
        assert _sanitize_for_tts("  Spaced  out  ") == "Spaced out"

    def test_removes_brackets(self):
        """Test that various brackets are removed."""
        assert _sanitize_for_tts("Click [here] for more") == "Click here for more"
        assert _sanitize_for_tts("Check {this} out") == "Check this out"
        assert _sanitize_for_tts("Use <tag> properly") == "Use tag properly"

    def test_removes_pipe_character(self):
        """Test that pipe characters are removed."""
        assert _sanitize_for_tts("Option A | Option B") == "Option A Option B"

    def test_real_llm_examples(self):
        """Test with actual problematic LLM outputs from logs."""
        examples = [
            (
                "😄 Okay, let's see... I'm vibing with the mood. How about playing \"Cute Real Sneeze Sound\" to lighten the vibe? Let's go! 🎵",
                'Okay, let\'s see... I\'m vibing with the mood. How about playing "Cute Real Sneeze Sound" to lighten the vibe? Let\'s go!'
            ),
            (
                "👍 Got it! I just played a cute sneeze sound to match the mood. Now, what's the next thing you want to chat about?",
                "Got it! I just played a cute sneeze sound to match the mood. Now, what's the next thing you want to chat about?"
            ),
            (
                "🔥 You got it! The Among Us sound is playing. Time to dive into the mystery—what do you think we're looking for?",
                "You got it! The Among Us sound is playing. Time to dive into the mystery what do you think we're looking for?"
            ),
            (
                "🕵️‍♂️ Alright, let's dig into the clues. I'll search for something relevant. One moment...",
                "Alright, let's dig into the clues. I'll search for something relevant. One moment..."
            ),
            (
                "😊 Perfect! If you need another sound, a clue, or just want to chat more, I'm here. Morning's still early, but we've got a lot of fun ahead. See you later! 🌞",
                "Perfect! If you need another sound, a clue, or just want to chat more, I'm here. Morning's still early, but we've got a lot of fun ahead. See you later!"
            ),
        ]

        for input_text, expected in examples:
            result = _sanitize_for_tts(input_text)
            # Check no emojis remain
            assert "😄" not in result
            assert "👍" not in result
            assert "🎵" not in result
            assert "🔥" not in result
            assert "🕵" not in result
            assert "😊" not in result
            assert "🌞" not in result
            # Check em-dash is handled
            assert "—" not in result

    def test_empty_string(self):
        """Test that empty strings are handled."""
        assert _sanitize_for_tts("") == ""
        assert _sanitize_for_tts("   ") == ""

    def test_emoji_only_string(self):
        """Test that emoji-only strings become empty."""
        assert _sanitize_for_tts("😄🎉👍") == ""
        assert _sanitize_for_tts("  🔥  ") == ""

    def test_removes_timestamp_prefix(self):
        """Test that timestamp prefixes are removed."""
        assert _sanitize_for_tts("[11:40 AM] Hello there") == "Hello there"
        assert _sanitize_for_tts("[2:30 PM] What's up") == "What's up"
        assert _sanitize_for_tts("[9:05] Good morning") == "Good morning"
        assert _sanitize_for_tts("[12:00 am] Late night") == "Late night"

    def test_removes_self_identification_prefix(self):
        """Test that bot self-identification prefixes are removed."""
        assert _sanitize_for_tts("Raf: yeah dude i got some sounds") == "yeah dude i got some sounds"
        assert _sanitize_for_tts("Bot: Here you go") == "Here you go"
        assert _sanitize_for_tts("Assistant: I can help") == "I can help"

    def test_removes_combined_timestamp_and_name(self):
        """Test that combined timestamp and name prefixes are removed."""
        assert _sanitize_for_tts("[11:40 AM] Raf: yeah dude") == "yeah dude"
        assert _sanitize_for_tts("[2:30 PM] Bot: Sure thing") == "Sure thing"
