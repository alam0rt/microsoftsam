"""Tests for the sound effects tool.

Tests cover:
- Tool definition and schema
- Listing available sounds
- Searching sounds
- Playing sounds
- Audio loading and conversion
- Error handling
"""

import json
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mumble_voice_bot.tools.sound_effects import SoundEffectsTool

# --- Fixtures ---


@pytest.fixture
def temp_sounds_dir(tmp_path):
    """Create a temporary sounds directory with test sound files."""
    sounds_dir = tmp_path / "sounds"
    sounds_dir.mkdir()

    # Create a simple WAV file for testing
    def create_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 44100):
        """Create a simple test WAV file."""
        n_samples = int(duration_sec * sample_rate)
        # Generate a simple sine wave
        t = np.linspace(0, duration_sec, n_samples, False)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(str(path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

    # Create first sound with full metadata
    create_wav(sounds_dir / "01_among_us_sus.wav")
    (sounds_dir / "01_among_us_sus.json").write_text(json.dumps({
        "title": "Among Us Sus",
        "slug": "among-us-sus",
        "description": "Suspicious sound from Among Us game",
        "tags": ["among us", "sus", "suspicious", "meme", "game"],
        "rank": 1,
    }))

    # Create second sound with minimal metadata
    create_wav(sounds_dir / "02_victory_fanfare.wav")
    (sounds_dir / "02_victory_fanfare.json").write_text(json.dumps({
        "title": "Victory Fanfare",
        "slug": "victory-fanfare",
        "description": "Triumphant celebration music",
        "tags": ["victory", "win", "celebration"],
        "rank": 2,
    }))

    # Create third sound with no metadata file
    create_wav(sounds_dir / "03_sad_trombone.wav")

    # Create stereo WAV for testing conversion
    n_samples = int(0.5 * 44100)
    stereo_audio = np.random.randint(-32768, 32767, (n_samples, 2), dtype=np.int16)
    stereo_path = sounds_dir / "04_stereo_sound.wav"
    with wave.open(str(stereo_path), 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(stereo_audio.tobytes())
    (sounds_dir / "04_stereo_sound.json").write_text(json.dumps({
        "title": "Stereo Sound",
        "tags": ["stereo", "test"],
    }))

    # Create a non-audio file that should be ignored
    (sounds_dir / "readme.txt").write_text("This is not an audio file")
    (sounds_dir / "metadata.json").write_text(json.dumps({"type": "summary"}))

    return sounds_dir


@pytest.fixture
def sound_effects_tool(temp_sounds_dir):
    """Create a SoundEffectsTool with temp directory."""
    return SoundEffectsTool(sounds_dir=temp_sounds_dir)


@pytest.fixture
def play_callback():
    """Create a mock play callback."""
    callback = AsyncMock()
    return callback


# --- Test Classes ---


class TestSoundEffectsToolDefinition:
    """Test tool definition and schema."""

    def test_tool_name(self, sound_effects_tool):
        """Test that tool name is correct."""
        assert sound_effects_tool.name == "sound_effects"

    def test_tool_description(self, sound_effects_tool):
        """Test that tool has a description."""
        assert "sound effects" in sound_effects_tool.description.lower()
        assert "play" in sound_effects_tool.description.lower()

    def test_description_includes_auto_play_hint(self, temp_sounds_dir):
        """Test that auto_play is mentioned in description when enabled."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir, auto_play=True)
        assert "proactive" in tool.description.lower()

        tool_no_auto = SoundEffectsTool(sounds_dir=temp_sounds_dir, auto_play=False)
        assert "proactive" not in tool_no_auto.description.lower()

    def test_parameters_schema(self, sound_effects_tool):
        """Test that parameters schema is valid."""
        params = sound_effects_tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert "action" in params["required"]

    def test_action_enum(self, sound_effects_tool):
        """Test that action has correct enum values."""
        action_prop = sound_effects_tool.parameters["properties"]["action"]
        assert set(action_prop["enum"]) == {"search", "play", "list", "web_search"}


class TestSoundIndexing:
    """Test sound indexing and metadata loading."""

    def test_build_index(self, sound_effects_tool):
        """Test that index is built correctly."""
        index = sound_effects_tool._build_index()

        # Should have 4 audio files (excluding txt and non-audio json)
        assert len(index) == 4
        assert "01_among_us_sus" in index
        assert "02_victory_fanfare" in index
        assert "03_sad_trombone" in index
        assert "04_stereo_sound" in index

    def test_metadata_loaded(self, sound_effects_tool):
        """Test that metadata is loaded from JSON files."""
        index = sound_effects_tool._build_index()

        sus_sound = index["01_among_us_sus"]
        assert sus_sound["title"] == "Among Us Sus"
        assert sus_sound["tags"] == ["among us", "sus", "suspicious", "meme", "game"]
        assert sus_sound["description"] == "Suspicious sound from Among Us game"

    def test_sound_without_metadata(self, sound_effects_tool):
        """Test that sounds without JSON metadata still work."""
        index = sound_effects_tool._build_index()

        sad_trombone = index["03_sad_trombone"]
        assert sad_trombone["title"] == "03 sad trombone"  # Generated from filename
        assert sad_trombone["tags"] == []

    def test_refresh_index(self, sound_effects_tool, temp_sounds_dir):
        """Test that refresh_index clears and rebuilds."""
        # Build initial index
        initial_count = len(sound_effects_tool._build_index())

        # Add a new sound
        n_samples = int(0.1 * 44100)
        audio = (np.sin(np.linspace(0, 1, n_samples)) * 32767).astype(np.int16)
        new_sound = temp_sounds_dir / "05_new_sound.wav"
        with wave.open(str(new_sound), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(audio.tobytes())

        # Refresh and check
        new_count = sound_effects_tool.refresh_index()
        assert new_count == initial_count + 1

    def test_empty_directory(self, tmp_path):
        """Test handling of empty sounds directory."""
        empty_dir = tmp_path / "empty_sounds"
        empty_dir.mkdir()

        tool = SoundEffectsTool(sounds_dir=empty_dir)
        index = tool._build_index()
        assert len(index) == 0

    def test_nonexistent_directory(self, tmp_path):
        """Test handling of nonexistent sounds directory."""
        tool = SoundEffectsTool(sounds_dir=tmp_path / "does_not_exist")
        index = tool._build_index()
        assert len(index) == 0


class TestSoundSearch:
    """Test sound search functionality."""

    def test_search_by_title(self, sound_effects_tool):
        """Test searching by title."""
        results = sound_effects_tool.search_sounds("among us")
        assert len(results) >= 1
        assert results[0]["name"] == "01_among_us_sus"

    def test_search_by_tag(self, sound_effects_tool):
        """Test searching by tag."""
        results = sound_effects_tool.search_sounds("suspicious")
        assert len(results) >= 1
        assert any(r["name"] == "01_among_us_sus" for r in results)

    def test_search_by_description(self, sound_effects_tool):
        """Test searching by description content."""
        results = sound_effects_tool.search_sounds("celebration")
        assert len(results) >= 1
        assert any(r["name"] == "02_victory_fanfare" for r in results)

    def test_search_limit(self, sound_effects_tool):
        """Test search result limit."""
        results = sound_effects_tool.search_sounds("sound", limit=2)
        assert len(results) <= 2

    def test_search_no_results(self, sound_effects_tool):
        """Test search with no matching results."""
        results = sound_effects_tool.search_sounds("xyznonexistent")
        assert len(results) == 0

    def test_search_relevance_ordering(self, sound_effects_tool):
        """Test that exact matches rank higher."""
        results = sound_effects_tool.search_sounds("sus")
        # "sus" should match among_us_sus highly
        assert results[0]["name"] == "01_among_us_sus"


class TestSoundPlayback:
    """Test sound playback functionality."""

    @pytest.mark.asyncio
    async def test_play_sound_by_name(self, temp_sounds_dir, play_callback):
        """Test playing a sound by exact name."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=play_callback,
            enable_web_search=False,  # Disable web search for local-only test
        )

        result = await tool.play_sound("01_among_us_sus", search_web=False)

        assert "plays" in result.lower()
        assert "Among Us Sus" in result
        play_callback.assert_called_once()

        # Check callback received PCM bytes
        pcm_bytes, sample_rate = play_callback.call_args[0]
        assert isinstance(pcm_bytes, bytes)
        assert len(pcm_bytes) > 0
        assert sample_rate == 48000

    @pytest.mark.asyncio
    async def test_play_sound_by_search(self, temp_sounds_dir, play_callback):
        """Test playing a sound by search term."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=play_callback,
            enable_web_search=False,
        )

        result = await tool.play_sound("victory", search_web=False)

        assert "plays" in result.lower()
        play_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_sound_not_found(self, temp_sounds_dir, play_callback):
        """Test error handling when sound not found."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=play_callback,
            enable_web_search=False,  # Disable web search for predictable test
        )

        result = await tool.play_sound("nonexistent_xyz", search_web=False)

        assert "not found" in result.lower()
        play_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_play_without_callback(self, temp_sounds_dir):
        """Test playing without a callback configured."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)  # No callback

        result = await tool.play_sound("01_among_us_sus")

        assert "not available" in result.lower()


class TestAudioLoading:
    """Test audio file loading and conversion."""

    def test_load_wav_mono(self, sound_effects_tool, temp_sounds_dir):
        """Test loading mono WAV file."""
        result = sound_effects_tool._load_audio(
            str(temp_sounds_dir / "01_among_us_sus.wav")
        )

        assert result is not None
        pcm_bytes, sample_rate = result
        assert isinstance(pcm_bytes, bytes)
        assert sample_rate == 48000
        # Should be 16-bit samples
        assert len(pcm_bytes) % 2 == 0

    def test_load_wav_stereo_converts_to_mono(self, sound_effects_tool, temp_sounds_dir):
        """Test that stereo WAV is converted to mono."""
        result = sound_effects_tool._load_audio(
            str(temp_sounds_dir / "04_stereo_sound.wav")
        )

        assert result is not None
        pcm_bytes, sample_rate = result
        assert isinstance(pcm_bytes, bytes)

    def test_load_nonexistent_file(self, sound_effects_tool):
        """Test loading nonexistent file returns None."""
        result = sound_effects_tool._load_audio("/nonexistent/file.wav")
        assert result is None

    def test_load_invalid_file(self, sound_effects_tool, temp_sounds_dir):
        """Test loading non-audio file returns None."""
        result = sound_effects_tool._load_audio(
            str(temp_sounds_dir / "readme.txt")
        )
        assert result is None


class TestExecuteAction:
    """Test the execute method with different actions."""

    @pytest.mark.asyncio
    async def test_execute_list(self, sound_effects_tool):
        """Test list action."""
        result = await sound_effects_tool.execute(action="list")

        assert "sounds" in result.lower()
        assert "among" in result.lower() or "sus" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_list_with_limit(self, sound_effects_tool):
        """Test list action with limit."""
        result = await sound_effects_tool.execute(action="list", limit=2)

        # Should only list 2 sounds
        lines = [line for line in result.split("\n") if line.startswith("-")]
        assert len(lines) <= 2

    @pytest.mark.asyncio
    async def test_execute_search(self, temp_sounds_dir):
        """Test search action."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir, enable_web_search=False)
        result = await tool.execute(
            action="search",
            query="sus"
        )

        assert "matching" in result.lower() or "sus" in result.lower()
        assert "among" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_search_no_query(self, sound_effects_tool):
        """Test search action without query."""
        result = await sound_effects_tool.execute(action="search")

        assert "looking for" in result.lower() or "search term" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_search_no_results(self, temp_sounds_dir):
        """Test search action with no matching results."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir, enable_web_search=False)
        result = await tool.execute(
            action="search",
            query="xyznonexistent123"
        )

        assert "no sounds found" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_play(self, temp_sounds_dir, play_callback):
        """Test play action."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=play_callback,
            enable_web_search=False,
        )

        result = await tool.execute(action="play", query="victory")

        assert "plays" in result.lower()
        play_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_play_no_query(self, sound_effects_tool):
        """Test play action without query."""
        result = await sound_effects_tool.execute(action="play")

        assert "play" in result.lower() and ("what" in result.lower() or "name" in result.lower() or "search" in result.lower())

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, sound_effects_tool):
        """Test unknown action."""
        result = await sound_effects_tool.execute(action="invalid_action")

        assert "unknown action" in result.lower()


class TestConfigOptions:
    """Test configuration options."""

    def test_custom_sample_rate(self, temp_sounds_dir, play_callback):
        """Test custom sample rate configuration."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=play_callback,
            sample_rate=22050,
        )

        assert tool.sample_rate == 22050

    def test_auto_play_default_true(self, temp_sounds_dir):
        """Test that auto_play defaults to True."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)
        assert tool.auto_play is True

    def test_auto_play_false(self, temp_sounds_dir):
        """Test setting auto_play to False."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir, auto_play=False)
        assert tool.auto_play is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_sound_library(self, tmp_path):
        """Test behavior with empty sound library."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = SoundEffectsTool(sounds_dir=empty_dir, enable_web_search=False)

        list_result = await tool.execute(action="list")
        assert "no sounds" in list_result.lower()

        search_result = await tool.execute(action="search", query="test")
        assert "no sounds found" in search_result.lower()

    @pytest.mark.asyncio
    async def test_playback_callback_error(self, temp_sounds_dir):
        """Test handling of playback callback errors."""
        error_callback = AsyncMock(side_effect=Exception("Playback failed"))

        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            play_callback=error_callback,
            enable_web_search=False,
        )

        result = await tool.play_sound("01_among_us_sus", search_web=False)

        assert "failed" in result.lower()

    def test_malformed_json_metadata(self, tmp_path):
        """Test handling of malformed JSON metadata."""
        sounds_dir = tmp_path / "sounds"
        sounds_dir.mkdir()

        # Create sound with malformed JSON
        n_samples = int(0.1 * 44100)
        audio = (np.sin(np.linspace(0, 1, n_samples)) * 32767).astype(np.int16)
        with wave.open(str(sounds_dir / "test.wav"), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(audio.tobytes())

        (sounds_dir / "test.json").write_text("{ invalid json }")

        tool = SoundEffectsTool(sounds_dir=sounds_dir)
        index = tool._build_index()

        # Should still index the sound, just without metadata
        assert "test" in index
        assert index["test"]["tags"] == []


class TestWebSearchFunctionality:
    """Test MyInstants web search functionality."""

    def test_web_search_enabled_by_default(self, temp_sounds_dir):
        """Test that web search is enabled by default."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)
        assert tool.enable_web_search is True

    def test_web_search_can_be_disabled(self, temp_sounds_dir):
        """Test that web search can be disabled."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir, enable_web_search=False)
        assert tool.enable_web_search is False

    def test_cache_enabled_by_default(self, temp_sounds_dir):
        """Test that caching is enabled by default."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)
        assert tool.cache_web_sounds is True

    def test_parse_myinstants_search_basic(self, temp_sounds_dir):
        """Test parsing of MyInstants search results HTML."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)

        # Simulate HTML with instant buttons
        html = '''
        <div class="instant">
            <a href="/en/instant/test-sound/">Test Sound</a>
            <button onclick="play('/media/sounds/test.mp3')">Play</button>
        </div>
        <div class="instant">
            <a href="/en/instant/another-sound/">Another Sound</a>
            <button onclick="play('/media/sounds/another.mp3')">Play</button>
        </div>
        '''

        results = tool._parse_myinstants_search(html, limit=10)

        assert len(results) >= 1
        assert results[0]["slug"] == "test-sound"
        assert results[0]["title"] == "Test Sound"

    def test_parse_myinstants_search_respects_limit(self, temp_sounds_dir):
        """Test that parsing respects the limit parameter."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)

        html = '''
        <a href="/en/instant/sound1/">Sound 1</a>
        <a href="/en/instant/sound2/">Sound 2</a>
        <a href="/en/instant/sound3/">Sound 3</a>
        '''

        results = tool._parse_myinstants_search(html, limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_myinstants_disabled(self, temp_sounds_dir):
        """Test that search returns empty when web search is disabled."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            enable_web_search=False
        )

        results = await tool.search_myinstants("test")

        assert results == []

    @pytest.mark.asyncio
    async def test_web_search_action(self, temp_sounds_dir):
        """Test web_search action when disabled."""
        tool = SoundEffectsTool(
            sounds_dir=temp_sounds_dir,
            enable_web_search=False
        )

        result = await tool.execute(action="web_search", query="test")

        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_web_search_action_no_query(self, temp_sounds_dir):
        """Test web_search action without query."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)

        result = await tool.execute(action="web_search")

        assert "looking for" in result.lower()

    def test_web_cache_ttl(self, temp_sounds_dir):
        """Test that web search cache has a TTL."""
        tool = SoundEffectsTool(sounds_dir=temp_sounds_dir)

        assert tool._web_cache_ttl > 0
        assert hasattr(tool, '_web_search_cache')

    def test_sounds_dir_created_if_missing(self, tmp_path):
        """Test that sounds directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_sounds_dir"
        assert not new_dir.exists()

        SoundEffectsTool(sounds_dir=new_dir)

        assert new_dir.exists()


class TestConfigIntegration:
    """Test configuration integration."""

    def test_all_config_options(self, tmp_path):
        """Test that all config options are properly handled."""
        sounds_dir = tmp_path / "sounds"

        tool = SoundEffectsTool(
            sounds_dir=sounds_dir,
            play_callback=AsyncMock(),
            auto_play=False,
            sample_rate=22050,
            enable_web_search=False,
            cache_web_sounds=False,
            request_timeout=5.0,
        )

        assert tool.auto_play is False
        assert tool.sample_rate == 22050
        assert tool.enable_web_search is False
        assert tool.cache_web_sounds is False
        assert tool.request_timeout == 5.0
