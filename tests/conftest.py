"""Shared test fixtures for Wyoming protocol tests."""
import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_audio_16k_pcm() -> bytes:
    """Generate test PCM audio at 16kHz (Whisper input format).

    Creates 1 second of 440Hz sine wave.
    """
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


@pytest.fixture
def test_audio_48k_pcm() -> bytes:
    """Generate test PCM audio at 48kHz (Mumble/LuxTTS format).

    Creates 1 second of 440Hz sine wave.
    """
    sample_rate = 48000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


@pytest.fixture
def reference_audio_path(tmp_path) -> Path:
    """Create a temporary reference audio file for voice cloning tests."""
    sample_rate = 48000
    duration = 2.0

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)

    wav_path = tmp_path / "reference.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return wav_path


@pytest.fixture
def mock_wyoming_info():
    """Create mock Wyoming Info for STT server."""
    from wyoming.info import AsrModel, AsrProgram, Attribution, Info

    return Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper ASR",
                version="1.0",
                attribution=Attribution(
                    name="Faster Whisper",
                    url="https://github.com/SYSTRAN/faster-whisper",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name="base",
                        description="Base model",
                        languages=["en"],
                        installed=True,
                        version="1.0",
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://openai.com",
                        ),
                    )
                ],
            )
        ]
    )


@pytest.fixture
def mock_wyoming_tts_info():
    """Create mock Wyoming Info for TTS server."""
    from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

    return Info(
        tts=[
            TtsProgram(
                name="luxtts",
                description="LuxTTS voice cloning TTS",
                version="1.0",
                attribution=Attribution(
                    name="LuxTTS",
                    url="https://github.com/ysharma3501/LuxTTS",
                ),
                installed=True,
                voices=[
                    TtsVoice(
                        name="cloned",
                        description="Voice cloned from reference audio",
                        languages=["en"],
                        installed=True,
                        version="1.0",
                        attribution=Attribution(
                            name="LuxTTS",
                            url="https://github.com/ysharma3501/LuxTTS",
                        ),
                    )
                ],
            )
        ]
    )
