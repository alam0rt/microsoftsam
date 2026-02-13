"""Audio utilities for PCM processing, RMS calculation, and resampling.

Extracted from the duplicated code in mumble_tts_bot.py and parrot_bot.py.
"""

import numpy as np
from scipy import signal


def pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate RMS (root mean square) of 16-bit PCM audio.

    Args:
        pcm_bytes: Raw PCM audio data (16-bit signed, little-endian).

    Returns:
        Integer RMS value. Returns 0 for empty input.
    """
    if len(pcm_bytes) < 2:
        return 0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(audio) == 0:
        return 0
    return int(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def resample_48k_to_16k(audio_float: np.ndarray) -> np.ndarray:
    """Resample audio from 48kHz to 16kHz using polyphase filtering.

    Args:
        audio_float: Float32 audio samples at 48kHz.

    Returns:
        Float32 audio samples at 16kHz.
    """
    return signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)


def pcm_bytes_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to float32 array in [-1.0, 1.0].

    Args:
        pcm_bytes: Raw PCM audio data (16-bit signed, little-endian).

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0].
    """
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def float_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
    """Convert float32 audio to 16-bit PCM bytes.

    Args:
        audio_float: Float32 audio samples in [-1.0, 1.0].

    Returns:
        Raw PCM audio data (16-bit signed, little-endian).
    """
    audio_float = np.clip(audio_float, -1.0, 1.0)
    return (audio_float * 32767).astype(np.int16).tobytes()


def normalize_for_stt(audio_float: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio RMS for STT processing.

    Args:
        audio_float: Float32 audio samples.
        target_rms: Target RMS level for normalization.

    Returns:
        Normalized float32 audio clipped to [-1.0, 1.0].
    """
    rms = np.sqrt(np.mean(audio_float ** 2))
    if rms > 0.001:
        audio_float = audio_float * (target_rms / rms)
        audio_float = np.clip(audio_float, -1.0, 1.0).astype(np.float32)
    return audio_float


def pcm_duration(pcm_bytes: bytes, sample_rate: int = 48000, sample_width: int = 2) -> float:
    """Calculate duration of PCM audio in seconds.

    Args:
        pcm_bytes: Raw PCM audio data.
        sample_rate: Sample rate in Hz (default 48kHz for Mumble).
        sample_width: Bytes per sample (default 2 for 16-bit).

    Returns:
        Duration in seconds.
    """
    return len(pcm_bytes) / (sample_rate * sample_width)


def prepare_for_stt(pcm_chunks: list[bytes], source_rate: int = 48000, target_rate: int = 16000) -> bytes:
    """Prepare raw PCM chunks for STT: concatenate, resample, normalize.

    Takes a list of 48kHz PCM chunks (from Mumble), resamples to 16kHz,
    normalizes, and returns 16-bit PCM bytes suitable for STT.

    Args:
        pcm_chunks: List of raw PCM audio chunks (16-bit, 48kHz).
        source_rate: Source sample rate (default 48kHz).
        target_rate: Target sample rate (default 16kHz).

    Returns:
        16-bit PCM bytes at target sample rate, normalized for STT.
    """
    pcm_data = b''.join(pcm_chunks)
    audio_float = pcm_bytes_to_float(pcm_data)

    # Resample if needed
    if source_rate != target_rate:
        if source_rate == 48000 and target_rate == 16000:
            audio_float = resample_48k_to_16k(audio_float)
        else:
            ratio = target_rate / source_rate
            num_samples = int(len(audio_float) * ratio)
            audio_float = signal.resample(audio_float, num_samples).astype(np.float32)

    # Normalize
    audio_float = normalize_for_stt(audio_float)

    return float_to_pcm_bytes(audio_float)
