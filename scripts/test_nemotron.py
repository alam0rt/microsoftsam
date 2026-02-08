#!/usr/bin/env python3
"""Test script for Nemotron STT provider.

This allows testing the ASR flow locally without Mumble.

Usage:
    # Test with a wav file
    python scripts/test_nemotron.py path/to/audio.wav
    
    # Test with microphone (requires pyaudio)
    python scripts/test_nemotron.py --mic
    
    # Test with generated speech
    python scripts/test_nemotron.py --generate "Hello, this is a test"
"""

import argparse
import asyncio
import os
import sys
import time

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vendor", "LuxTTS"))

import numpy as np


def generate_test_audio(text: str, output_path: str = "/tmp/test_speech.wav"):
    """Generate test audio using TTS."""
    try:
        from zipvoice.luxvoice import LuxTTS
        import soundfile as sf
        
        print(f"[TTS] Generating audio for: '{text}'")
        tts = LuxTTS('YatharthS/LuxTTS', device='cuda')
        
        # Use default voice
        audio = tts.generate(text, steps=4)
        sf.write(output_path, audio, 24000)
        print(f"[TTS] Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Error] Failed to generate audio: {e}")
        return None


def record_from_mic(duration: float = 5.0, output_path: str = "/tmp/mic_recording.wav"):
    """Record from microphone."""
    try:
        import sounddevice as sd
        import soundfile as sf
        
        sample_rate = 16000
        print(f"[Mic] Recording {duration}s at {sample_rate}Hz...")
        print("[Mic] Speak now!")
        
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        sf.write(output_path, audio, sample_rate)
        print(f"[Mic] Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Error] Failed to record: {e}")
        print("[Hint] Install sounddevice: pip install sounddevice")
        return None


async def test_nemotron_transcribe(audio_path: str):
    """Test Nemotron transcription on an audio file."""
    from mumble_voice_bot.providers.nemotron_stt import NemotronStreamingASR, NemotronConfig
    import soundfile as sf
    
    print(f"\n{'='*60}")
    print(f"Testing Nemotron STT on: {audio_path}")
    print(f"{'='*60}\n")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    print(f"[Audio] Loaded: {len(audio)/sr:.2f}s at {sr}Hz")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy import signal
        print(f"[Audio] Resampling from {sr}Hz to 16000Hz...")
        samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, samples)
        sr = 16000
    
    # Convert to int16 PCM bytes
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    
    # Initialize Nemotron
    config = NemotronConfig(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        chunk_size_ms=160,
        device="cuda",
    )
    
    print("[Nemotron] Initializing...")
    start = time.time()
    
    asr = NemotronStreamingASR(config)
    await asr.initialize()
    
    init_time = time.time() - start
    print(f"[Nemotron] Initialized in {init_time:.2f}s")
    
    # Transcribe
    print("\n[Nemotron] Transcribing...")
    start = time.time()
    
    result = await asr.transcribe(
        audio_data=pcm_bytes,
        sample_rate=16000,
        sample_width=2,
        channels=1,
    )
    
    transcribe_time = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"RESULT: {result.text}")
    print(f"{'='*60}")
    print(f"\n[Stats] Transcription took {transcribe_time:.2f}s")
    print(f"[Stats] Real-time factor: {transcribe_time / (len(audio)/sr):.2f}x")
    
    return result


async def test_nemotron_streaming(audio_path: str, chunk_ms: int = 160):
    """Test Nemotron streaming transcription."""
    from mumble_voice_bot.providers.nemotron_stt import NemotronStreamingASR, NemotronConfig
    import soundfile as sf
    
    print(f"\n{'='*60}")
    print(f"Testing Nemotron STREAMING on: {audio_path}")
    print(f"{'='*60}\n")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    print(f"[Audio] Loaded: {len(audio)/sr:.2f}s at {sr}Hz")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy import signal
        print(f"[Audio] Resampling from {sr}Hz to 16000Hz...")
        samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, samples)
        sr = 16000
    
    # Convert to int16
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)
    
    # Initialize Nemotron
    config = NemotronConfig(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        chunk_size_ms=chunk_ms,
        device="cuda",
    )
    
    print("[Nemotron] Initializing...")
    asr = NemotronStreamingASR(config)
    await asr.initialize()
    print("[Nemotron] Ready for streaming")
    
    # Create async generator for audio chunks
    chunk_samples = int(chunk_ms * sr / 1000)
    
    async def audio_chunk_generator():
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            yield chunk.tobytes()
            # Simulate real-time
            await asyncio.sleep(chunk_ms / 1000 * 0.1)  # 10x faster than real-time
    
    # Stream transcribe
    print("\n[Streaming] Transcribing in chunks...\n")
    start = time.time()
    
    full_text = ""
    async for text, is_final in asr.transcribe_streaming(audio_chunk_generator(), sample_rate=16000):
        if is_final:
            print(f"  FINAL: {text}")
        else:
            print(f"  partial: {text}")
        full_text += text
    
    total_time = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"FULL TRANSCRIPT: {full_text}")
    print(f"{'='*60}")
    print(f"\n[Stats] Total time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron STT")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file to transcribe")
    parser.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--mic-duration", type=float, default=5.0, help="Mic recording duration")
    parser.add_argument("--generate", type=str, help="Generate test audio with this text")
    parser.add_argument("--streaming", action="store_true", help="Test streaming transcription")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk size for streaming (ms)")
    
    args = parser.parse_args()
    
    # Determine audio source
    audio_path = None
    
    if args.generate:
        audio_path = generate_test_audio(args.generate)
    elif args.mic:
        audio_path = record_from_mic(args.mic_duration)
    elif args.audio_file:
        audio_path = args.audio_file
    else:
        # Default test
        print("No audio source specified. Use --help for options.")
        print("\nRunning with generated test audio...")
        audio_path = generate_test_audio("Hello, this is a test of the Nemotron speech recognition system.")
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"[Error] Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Run test
    if args.streaming:
        asyncio.run(test_nemotron_streaming(audio_path, args.chunk_ms))
    else:
        asyncio.run(test_nemotron_transcribe(audio_path))


if __name__ == "__main__":
    main()
