#!/usr/bin/env python3
"""
Parrot Bot - A minimal Mumble bot that echoes back what users say in their own voice.

No LLM - just ASR + voice cloning TTS.

Listens to voice in a Mumble channel, transcribes with NeMo Nemotron ASR,
clones the speaker's voice from their audio, and speaks the transcription back.

Usage:
    python parrot_bot.py --host mumble.example.com --user ParrotBot
"""
import argparse
import asyncio
import io
import logging
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import signal

# Module logger
logger = logging.getLogger(__name__)

# Add vendor paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

import pymumble_py3 as pymumble
import torch
from pymumble_py3.constants import PYMUMBLE_CLBK_SOUNDRECEIVED
from zipvoice.luxvoice import LuxTTS

# Import NeMo STT
try:
    from mumble_voice_bot.providers.nemotron_stt import NemotronConfig, NemotronStreamingASR
    NEMOTRON_AVAILABLE = True
except ImportError:
    NEMOTRON_AVAILABLE = False
    print("[ERROR] NeMo Nemotron STT not available. Install with: pip install nemo_toolkit")
    sys.exit(1)


def pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate RMS of PCM audio (16-bit signed, little-endian)."""
    if len(pcm_bytes) < 2:
        return 0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class StreamingLuxTTS:
    """Wrapper around LuxTTS for speech synthesis (used in standalone mode)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading LuxTTS on {device}...")
        self.tts = LuxTTS(device=device)
        logger.info("LuxTTS ready")

    def encode_prompt(self, audio_path: str, rms: float = 0.01, duration: float = 5.0) -> dict:
        """Encode a reference audio file for voice cloning."""
        return self.tts.encode_prompt(audio_path, duration=duration, rms=rms)

    def generate_speech(self, text: str, voice_prompt: dict, num_steps: int = 4, 
                        speed: float = 1.0):
        """Generate speech from text using the voice prompt.
        
        Returns a tensor that can be converted to numpy.
        """
        return self.tts.generate_speech(
            text, voice_prompt, num_steps=num_steps, speed=speed
        )


class ParrotBot:
    """A minimal Mumble bot that echoes back what users say in their own voice."""

    def __init__(
        self,
        host: str,
        user: str = "ParrotBot",
        port: int = 64738,
        password: str = None,
        channel: str = None,
        device: str = "cuda",
        asr_threshold: int = 2000,
        min_speech_duration: float = 0.5,
        silence_timeout: float = 1.5,
        # Shared services (optional - used when running in multi-persona mode)
        shared_tts=None,
        shared_stt=None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.asr_threshold = asr_threshold
        self.min_speech_duration = min_speech_duration
        self.silence_timeout = silence_timeout

        # Setup logging
        self.logger = logging.getLogger(f"parrot.{user}")
        
        # Use shared services if provided, otherwise create our own
        if shared_tts is not None:
            self.logger.info("Using shared TTS")
            self.tts = shared_tts
            self._owns_tts = False
        else:
            self.logger.info(f"Initializing TTS on {device}...")
            self.tts = StreamingLuxTTS(device=device)
            self._owns_tts = True

        if shared_stt is not None:
            self.logger.info("Using shared STT")
            self.stt = shared_stt
            self._owns_stt = False
        else:
            self.logger.info("Initializing NeMo Nemotron STT...")
            stt_config = NemotronConfig(
                model_name="nvidia/nemotron-speech-streaming-en-0.6b",
                chunk_size_ms=160,
                device=device,
            )
            self.stt = NemotronStreamingASR(stt_config)
            asyncio.run(self.stt.initialize())
            self._owns_stt = True
        self.logger.info("Ready")

        # Audio buffers per user
        self.user_audio_buffers = {}  # user_id -> list of PCM chunks
        self.user_last_audio_time = {}  # user_id -> timestamp
        self.user_is_speaking = {}  # user_id -> bool

        # Mumble client
        self.mumble = None
        self._speaking = threading.Event()
        self._shutdown = threading.Event()
        self._tts_queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Parrot")

        # Stats
        self._max_rms = 0

    def start(self):
        """Connect to Mumble server."""
        self.logger.info(f"Connecting to {self.host}:{self.port} as {self.user}...")
        
        self.mumble = pymumble.Mumble(
            self.host, self.user, 
            port=self.port, 
            password=self.password or "",
            reconnect=True
        )
        self.mumble.set_application_string("ParrotBot")
        self.mumble.set_codec_profile("audio")
        self.mumble.set_receive_sound(True)
        
        # Register audio callback
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED, self.on_sound_received
        )
        
        self.mumble.start()
        self.mumble.is_ready()
        
        # Join channel if specified
        if self.channel:
            try:
                ch = self.mumble.channels.find_by_name(self.channel)
                ch.move_in()
                self.logger.info(f"Joined channel: {self.channel}")
            except Exception as e:
                self.logger.warning(f"Failed to join channel '{self.channel}': {e}")
        
        self.logger.info(f"Connected! Listening for speech...")

    def run_forever(self):
        """Run the bot until interrupted."""
        # Start TTS playback thread
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()
        
        # Start silence checker thread
        silence_thread = threading.Thread(target=self._silence_checker, daemon=True)
        silence_thread.start()
        
        try:
            while not self._shutdown.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self._shutdown.set()
            if self.mumble:
                self.mumble.stop()

    def on_sound_received(self, user, sound_chunk):
        """Handle incoming audio from users."""
        user_id = user['session']
        user_name = user.get('name', 'Unknown')
        
        # Ignore our own audio
        if user_id == self.mumble.users.myself_session:
            return
        
        # Ignore audio while we're speaking (avoid echo)
        if self._speaking.is_set():
            return
        
        # Calculate RMS
        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)
        
        # Initialize buffer for new users
        if user_id not in self.user_audio_buffers:
            self.user_audio_buffers[user_id] = []
            self.user_is_speaking[user_id] = False
        
        # Voice activity detection
        if rms >= self.asr_threshold:
            if not self.user_is_speaking.get(user_id):
                self.logger.debug(f"{user_name} started speaking (RMS={rms})")
                self.user_is_speaking[user_id] = True
            
            # Buffer the audio
            self.user_audio_buffers[user_id].append(sound_chunk.pcm)
            self.user_last_audio_time[user_id] = time.time()

    def _silence_checker(self):
        """Check for silence and process completed utterances."""
        while not self._shutdown.is_set():
            time.sleep(0.1)
            current_time = time.time()
            
            for user_id in list(self.user_is_speaking.keys()):
                if not self.user_is_speaking.get(user_id):
                    continue
                
                last_audio = self.user_last_audio_time.get(user_id, 0)
                if current_time - last_audio >= self.silence_timeout:
                    # User stopped speaking
                    self.user_is_speaking[user_id] = False
                    
                    # Get the buffered audio
                    audio_chunks = self.user_audio_buffers.get(user_id, [])
                    self.user_audio_buffers[user_id] = []
                    
                    if audio_chunks:
                        # Get user info
                        try:
                            user = self.mumble.users[user_id]
                            user_name = user.get('name', 'Unknown')
                        except:
                            user_name = 'Unknown'
                        
                        # Process in background
                        self._executor.submit(
                            self._process_utterance, user_id, user_name, audio_chunks
                        )

    def _process_utterance(self, user_id: int, user_name: str, audio_chunks: list):
        """Process a completed utterance: transcribe and echo back."""
        # Concatenate audio
        pcm_data = b''.join(audio_chunks)
        duration = len(pcm_data) / (48000 * 2)  # 48kHz, 16-bit mono
        
        if duration < self.min_speech_duration:
            self.logger.debug(f"Ignoring short utterance from {user_name} ({duration:.1f}s)")
            return
        
        self.logger.debug(f"Processing {duration:.1f}s of audio from {user_name}...")
        
        # Convert 48kHz -> 16kHz for STT
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)
        
        # Normalize for STT
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        if rms_16k > 0.001:
            audio_16k = audio_16k * (0.1 / rms_16k)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
        
        # Convert back to PCM bytes for STT
        audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
        pcm_16k_bytes = audio_16k_int16.tobytes()
        
        # Transcribe with timing
        import time as _time
        transcribe_start = _time.time()
        try:
            result = asyncio.run(self.stt.transcribe(
                audio_data=pcm_16k_bytes,
                sample_rate=16000,
                sample_width=2,
                channels=1,
                language="en",
            ))
            text = result.text.strip()
            transcribe_time = _time.time() - transcribe_start
        except Exception as e:
            self.logger.error(f"STT error: {e}")
            return
        
        if not text or len(text) < 2:
            self.logger.debug(f"No speech detected from {user_name}")
            return
        
        # Log ASR result in same format as main bot
        self.logger.info(f'ASR ({transcribe_time*1000:.0f}ms): "{text}" [from {user_name}, {duration:.1f}s audio]')
        
        # Clone voice from the user's audio
        self.logger.debug(f"Cloning {user_name}'s voice...")
        try:
            # Write audio to temp file for voice cloning
            import soundfile as sf
            import tempfile
            
            # Convert bytes to float for soundfile
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Limit to 5 seconds for voice cloning
            max_samples = int(5.0 * 48000)
            if len(audio_float) > max_samples:
                audio_float = audio_float[:max_samples]
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_float, 48000)
                temp_path = f.name
            
            try:
                voice_prompt = self.tts.encode_prompt(temp_path, rms=0.01)
                self.logger.debug("Voice prompt encoded successfully")
            finally:
                os.unlink(temp_path)
        except Exception as e:
            self.logger.error(f"Voice cloning error: {e}")
            traceback.print_exc()
            return
        
        # Queue TTS job (voice cloning done, now queue text for streaming synthesis)
        self._tts_queue.put((user_name, text, voice_prompt))

    def _tts_worker(self):
        """Background thread for streaming TTS playback."""
        while not self._shutdown.is_set():
            try:
                item = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            user_name, text, voice_prompt = item
            self.logger.info(f'TTS generating: "{text}" [as {user_name}]')
            
            self._speaking.set()
            try:
                self._stream_tts(text, voice_prompt)
            except Exception as e:
                self.logger.error(f"TTS streaming error: {e}")
                traceback.print_exc()
            finally:
                self._speaking.clear()
                time.sleep(0.3)  # Brief pause after speaking

    def _stream_tts(self, text: str, voice_prompt: dict):
        """Stream TTS audio directly to Mumble as it's generated."""
        tts_start = time.time()
        total_audio_samples = 0
        first_chunk = True
        
        # Use generate_speech_streaming if available (shared TTS from mumble_tts_bot)
        # Falls back to generate_speech for standalone mode
        if hasattr(self.tts, 'generate_speech_streaming'):
            # Streaming mode - yield audio sentence by sentence
            for wav_chunk in self.tts.generate_speech_streaming(
                text, voice_prompt, num_steps=4, speed=1.0
            ):
                if self._shutdown.is_set():
                    break
                    
                if first_chunk:
                    first_audio_time = time.time() - tts_start
                    self.logger.info(f"TTS first audio: {first_audio_time*1000:.0f}ms")
                    first_chunk = False
                
                # Convert tensor to PCM and send to Mumble
                wav_float = wav_chunk.numpy().squeeze() if hasattr(wav_chunk, 'numpy') else wav_chunk
                if hasattr(wav_chunk, 'cpu'):
                    wav_float = wav_chunk.cpu().numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                chunk_samples = len(pcm)
                total_audio_samples += chunk_samples
                
                # Send to Mumble
                self.mumble.sound_output.add_sound(pcm.tobytes())
                
                # Wait for most of audio to play before generating next chunk
                # This creates natural pacing and prevents buffer overflow
                chunk_duration_sec = chunk_samples / 48000
                wait_time = chunk_duration_sec * 0.9 + 0.15  # 150ms between sentences
                if wait_time > 0.1:
                    time.sleep(wait_time)
        else:
            # Non-streaming fallback (standalone mode with basic LuxTTS)
            audio = self.tts.generate_speech(text, voice_prompt, num_steps=4, speed=1.0)
            if hasattr(audio, 'cpu'):
                audio_np = audio.cpu().numpy().squeeze()
            elif hasattr(audio, 'numpy'):
                audio_np = audio.numpy().squeeze()
            else:
                audio_np = np.array(audio).squeeze()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm = (audio_np * 32767).astype(np.int16)
            total_audio_samples = len(pcm)
            
            first_audio_time = time.time() - tts_start
            self.logger.info(f"TTS first audio: {first_audio_time*1000:.0f}ms")
            
            # Stream to Mumble in chunks
            pcm_bytes = pcm.tobytes()
            chunk_size = 48000 * 2 // 50  # 20ms chunks
            for i in range(0, len(pcm_bytes), chunk_size):
                if self._shutdown.is_set():
                    break
                chunk = pcm_bytes[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                self.mumble.sound_output.add_sound(chunk)
                time.sleep(0.018)
        
        tts_total = time.time() - tts_start
        audio_duration_ms = (total_audio_samples / 48000) * 1000
        self.logger.info(f"TTS complete: {tts_total*1000:.0f}ms synthesis, {audio_duration_ms:.0f}ms audio")


def main():
    parser = argparse.ArgumentParser(
        description="Parrot Bot - Echo back what users say in their own voice"
    )
    parser.add_argument("--host", required=True, help="Mumble server hostname")
    parser.add_argument("--port", type=int, default=64738, help="Mumble server port")
    parser.add_argument("--user", default="ParrotBot", help="Bot username")
    parser.add_argument("--password", help="Server password")
    parser.add_argument("--channel", help="Channel to join")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], 
                        help="Device for ML models")
    parser.add_argument("--asr-threshold", type=int, default=2000,
                        help="RMS threshold for voice activity detection")
    parser.add_argument("--min-speech", type=float, default=0.5,
                        help="Minimum speech duration in seconds")
    parser.add_argument("--silence-timeout", type=float, default=1.5,
                        help="Silence duration to consider speech ended")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging for standalone mode
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    bot = ParrotBot(
        host=args.host,
        user=args.user,
        port=args.port,
        password=args.password,
        channel=args.channel,
        device=args.device,
        asr_threshold=args.asr_threshold,
        min_speech_duration=args.min_speech,
        silence_timeout=args.silence_timeout,
    )
    
    bot.start()
    bot.run_forever()


if __name__ == "__main__":
    main()
