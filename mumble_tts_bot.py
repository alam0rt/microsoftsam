#!/usr/bin/env python3
"""
Mumble TTS Bot - Reads text messages aloud using LuxTTS voice cloning.

Usage:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav
    
    # With ASR (transcribe voice to text):
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav --asr
    
    # Debug mode to tune VAD threshold:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav --asr --debug-rms
"""
import argparse
import os
import re
import sys
import time

# Add vendor paths for pymumble and LuxTTS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

import numpy as np
from scipy import signal

import pymumble_py3 as pymumble
from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, PYMUMBLE_CLBK_SOUNDRECEIVED

from zipvoice.luxvoice import LuxTTS


def pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate RMS (Root Mean Square) of 16-bit PCM audio.
    
    Returns an integer RMS value comparable to audioop.rms().
    Typical speech is 1000-10000, silence is <500.
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(audio) == 0:
        return 0
    # Calculate RMS and return as integer for compatibility with thresholds
    return int(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def strip_html(text: str) -> str:
    """Remove HTML tags from text (Mumble messages can contain HTML)."""
    return re.sub(r'<[^>]+>', '', text)


def extract_last_sentence(text: str) -> str:
    """Extract the last complete sentence from text.
    
    This helps with garbled ASR output by just taking the last
    recognizable sentence to speak back.
    """
    text = text.strip()
    if not text:
        return ""
    
    # Split on sentence-ending punctuation, keeping the punctuation
    # Match .!? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty strings and very short fragments
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
    
    if not sentences:
        # No clear sentences, just return the whole thing (trimmed)
        return text[:200] if len(text) > 200 else text
    
    # Return the last sentence
    last = sentences[-1]
    
    # If it doesn't end with punctuation, add a period
    if not re.search(r'[.!?]$', last):
        last = last + '.'
    
    return last


def get_best_device() -> str:
    """Auto-detect the best available compute device.
    
    Returns 'cuda' if NVIDIA GPU available, 'mps' for Apple Silicon,
    otherwise 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA available: {device_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[Device] MPS (Apple Silicon) available")
            return 'mps'
        else:
            print("[Device] Using CPU")
            return 'cpu'
    except ImportError:
        print("[Device] PyTorch not available, defaulting to CPU")
        return 'cpu'


# Common Whisper hallucination patterns (especially with whisper-tiny)
HALLUCINATION_PATTERNS = [
    r'^\.+$',  # Just dots/periods
    r'^[\s\.\,\!\?]+$',  # Just punctuation
    r'(.)\1{4,}',  # Same character repeated 5+ times
    r'(\b\w+\b)(\s+\1){2,}',  # Same word repeated 3+ times
    r'^(thanks for watching|subscribe|like and subscribe|thank you for watching)',  # YouTube artifacts
    r'^(music|applause|laughter|silence)$',  # Sound descriptions
    r'^\[.*\]$',  # Just bracketed text like [Music]
    r'^you$',  # Common single-word hallucination
]


class MumbleTTSBot:
    """A Mumble bot that speaks text messages using LuxTTS."""
    
    def __init__(
        self,
        host: str,
        user: str,
        port: int = 64738,
        password: str = '',
        channel: str = None,
        reference_audio: str = 'reference.wav',
        device: str = 'cpu',
        num_steps: int = 4,
        enable_asr: bool = False,
        asr_threshold: int = 2000,
        debug_rms: bool = False,
        debug_audio: bool = False,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        self.enable_asr = enable_asr
        
        # VAD (Voice Activity Detection) settings
        self.asr_threshold = asr_threshold  # RMS threshold for speech detection
        self.debug_rms = debug_rms  # Show RMS levels for threshold tuning
        self.debug_audio = debug_audio  # Save audio files for debugging
        self._max_rms = 0  # Track max RMS for debug display
        self._debug_audio_counter = 0  # Counter for debug audio files
        
        # ASR state per user
        self.audio_buffers = {}  # user_id -> list of PCM bytes
        self.speech_active_until = {}  # user_id -> timestamp when speech hold expires
        self.last_transcription_time = {}  # user_id -> timestamp of last incremental transcription
        self.speech_hold_duration = 1.2  # seconds to wait after speech stops before final transcription
        self.mimic_hold_duration = 0.6  # shorter hold for mimic mode - respond quickly like a child
        self.min_speech_duration = 0.8  # minimum seconds of audio to transcribe (shorter = more hallucinations)
        self.max_speech_duration = 30.0  # maximum seconds to buffer before forced transcription
        self.incremental_interval = 4.0  # transcribe every N seconds during ongoing speech
        
        # Mimic mode: capture user's voice and speak it back to them (like an annoying child)
        self.mimic_pending = {}  # user_id -> True if actively mimicking this user
        
        # Initialize TTS
        print(f"Loading LuxTTS model on {device}...")
        self.tts = LuxTTS('YatharthS/LuxTTS', device=device, threads=2)
        
        print(f"Encoding reference audio: {reference_audio}")
        self.voice_prompt = self.tts.encode_prompt(reference_audio, rms=0.01)
        
        # Initialize Mumble connection
        print(f"Connecting to {host}:{port} as '{user}'...")
        self.mumble = pymumble.Mumble(
            host=host,
            user=user,
            port=port,
            password=password,
            reconnect=True,
        )
        
        # Enable receiving audio if ASR is enabled
        if self.enable_asr:
            self.mumble.set_receive_sound(True)
            print(f"ASR enabled - listening for voice (threshold: {asr_threshold})")
            if self.debug_rms:
                print("Debug RMS mode: showing audio levels for threshold tuning")
        
        # Set up text message callback
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            self.on_message
        )
        
        # Set up sound received callback for ASR
        if self.enable_asr:
            self.mumble.callbacks.set_callback(
                PYMUMBLE_CLBK_SOUNDRECEIVED,
                self.on_sound_received
            )
        
    def start(self):
        """Start the bot and connect to the server."""
        self.mumble.start()
        self.mumble.is_ready()  # Blocks until connected
        print("Connected to Mumble server!")
        
        # Join channel if specified
        if self.channel:
            self.join_channel(self.channel)
            
    def join_channel(self, channel_name: str):
        """Join a channel by name."""
        try:
            channel = self.mumble.channels.find_by_name(channel_name)
            channel.move_in()
            print(f"Joined channel: {channel_name}")
        except Exception as e:
            print(f"Failed to join channel '{channel_name}': {e}")
    
    def on_message(self, message):
        """Callback for received text messages."""
        # Extract and clean the message text
        text = strip_html(message.message)
        
        if not text.strip():
            return
            
        # Get sender name if available
        sender = "Someone"
        if hasattr(message, 'actor') and message.actor in self.mumble.users:
            sender = self.mumble.users[message.actor]['name']
        
        print(f"[{sender}]: {text}")
        
        # Check for @mimic command
        if text.strip().lower() == '@mimic':
            if not self.enable_asr:
                print("[Mimic] ASR is not enabled, cannot mimic")
                return
            # Get user ID from actor
            if hasattr(message, 'actor'):
                user_id = message.actor
                self.mimic_pending[user_id] = True
                print(f"[Mimic] Now mimicking {sender} (session {user_id})! Say @stop to stop.")
            return
        
        # Check for @stop command
        if text.strip().lower() == '@stop':
            if hasattr(message, 'actor'):
                user_id = message.actor
                if self.mimic_pending.get(user_id):
                    self.mimic_pending[user_id] = False
                    print(f"[Mimic] Stopped mimicking {sender}")
            return
        
        # Generate and play speech
        try:
            self.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
    
    def on_sound_received(self, user, sound_chunk):
        """Callback for received audio - implements VAD-based speech detection."""
        user_id = user['session']
        user_name = user.get('name', 'Unknown')
        
        # Calculate RMS (Root Mean Square) energy of the audio
        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)
        
        # Debug display for threshold tuning
        if self.debug_rms:
            bar_width = min(rms // 100, 50)
            threshold_pos = min(self.asr_threshold // 100, 50)
            if rms < self.asr_threshold:
                bar = '-' * bar_width
            else:
                bar = '-' * threshold_pos + '+' * (bar_width - threshold_pos)
            print(f'\r[{user_name:12}] RMS: {rms:5d} / {self._max_rms:5d}  |{bar:<50}|', end='', flush=True)
        
        # Initialize buffer for new users
        if user_id not in self.audio_buffers:
            self.audio_buffers[user_id] = []
            self.speech_active_until[user_id] = 0
            self.last_transcription_time[user_id] = 0
            self.mimic_pending[user_id] = False
        
        current_time = time.time()
        
        # Check if this chunk contains speech (above threshold)
        if rms > self.asr_threshold:
            # Speech detected - add to buffer and extend hold time
            # Use shorter hold time in mimic mode for faster response
            hold_time = self.mimic_hold_duration if self.mimic_pending.get(user_id) else self.speech_hold_duration
            self.audio_buffers[user_id].append(sound_chunk.pcm)
            self.speech_active_until[user_id] = current_time + hold_time
            
            buffer_duration = self._get_buffer_duration(user_id)
            time_since_last = current_time - self.last_transcription_time[user_id]
            
            # If mimic mode is active, skip incremental ASR - we'll process on speech end
            if self.mimic_pending.get(user_id):
                # Force process if we've accumulated too much (max 15s for mimic)
                if buffer_duration >= 15.0:
                    if self.debug_rms:
                        print()
                    print(f"[Mimic] Max buffer reached, processing...")
                    self._mimic_user(user, user_id)
                # Otherwise just accumulate
            # Incremental transcription: transcribe periodically during ongoing speech
            elif buffer_duration >= self.min_speech_duration and time_since_last >= self.incremental_interval:
                if self.debug_rms:
                    print()  # Newline after RMS display
                self._transcribe_buffer(user, user_id, incremental=True)
            
            # Check for max duration (forced transcription for very long speech)
            elif buffer_duration >= self.max_speech_duration:
                if self.debug_rms:
                    print()  # Newline after RMS display
                print(f"[ASR] Max duration reached for {user_name}, transcribing...")
                self._transcribe_buffer(user, user_id, incremental=False)
        else:
            # Below threshold - but if we're in "hold" period, still collect audio
            # This captures trailing sounds and pauses between words
            if current_time < self.speech_active_until[user_id]:
                self.audio_buffers[user_id].append(sound_chunk.pcm)
            elif self.audio_buffers[user_id]:
                # Hold period expired and we have buffered audio - user stopped speaking
                if self.debug_rms:
                    print()  # Newline after RMS display
                
                # Check if this user has mimic mode active
                if self.mimic_pending.get(user_id):
                    # Mimic mode: repeat what they said back to them (like an annoying child)
                    buffer_duration = self._get_buffer_duration(user_id)
                    if buffer_duration >= 1.5:  # Need at least 1.5s for decent clone
                        self._mimic_user(user, user_id)
                    else:
                        # Too short - just discard and wait for next utterance
                        self.audio_buffers[user_id] = []
                else:
                    # Normal final transcription
                    self._transcribe_buffer(user, user_id, incremental=False)
    
    def _get_buffer_duration(self, user_id):
        """Calculate total duration of buffered audio in seconds."""
        if user_id not in self.audio_buffers:
            return 0
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[user_id])
        # 48kHz, 16-bit (2 bytes per sample), mono
        return total_bytes / (48000 * 2)
    
    def _is_hallucination(self, text: str) -> bool:
        """Check if transcribed text is likely a Whisper hallucination."""
        text_lower = text.lower().strip()
        
        # Too short is suspicious
        if len(text_lower) < 2:
            return True
        
        # Check against known hallucination patterns
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _transcribe_buffer(self, user, user_id, incremental=False):
        """Transcribe buffered audio using Whisper.
        
        Args:
            user: User dict from pymumble
            user_id: User session ID
            incremental: If True, this is a partial transcription during ongoing speech
        """
        if not self.audio_buffers.get(user_id):
            return
        
        user_name = user.get('name', 'Unknown')
        buffer_duration = self._get_buffer_duration(user_id)
        
        # Skip if too short (likely noise)
        if buffer_duration < self.min_speech_duration:
            if not incremental:  # Only print skip message for final transcription
                print(f"[ASR] Skipping short audio from {user_name} ({buffer_duration:.2f}s < {self.min_speech_duration}s)")
                self.audio_buffers[user_id] = []
            return
        
        # Concatenate all PCM chunks
        pcm_data = b''.join(self.audio_buffers[user_id])
        
        # For incremental: clear buffer and update timestamp
        # For final: also clear buffer
        self.audio_buffers[user_id] = []
        self.last_transcription_time[user_id] = time.time()
        
        if not incremental:
            self._max_rms = 0  # Reset max RMS tracking only on final
        
        # Convert PCM bytes to numpy array (16-bit signed, 48kHz mono)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Convert to float32 [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Calculate RMS before any processing
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Skip if mostly silence (prevents Whisper hallucinations)
        if rms < 0.02:
            if not incremental:
                print(f"[ASR] Skipping quiet audio from {user_name} (RMS: {rms:.4f} < 0.02)")
            return
        
        # Note: Normalization happens AFTER resampling below
        
        # Resample from 48kHz to 16kHz for Whisper FIRST
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)
        
        # Calculate RMS after resampling
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        
        # Normalize audio AFTER resampling for consistent Whisper input
        # Whisper works best with audio around 0.1-0.2 RMS
        target_rms = 0.1
        if rms_16k > 0.001:  # Avoid division by near-zero
            audio_16k = audio_16k * (target_rms / rms_16k)
            # Clip to prevent clipping artifacts
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
        
        marker = "..." if incremental else ""
        print(f"[ASR] Transcribing {buffer_duration:.1f}s from {user_name} (RMS: {rms:.3f} @ 48k -> {rms_16k:.3f} @ 16k -> {target_rms}){marker}")
        
        # Debug: save audio to file for inspection
        if self.debug_audio:
            import scipy.io.wavfile as wavfile
            self._debug_audio_counter += 1
            filename = f"debug_audio_{self._debug_audio_counter:04d}_{user_name}.wav"
            wavfile.write(filename, 16000, audio_16k)
            print(f"[DEBUG] Saved audio to {filename} (final RMS: {np.sqrt(np.mean(audio_16k**2)):.3f})")
        
        # Transcribe using Whisper
        try:
            result = self.tts.transcriber(audio_16k)
            text = result.get('text', '').strip()
            
            # Filter out likely hallucinations from whisper-tiny
            if text and not self._is_hallucination(text):
                prefix = f"[ASR {user_name}]:" if not incremental else f"[ASR {user_name} ...]:"
                print(f"{prefix} {text}")
            elif not incremental:
                if text:
                    print(f"[ASR] Filtered hallucination from {user_name}: {text[:50]}...")
                else:
                    print(f"[ASR] No speech detected in audio from {user_name}")
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
    
    def _mimic_user(self, user, user_id):
        """Capture user's voice, transcribe it, and speak it back in their voice.
        
        This uses the captured audio as both:
        1. Reference audio for voice cloning
        2. Input for transcription to get the text
        """
        if not self.audio_buffers.get(user_id):
            return
        
        user_name = user.get('name', 'Unknown')
        buffer_duration = self._get_buffer_duration(user_id)
        
        # Cap at 25 seconds to avoid Whisper's 30s limit
        max_mimic_duration = 25.0
        if buffer_duration > max_mimic_duration:
            print(f"[Mimic] Trimming {buffer_duration:.1f}s to {max_mimic_duration}s")
            # Keep only the last max_mimic_duration seconds
            bytes_to_keep = int(max_mimic_duration * 48000 * 2)
            total_bytes = sum(len(chunk) for chunk in self.audio_buffers[user_id])
            bytes_to_skip = total_bytes - bytes_to_keep
            
            # Skip chunks until we've skipped enough
            skipped = 0
            while self.audio_buffers[user_id] and skipped < bytes_to_skip:
                chunk = self.audio_buffers[user_id].pop(0)
                skipped += len(chunk)
            
            buffer_duration = self._get_buffer_duration(user_id)
        
        print(f"[Mimic] Processing {buffer_duration:.1f}s of audio from {user_name}...")
        
        # Concatenate all PCM chunks
        pcm_data = b''.join(self.audio_buffers[user_id])
        self.audio_buffers[user_id] = []
        
        # Convert PCM bytes to numpy array (16-bit signed, 48kHz mono)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 0.02:
            print(f"[Mimic] Audio too quiet from {user_name} (RMS: {rms:.4f})")
            return
        
        # Resample to 16kHz for Whisper
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        
        # Normalize for Whisper
        target_rms = 0.1
        if rms_16k > 0.001:
            audio_16k_normalized = audio_16k * (target_rms / rms_16k)
            audio_16k_normalized = np.clip(audio_16k_normalized, -1.0, 1.0).astype(np.float32)
        else:
            audio_16k_normalized = audio_16k
        
        # Step 1: Transcribe the audio to get the text
        try:
            result = self.tts.transcriber(audio_16k_normalized)
            full_text = result.get('text', '').strip()
            
            if not full_text or self._is_hallucination(full_text):
                print(f"[Mimic] Could not transcribe speech from {user_name}")
                return
            
            # Extract just the last sentence for cleaner TTS
            text = extract_last_sentence(full_text)
            
            print(f"[Mimic] {user_name} said: \"{full_text}\"")
            if text != full_text:
                print(f"[Mimic] Using last sentence: \"{text}\"")
            
        except Exception as e:
            print(f"[Mimic] Transcription error: {e}")
            return
        
        # Step 2: Use the original 48kHz audio as voice reference
        # LuxTTS encode_prompt expects audio data - we'll save temporarily
        import tempfile
        import scipy.io.wavfile as wavfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                wavfile.write(temp_path, 48000, audio_int16)
            
            # Encode the user's voice as a prompt
            print(f"[Mimic] Cloning {user_name}'s voice...")
            user_voice_prompt = self.tts.encode_prompt(temp_path, rms=0.01)
            
            # Step 3: Generate speech with their voice saying their words
            print(f"[Mimic] Speaking back: \"{text}\"")
            wav = self.tts.generate_speech(text, user_voice_prompt, num_steps=self.num_steps)
            wav_float = wav.numpy().squeeze()
            
            # Convert and send
            wav_float = np.clip(wav_float, -1.0, 1.0)
            pcm_out = (wav_float * 32767).astype(np.int16)
            self.mumble.sound_output.add_sound(pcm_out.tobytes())
            
            print(f"[Mimic] Done mimicking {user_name}!")
            
            # Keep mimic mode active - don't reset mimic_pending
            
        except Exception as e:
            print(f"[Mimic] Voice cloning/synthesis error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    def speak(self, text: str):
        """Generate speech from text and send to Mumble."""
        # Generate speech with LuxTTS
        wav = self.tts.generate_speech(text, self.voice_prompt, num_steps=self.num_steps)
        wav_float = wav.numpy().squeeze()
        
        # Convert float32 [-1, 1] to int16 PCM
        # Clip to prevent overflow
        wav_float = np.clip(wav_float, -1.0, 1.0)
        pcm = (wav_float * 32767).astype(np.int16)
        
        # Send to Mumble (48kHz 16-bit mono PCM)
        self.mumble.sound_output.add_sound(pcm.tobytes())
        
    def run_forever(self):
        """Keep the bot running."""
        print("Bot is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description='Mumble TTS Bot - Reads text messages using LuxTTS'
    )
    parser.add_argument('--host', default='localhost',
                        help='Mumble server address')
    parser.add_argument('--port', type=int, default=64738,
                        help='Mumble server port')
    parser.add_argument('--user', default='TTSBot',
                        help='Bot username')
    parser.add_argument('--password', default='',
                        help='Server password')
    parser.add_argument('--channel', default=None,
                        help='Channel to join')
    parser.add_argument('--reference', default='reference.wav',
                        help='Reference audio file for voice cloning')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device (auto = detect best available)')
    parser.add_argument('--steps', type=int, default=4,
                        help='Number of inference steps (quality vs speed)')
    parser.add_argument('--no-asr', action='store_true',
                        help='Disable automatic speech recognition')
    parser.add_argument('--asr-threshold', type=int, default=2000,
                        help='RMS threshold for voice activity detection (default: 2000, use --debug-rms to tune)')
    parser.add_argument('--debug-rms', action='store_true',
                        help='Show real-time RMS levels to help tune --asr-threshold')
    parser.add_argument('--debug-audio', action='store_true',
                        help='Save audio files sent to Whisper for debugging')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device if args.device != 'auto' else get_best_device()
    
    bot = MumbleTTSBot(
        host=args.host,
        user=args.user,
        port=args.port,
        password=args.password,
        channel=args.channel,
        reference_audio=args.reference,
        device=device,
        num_steps=args.steps,
        enable_asr=not args.no_asr,
        asr_threshold=args.asr_threshold,
        debug_rms=args.debug_rms,
        debug_audio=args.debug_audio,
    )
    
    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
