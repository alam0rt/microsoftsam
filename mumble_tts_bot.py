#!/usr/bin/env python3
"""
Mumble TTS Bot - Reads text messages aloud using LuxTTS voice cloning.

Usage:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav
    
    # With ASR (transcribe voice to text):
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav --asr
    
    # Debug mode to tune VAD threshold:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav --asr --debug-rms

Commands (send as text messages in Mumble):
    @mimic          - Start mimicking the sender's voice (repeats back what they say)
    @stop           - Stop mimicking
    @save <name>    - Save the last mimic'd voice (or current voice) with a name
    @voice <name>   - Switch to a previously saved voice
    @voices         - List all available saved voices
    @clone <url>    - Clone voice from a URL (wav/mp3) and use it
"""
import argparse
import os
import re
import sys
import tempfile
import time
import threading
import queue
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List

# Add vendor paths for pymumble and LuxTTS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

import numpy as np
import torch
from scipy import signal

import pymumble_py3 as pymumble
from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, PYMUMBLE_CLBK_SOUNDRECEIVED

from zipvoice.luxvoice import LuxTTS


# =============================================================================
# StreamingLuxTTS - Subclass that adds streaming and fixes upstream issues
# =============================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for streaming TTS.
    
    Splits on common sentence-ending punctuation to allow audio playback
    to start as soon as the first sentence is generated.
    """
    # Split on sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'(?<=[.!?;:,])\s+', text.strip())
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


class StreamingLuxTTS(LuxTTS):
    """
    Extended LuxTTS with streaming support and bug fixes.
    
    This subclass avoids modifying the upstream LuxTTS vendor code by:
    1. Monkey-patching the Whisper transcriber to force English (prevents 蚊蚊蚊 hallucinations)
    2. Fixing MPS device handling (MPS doesn't use device index like CUDA)
    3. Adding streaming generation for faster perceived response times
    """
    
    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        # Call parent constructor
        super().__init__(model_path=model_path, device=device, threads=threads)
        
        # Fix 1: Patch transcriber to force English language
        # Without this, Whisper can hallucinate Chinese/other characters from noise
        self._patch_transcriber_for_english()
        
        # Fix 2: MPS device handling is already done in parent, but verify
        # (upstream has a bug with torch.device(device, 0) for MPS)
        
    def _patch_transcriber_for_english(self):
        """Wrap the transcriber to always use English language detection.
        
        This prevents Whisper from hallucinating non-English text (like 蚊蚊蚊蚊)
        when processing noise or unclear audio.
        """
        original_transcriber = self.transcriber
        
        def english_transcriber(audio, **kwargs):
            # Force English language and transcription task
            # This is passed as generate_kwargs to the HuggingFace pipeline
            result = original_transcriber(
                audio,
                generate_kwargs={"language": "en", "task": "transcribe"},
                **kwargs
            )
            return result
        
        self.transcriber = english_transcriber
        print("[StreamingLuxTTS] Patched transcriber for English-only mode")
    
    def generate_speech_streaming(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        """
        Stream speech generation by splitting text into sentences and yielding audio chunks.
        
        This allows playback to start as soon as the first sentence is generated,
        significantly reducing perceived latency for longer texts - making the bot
        feel more human-like and responsive.
        
        Args:
            text: The text to synthesize
            encode_dict: The encoded voice prompt from encode_prompt()
            num_steps: Number of diffusion steps (fewer = faster but lower quality)
            guidance_scale: Classifier-free guidance scale
            t_shift: Temperature-like parameter for sampling
            speed: Speech speed multiplier
            return_smooth: If True, return 24kHz audio; if False, return 48kHz
            
        Yields:
            torch.Tensor: Audio chunks as they are generated (48kHz by default)
        """
        # Split text into sentences for streaming
        sentences = split_into_sentences(text)
        
        # If text is very short or has no sentence breaks, just yield the whole thing
        if len(sentences) <= 1:
            wav = self.generate_speech(
                text, encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth
            )
            yield wav
            return
        
        # Generate and yield each sentence's audio as soon as it's ready
        for sentence in sentences:
            if not sentence.strip():
                continue
            wav = self.generate_speech(
                sentence, encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth
            )
            yield wav


# =============================================================================
# Helper functions
# =============================================================================


# =============================================================================
# Helper functions
# =============================================================================

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
        voices_dir: str = 'voices',
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        self.enable_asr = enable_asr
        
        # Voice management
        self.voices_dir = voices_dir
        self.current_voice_name = 'default'  # Track which voice is active
        self.last_mimic_voice = None  # Store the last mimic'd voice prompt for @save
        
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
        self.speech_start_time = {}  # user_id -> timestamp when speech started (for latency tracking)
        self.speech_hold_duration = 1.2  # seconds to wait after speech stops before final transcription
        self.mimic_hold_duration = 0.6  # shorter hold for mimic mode - respond quickly like a child
        self.min_speech_duration = 0.8  # minimum seconds of audio to transcribe (shorter = more hallucinations)
        self.max_speech_duration = 30.0  # maximum seconds to buffer before forced transcription
        self.incremental_interval = 4.0  # transcribe every N seconds during ongoing speech
        
        # Mimic mode: capture user's voice and speak it back to them (like an annoying child)
        self.mimic_pending = {}  # user_id -> True if actively mimicking this user
        
        # Interrupt feature: stop TTS when user speaks
        self._interrupted = threading.Event()  # Signal to stop current TTS
        self._interrupted_text = None  # Store interrupted text for potential resume
        self._speaking = threading.Event()  # Track if bot is currently speaking
        
        # Threading for non-blocking ASR and TTS
        # Use separate thread pools so ASR doesn't block TTS and vice versa
        self._asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ASR")
        self._tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self._tts_queue = queue.Queue()  # Queue of (text, voice_prompt) to speak
        self._tts_lock = threading.Lock()  # Protect voice_prompt access
        self._shutdown = threading.Event()  # Signal for clean shutdown
        
        # Start TTS worker thread
        self._tts_worker_thread = threading.Thread(target=self._tts_worker, daemon=True, name="TTS-Worker")
        self._tts_worker_thread.start()
        
        # Initialize TTS with our streaming-capable subclass
        print(f"Loading StreamingLuxTTS model on {device}...")
        self.tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)
        
        # Ensure voices directory exists
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # Check if we should load from a saved voice or encode fresh
        reference_name = os.path.splitext(os.path.basename(reference_audio))[0]
        saved_voice_path = os.path.join(self.voices_dir, f"{reference_name}.pt")
        
        if os.path.exists(saved_voice_path):
            print(f"Loading saved voice from: {saved_voice_path}")
            self.voice_prompt = torch.load(saved_voice_path, weights_only=False, map_location=self.device)
            # Ensure tensors are on correct device
            self.voice_prompt = self._ensure_voice_on_device(self.voice_prompt)
            self.current_voice_name = reference_name
        else:
            print(f"Encoding reference audio: {reference_audio}")
            self.voice_prompt = self.tts.encode_prompt(reference_audio, rms=0.01)
            self.current_voice_name = reference_name
            # Auto-save the reference voice for faster startup next time
            self._save_voice(reference_name, self.voice_prompt)
            print(f"Saved voice as '{reference_name}' for faster loading next time")
        
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
        
        # Check for @save <name> command - save current or last mimic'd voice
        if text.strip().lower().startswith('@save '):
            voice_name = text.strip()[6:].strip()
            if voice_name:
                self._handle_save_command(voice_name, sender)
            else:
                print(f"[Voice] No name provided for @save")
            return
        
        # Check for @voice <name> command - switch to a saved voice
        if text.strip().lower().startswith('@voice '):
            voice_name = text.strip()[7:].strip()
            if voice_name:
                self._handle_voice_command(voice_name)
            else:
                print(f"[Voice] No name provided for @voice")
            return
        
        # Check for @voices command - list available voices
        if text.strip().lower() == '@voices':
            self._handle_voices_command()
            return
        
        # Check for @clone <url> command - clone voice from URL
        if text.strip().lower().startswith('@clone '):
            url = text.strip()[7:].strip()
            if url:
                self._handle_clone_command(url)
            else:
                print(f"[Clone] No URL provided for @clone")
            return
        
        # Generate and play speech
        try:
            self.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
    
    def _save_voice(self, name: str, voice_prompt: dict) -> bool:
        """Save a voice prompt to disk.
        
        Args:
            name: Name for the voice (will be sanitized)
            voice_prompt: The encoded voice prompt dict from encode_prompt()
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Sanitize name - only allow alphanumeric, dash, underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        if not safe_name:
            safe_name = 'unnamed'
        
        voice_path = os.path.join(self.voices_dir, f"{safe_name}.pt")
        try:
            torch.save(voice_prompt, voice_path)
            print(f"[Voice] Saved voice '{safe_name}' to {voice_path}")
            return True
        except Exception as e:
            print(f"[Voice] Failed to save voice: {e}")
            return False
    
    def _load_voice(self, name: str) -> dict | None:
        """Load a voice prompt from disk.
        
        Args:
            name: Name of the voice to load
            
        Returns:
            The voice prompt dict, or None if not found
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        voice_path = os.path.join(self.voices_dir, f"{safe_name}.pt")
        
        if not os.path.exists(voice_path):
            print(f"[Voice] Voice '{safe_name}' not found at {voice_path}")
            return None
        
        try:
            # Load with map_location to handle CPU/GPU transfers
            voice_prompt = torch.load(voice_path, weights_only=False, map_location=self.device)
            
            # Ensure all tensors are on the correct device
            voice_prompt = self._ensure_voice_on_device(voice_prompt)
            
            print(f"[Voice] Loaded voice '{safe_name}' from {voice_path}")
            return voice_prompt
        except Exception as e:
            print(f"[Voice] Failed to load voice: {e}")
            return None
    
    def _ensure_voice_on_device(self, voice_prompt: dict) -> dict:
        """Ensure all tensors in voice_prompt are on the correct device."""
        result = {}
        for key, value in voice_prompt.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def _validate_voice_prompt(self, voice_prompt: dict) -> bool:
        """Validate a voice prompt to catch corrupted or malformed data.
        
        Returns True if valid, False if there's a problem.
        """
        required_keys = ['prompt_tokens', 'prompt_features_lens', 'prompt_features', 'prompt_rms']
        
        for key in required_keys:
            if key not in voice_prompt:
                print(f"[Voice] Missing required key: {key}")
                return False
        
        # Check prompt_features dimensions - should be (1, seq_len, feature_dim)
        # feature_dim is typically 100 for vocos features
        pf = voice_prompt['prompt_features']
        if isinstance(pf, torch.Tensor):
            if len(pf.shape) != 3:
                print(f"[Voice] Invalid prompt_features shape: {pf.shape} (expected 3D)")
                return False
            batch, seq_len, feat_dim = pf.shape
            if batch != 1:
                print(f"[Voice] Invalid batch size: {batch} (expected 1)")
                return False
            if feat_dim != 100:
                print(f"[Voice] Invalid feature dim: {feat_dim} (expected 100)")
                return False
            if seq_len > 5000:  # ~50 seconds of audio at 100 frames/sec
                print(f"[Voice] Suspiciously long sequence: {seq_len} (max expected ~5000)")
                return False
            # Check for NaN/Inf
            if torch.isnan(pf).any() or torch.isinf(pf).any():
                print(f"[Voice] prompt_features contains NaN or Inf values")
                return False
        
        return True
    
    def _list_voices(self) -> List[str]:
        """List all available saved voices.
        
        Returns:
            List of voice names (without .pt extension)
        """
        voices = []
        if os.path.exists(self.voices_dir):
            for f in os.listdir(self.voices_dir):
                if f.endswith('.pt'):
                    voices.append(f[:-3])  # Remove .pt extension
        return sorted(voices)
    
    def _handle_save_command(self, name: str, sender: str):
        """Handle @save <name> command.
        
        Saves either the last mimic'd voice or the current active voice.
        """
        if self.last_mimic_voice is not None:
            # Save the last mimic'd voice
            if self._save_voice(name, self.last_mimic_voice):
                print(f"[Voice] {sender} saved last mimic'd voice as '{name}'")
        else:
            # Save the current active voice
            if self._save_voice(name, self.voice_prompt):
                print(f"[Voice] {sender} saved current voice as '{name}'")
    
    def _handle_voice_command(self, name: str):
        """Handle @voice <name> command - switch to a saved voice."""
        voice_prompt = self._load_voice(name)
        if voice_prompt is not None:
            with self._tts_lock:
                self.voice_prompt = voice_prompt
                self.current_voice_name = name
            print(f"[Voice] Switched to voice '{name}'")
            # Speak a confirmation (blocking so it uses the new voice)
            self.speak(f"Voice switched to {name}", blocking=True)
        else:
            available = self._list_voices()
            print(f"[Voice] Available voices: {', '.join(available) if available else 'none'}")
    
    def _handle_voices_command(self):
        """Handle @voices command - list and speak available voices."""
        voices = self._list_voices()
        if voices:
            print(f"[Voice] Available voices: {', '.join(voices)}")
            print(f"[Voice] Current voice: {self.current_voice_name}")
            # Speak the list
            voice_list = ', '.join(voices)
            self.speak(f"Available voices: {voice_list}. Current voice is {self.current_voice_name}.")
        else:
            print(f"[Voice] No saved voices found in {self.voices_dir}")
            self.speak("No saved voices found.")

    def _handle_clone_command(self, url: str):
        """Handle @clone <url> command - download audio and use as voice reference.
        
        Args:
            url: URL to a wav or mp3 audio file
        """
        print(f"[Clone] Downloading audio from: {url}")
        
        # Determine file extension from URL
        url_lower = url.lower()
        if '.wav' in url_lower:
            suffix = '.wav'
        elif '.mp3' in url_lower:
            suffix = '.mp3'
        else:
            # Default to wav, let soundfile figure it out
            suffix = '.wav'
        
        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                
                # Set up request with user agent (some servers reject bare requests)
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'MumbleTTSBot/1.0'
                })
                
                # Create SSL context that doesn't verify certificates
                # (many audio hosting sites have certificate issues)
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
                
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as response:
                    tmp.write(response.read())
            
            print(f"[Clone] Downloaded to {tmp_path}")
            
            # Load and encode the voice
            import soundfile as sf
            audio, sr = sf.read(tmp_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed (LuxTTS expects 16kHz)
            if sr != 16000:
                from scipy import signal as scipy_signal
                num_samples = int(len(audio) * 16000 / sr)
                audio = scipy_signal.resample(audio, num_samples)
                sr = 16000
            
            # Ensure float32
            audio = audio.astype(np.float32)
            
            # Need at least 2.5s for voice cloning
            duration = len(audio) / sr
            if duration < 2.5:
                print(f"[Clone] Audio too short ({duration:.1f}s < 2.5s required)")
                self.speak("Audio too short. Need at least 2.5 seconds.")
                return
            
            print(f"[Clone] Encoding voice from {duration:.1f}s of audio...")
            
            # Encode the voice prompt
            voice_prompt = self.tts.encode_prompt(audio, sr, duration)
            
            # Validate the voice prompt
            if not self._validate_voice_prompt(voice_prompt):
                print(f"[Clone] Voice encoding failed validation")
                self.speak("Failed to encode voice from that audio.")
                return
            
            # Update the current voice
            with self._tts_lock:
                self.voice_prompt = voice_prompt
                self.current_voice_name = "cloned"
            
            print(f"[Clone] Voice cloned successfully!")
            self.speak("Voice cloned successfully!", blocking=True)
            
        except urllib.error.URLError as e:
            print(f"[Clone] Download failed: {e}")
            self.speak("Failed to download audio.")
        except Exception as e:
            print(f"[Clone] Error: {e}")
            self.speak(f"Clone failed: {str(e)[:50]}")
        finally:
            # Clean up temp file
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass

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
            self.speech_start_time[user_id] = 0
        
        current_time = time.time()
        
        # Check if user is speaking while bot is speaking - trigger interrupt
        if rms > self.asr_threshold and self._speaking.is_set():
            if not self._interrupted.is_set():
                print(f"\n[Interrupt] {user_name} started speaking, interrupting TTS...")
                self._interrupted.set()
        
        # Check if this chunk contains speech (above threshold)
        if rms > self.asr_threshold:
            # Speech detected - add to buffer and extend hold time
            # Use shorter hold time in mimic mode for faster response
            hold_time = self.mimic_hold_duration if self.mimic_pending.get(user_id) else self.speech_hold_duration
            
            # Track when speech started (for latency measurement)
            if not self.audio_buffers[user_id]:
                self.speech_start_time[user_id] = current_time
            
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
                    print(f"[Mimic] Max buffer reached, processing async...")
                    # Copy buffer and clear it, then process async
                    audio_data = list(self.audio_buffers[user_id])
                    self.audio_buffers[user_id] = []
                    self._asr_executor.submit(self._mimic_user_async, user.copy(), user_id, audio_data)
                # Otherwise just accumulate
            # Incremental transcription: transcribe periodically during ongoing speech
            elif buffer_duration >= self.min_speech_duration and time_since_last >= self.incremental_interval:
                if self.debug_rms:
                    print()  # Newline after RMS display
                # Copy buffer for async transcription (don't clear - it's incremental)
                audio_data = list(self.audio_buffers[user_id])
                self.last_transcription_time[user_id] = current_time
                self._asr_executor.submit(self._transcribe_buffer_async, user.copy(), user_id, audio_data, True)
            
            # Check for max duration (forced transcription for very long speech)
            elif buffer_duration >= self.max_speech_duration:
                if self.debug_rms:
                    print()  # Newline after RMS display
                print(f"[ASR] Max duration reached for {user_name}, transcribing async...")
                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                self._asr_executor.submit(self._transcribe_buffer_async, user.copy(), user_id, audio_data, False)
        else:
            # Below threshold - but if we're in "hold" period, still collect audio
            # This captures trailing sounds and pauses between words
            if current_time < self.speech_active_until[user_id]:
                self.audio_buffers[user_id].append(sound_chunk.pcm)
            elif self.audio_buffers[user_id]:
                # Hold period expired and we have buffered audio - user stopped speaking
                if self.debug_rms:
                    print()  # Newline after RMS display
                
                # Copy buffer and clear it for async processing
                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                
                # Check if this user has mimic mode active
                if self.mimic_pending.get(user_id):
                    # Mimic mode: repeat what they said back to them (like an annoying child)
                    buffer_duration = len(b''.join(audio_data)) / (48000 * 2)
                    # Need at least 2.5s for decent voice cloning (LuxTTS needs enough audio
                    # for the convolution layers - shorter audio causes kernel size errors)
                    if buffer_duration >= 2.5:
                        self._asr_executor.submit(self._mimic_user_async, user.copy(), user_id, audio_data)
                    else:
                        # Too short - just discard and wait for next utterance
                        print(f"[Mimic] Audio too short ({buffer_duration:.1f}s < 2.5s), waiting for more...")
                else:
                    # Normal final transcription (async)
                    self._asr_executor.submit(self._transcribe_buffer_async, user.copy(), user_id, audio_data, False)
    
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
        
        return False
    
    def _transcribe_buffer_async(self, user: dict, user_id: int, audio_chunks: list, incremental: bool):
        """Async version of transcribe that takes audio data directly.
        
        This runs in the ASR thread pool so it doesn't block callbacks.
        """
        user_name = user.get('name', 'Unknown')
        
        # Concatenate all PCM chunks
        pcm_data = b''.join(audio_chunks)
        buffer_duration = len(pcm_data) / (48000 * 2)
        
        # Skip if too short (likely noise)
        if buffer_duration < self.min_speech_duration:
            if not incremental:
                print(f"[ASR] Skipping short audio from {user_name} ({buffer_duration:.2f}s < {self.min_speech_duration}s)")
            return
        
        # Convert PCM bytes to numpy array (16-bit signed, 48kHz mono)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 0.02:
            if not incremental:
                print(f"[ASR] Skipping quiet audio from {user_name} (RMS: {rms:.4f} < 0.02)")
            return
        
        # Resample from 48kHz to 16kHz for Whisper
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        
        # Normalize for Whisper
        target_rms = 0.1
        if rms_16k > 0.001:
            audio_16k = audio_16k * (target_rms / rms_16k)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
        
        marker = "..." if incremental else ""
        print(f"[ASR] Transcribing {buffer_duration:.1f}s from {user_name}{marker}")
        
        try:
            result = self.tts.transcriber(audio_16k)
            text = result.get('text', '').strip()
            
            if text and not self._is_hallucination(text):
                prefix = f"[ASR {user_name}]:" if not incremental else f"[ASR {user_name} ...]:"
                print(f"{prefix} {text}")
            elif not incremental:
                if text:
                    print(f"[ASR] Filtered hallucination from {user_name}: {text[:50]}...")
                else:
                    print(f"[ASR] No speech detected from {user_name}")
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
    
    def _mimic_user_async(self, user: dict, user_id: int, audio_chunks: list):
        """Async version of mimic that takes audio data directly.
        
        This runs in the ASR thread pool so it doesn't block callbacks.
        """
        user_name = user.get('name', 'Unknown')
        
        # Concatenate all PCM chunks
        pcm_data = b''.join(audio_chunks)
        buffer_duration = len(pcm_data) / (48000 * 2)
        
        # Clear CUDA cache before processing
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Cap at 25 seconds
        max_mimic_duration = 25.0
        min_mimic_duration = 2.5
        
        if buffer_duration < min_mimic_duration:
            print(f"[Mimic] Audio too short from {user_name} ({buffer_duration:.1f}s)")
            return
        
        if buffer_duration > max_mimic_duration:
            print(f"[Mimic] Trimming {buffer_duration:.1f}s to {max_mimic_duration}s")
            bytes_to_keep = int(max_mimic_duration * 48000 * 2)
            pcm_data = pcm_data[-bytes_to_keep:]
            buffer_duration = max_mimic_duration
        
        # Track latency
        processing_start = time.time()
        print(f"[Mimic] Processing {buffer_duration:.1f}s of audio from {user_name}...")
        
        # Convert PCM bytes to numpy array
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
        
        # Step 1: Transcribe
        transcribe_start = time.time()
        try:
            result = self.tts.transcriber(audio_16k_normalized)
            full_text = result.get('text', '').strip()
            transcribe_end = time.time()
            
            if not full_text or self._is_hallucination(full_text):
                print(f"[Mimic] Could not transcribe speech from {user_name}")
                return
            
            text = extract_last_sentence(full_text)
            print(f"[Mimic] {user_name} said: \"{full_text}\"")
            print(f"[Latency] Transcription: {transcribe_end - transcribe_start:.2f}s")
            if text != full_text:
                print(f"[Mimic] Using last sentence: \"{text}\"")
                
        except Exception as e:
            print(f"[Mimic] Transcription error: {e}")
            return
        
        # Step 2: Clone voice
        import tempfile
        import scipy.io.wavfile as wavfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                wavfile.write(temp_path, 48000, audio_int16)
            
            clone_start = time.time()
            ref_duration = min(buffer_duration, 10.0)
            print(f"[Mimic] Cloning {user_name}'s voice (using {ref_duration:.1f}s reference)...")
            user_voice_prompt = self.tts.encode_prompt(temp_path, duration=ref_duration, rms=0.01)
            clone_end = time.time()
            print(f"[Latency] Voice cloning: {clone_end - clone_start:.2f}s")
            
            # Validate voice prompt
            if not self._validate_voice_prompt(user_voice_prompt):
                print(f"[Mimic] Invalid voice prompt, skipping...")
                return
            
            # Step 3: Generate TTS (queue it for the TTS worker)
            tts_start = time.time()
            print(f"[Mimic] Speaking back: \"{text}\"")
            
            # Generate speech synchronously in this thread (we're already in a worker)
            wav = self.tts.generate_speech(text, user_voice_prompt, num_steps=self.num_steps)
            wav_float = wav.numpy().squeeze()
            tts_end = time.time()
            print(f"[Latency] TTS generation: {tts_end - tts_start:.2f}s")
            
            # Send audio
            wav_float = np.clip(wav_float, -1.0, 1.0)
            pcm_out = (wav_float * 32767).astype(np.int16)
            self.mumble.sound_output.add_sound(pcm_out.tobytes())
            
            # Latency summary
            total_processing = time.time() - processing_start
            print(f"[Latency] TOTAL: {total_processing:.2f}s processing")
            print(f"[Mimic] Done mimicking {user_name}!")
            
            # Store for @save
            self.last_mimic_voice = user_voice_prompt
            
        except Exception as e:
            print(f"[Mimic] Voice cloning/synthesis error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
            if self.device == 'cuda':
                torch.cuda.empty_cache()

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
        
        # Clear CUDA cache before processing to prevent memory fragmentation
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Cap at 25 seconds to avoid Whisper's 30s limit
        max_mimic_duration = 25.0
        min_mimic_duration = 2.5  # LuxTTS needs enough audio for convolution layers
        
        if buffer_duration < min_mimic_duration:
            print(f"[Mimic] Audio too short from {user_name} ({buffer_duration:.1f}s < {min_mimic_duration}s)")
            self.audio_buffers[user_id] = []
            return
        
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
        
        # Track latency from when speech started
        speech_start = self.speech_start_time.get(user_id, time.time())
        processing_start = time.time()
        wait_latency = processing_start - speech_start - buffer_duration  # Time spent waiting after speech ended
        
        print(f"[Mimic] Processing {buffer_duration:.1f}s of audio from {user_name}...")
        print(f"[Latency] Speech detected -> processing: {processing_start - speech_start:.2f}s (includes {buffer_duration:.1f}s of speech + {wait_latency:.2f}s hold)")
        
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
        transcribe_start = time.time()
        try:
            result = self.tts.transcriber(audio_16k_normalized)
            full_text = result.get('text', '').strip()
            transcribe_end = time.time()
            
            if not full_text or self._is_hallucination(full_text):
                print(f"[Mimic] Could not transcribe speech from {user_name}")
                return
            
            # Extract just the last sentence for cleaner TTS
            text = extract_last_sentence(full_text)
            
            print(f"[Mimic] {user_name} said: \"{full_text}\"")
            print(f"[Latency] Transcription: {transcribe_end - transcribe_start:.2f}s")
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
            # Pass the actual duration so LuxTTS doesn't try to load more than we have
            # LuxTTS internally loads at 24kHz, so we need at least 2s for the conv layers
            clone_start = time.time()
            ref_duration = min(buffer_duration, 10.0)  # Cap at 10s for efficiency
            print(f"[Mimic] Cloning {user_name}'s voice (using {ref_duration:.1f}s reference)...")
            user_voice_prompt = self.tts.encode_prompt(temp_path, duration=ref_duration, rms=0.01)
            clone_end = time.time()
            print(f"[Latency] Voice cloning: {clone_end - clone_start:.2f}s")
            
            # Validate the voice prompt to catch corrupted tensors early
            if not self._validate_voice_prompt(user_voice_prompt):
                print(f"[Mimic] Invalid voice prompt generated, skipping...")
                return
            
            # Step 3: Generate speech with their voice saying their words
            tts_start = time.time()
            print(f"[Mimic] Speaking back: \"{text}\"")
            wav = self.tts.generate_speech(text, user_voice_prompt, num_steps=self.num_steps)
            wav_float = wav.numpy().squeeze()
            tts_end = time.time()
            print(f"[Latency] TTS generation: {tts_end - tts_start:.2f}s")
            
            # Convert and send
            wav_float = np.clip(wav_float, -1.0, 1.0)
            pcm_out = (wav_float * 32767).astype(np.int16)
            self.mumble.sound_output.add_sound(pcm_out.tobytes())
            
            # Total latency summary
            total_processing = time.time() - processing_start
            total_from_speech = time.time() - speech_start
            print(f"[Latency] TOTAL: {total_from_speech:.2f}s from speech start, {total_processing:.2f}s processing")
            print(f"[Latency] Breakdown: transcribe={transcribe_end - transcribe_start:.2f}s, clone={clone_end - clone_start:.2f}s, tts={tts_end - tts_start:.2f}s")
            print(f"[Mimic] Done mimicking {user_name}!")
            
            # Store the mimic'd voice so it can be saved with @save
            self.last_mimic_voice = user_voice_prompt
            
            # Keep mimic mode active - don't reset mimic_pending
            
        except Exception as e:
            print(f"[Mimic] Voice cloning/synthesis error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Clear CUDA cache after processing to free memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def _tts_worker(self):
        """Background worker that processes TTS requests from the queue.
        
        This runs in a dedicated thread so TTS generation doesn't block
        the main thread or ASR processing.
        """
        print("[TTS Worker] Started")
        while not self._shutdown.is_set():
            try:
                # Wait for work with timeout so we can check shutdown
                try:
                    text, voice_prompt = self._tts_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Generate and play speech
                try:
                    self._speak_sync(text, voice_prompt)
                except Exception as e:
                    print(f"[TTS Worker] Error: {e}")
                finally:
                    self._tts_queue.task_done()
                    
            except Exception as e:
                print(f"[TTS Worker] Unexpected error: {e}")
        
        print("[TTS Worker] Stopped")
    
    def _speak_sync(self, text: str, voice_prompt: dict, streaming: bool = True):
        """Synchronously generate and play speech (runs in worker thread).
        
        Args:
            text: The text to speak
            voice_prompt: The voice prompt to use
            streaming: If True, use streaming generation
        """
        # Mark that we're speaking (for interrupt detection)
        self._speaking.set()
        self._interrupted.clear()
        
        try:
            if streaming:
                # Use streaming for faster perceived response time
                for wav_chunk in self.tts.generate_speech_streaming(
                    text, voice_prompt, num_steps=self.num_steps
                ):
                    # Check for interrupt between chunks
                    if self._interrupted.is_set():
                        print(f"[Interrupt] Stopping TTS playback")
                        # Clear the sound output buffer
                        self.mumble.sound_output.clear_buffer()
                        break
                    
                    wav_float = wav_chunk.numpy().squeeze()
                    wav_float = np.clip(wav_float, -1.0, 1.0)
                    pcm = (wav_float * 32767).astype(np.int16)
                    self.mumble.sound_output.add_sound(pcm.tobytes())
            else:
                # Non-streaming: generate all audio at once
                wav = self.tts.generate_speech(text, voice_prompt, num_steps=self.num_steps)
                
                # Can still be interrupted before playback starts
                if self._interrupted.is_set():
                    print(f"[Interrupt] Stopping TTS before playback")
                    return
                
                wav_float = wav.numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                self.mumble.sound_output.add_sound(pcm.tobytes())
        finally:
            # Mark that we're done speaking
            self._speaking.clear()

    def speak(self, text: str, voice_prompt: dict = None, blocking: bool = False):
        """Queue text to be spoken (non-blocking by default).
        
        Args:
            text: The text to speak
            voice_prompt: Voice to use (defaults to self.voice_prompt)
            blocking: If True, wait for speech to complete
        """
        if voice_prompt is None:
            with self._tts_lock:
                voice_prompt = self.voice_prompt
        
        if blocking:
            # Synchronous - used for confirmations like @voice switching
            self._speak_sync(text, voice_prompt)
        else:
            # Queue for async processing
            self._tts_queue.put((text, voice_prompt))
        
    def run_forever(self):
        """Keep the bot running."""
        print("Bot is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._shutdown.set()
            # Wait for TTS queue to drain
            self._tts_queue.join()
            self._asr_executor.shutdown(wait=True)
            self._tts_executor.shutdown(wait=True)


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
    parser.add_argument('--voices-dir', default='voices',
                        help='Directory to store/load saved voice prompts (default: voices/)')
    
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
        voices_dir=args.voices_dir,
    )
    
    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
