#!/usr/bin/env python3
"""
Mumble Voice Bot - LLM-powered voice assistant for Mumble.

Listens to voice in a Mumble channel, transcribes with Whisper, generates 
responses with an LLM, and speaks back using LuxTTS voice cloning.

Usage:
    # Basic usage with default Ollama endpoint
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav
    
    # With custom LLM endpoint
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav \\
        --llm-endpoint http://localhost:8000/v1/chat/completions \\
        --llm-model Qwen/Qwen3-32B
    
    # Debug mode to tune VAD threshold
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav --debug-rms
"""
import argparse
import asyncio
import os
import random
import re
import sys
import time
import threading
import queue
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

# Import LLM components
try:
    from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
    from mumble_voice_bot.config import load_config
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[Warning] LLM modules not available. Install with: pip install httpx pyyaml")


# =============================================================================
# StreamingLuxTTS - Subclass that adds streaming and fixes upstream issues
# =============================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for streaming TTS."""
    sentences = re.split(r'(?<=[.!?;:,])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class StreamingLuxTTS(LuxTTS):
    """Extended LuxTTS with streaming support and bug fixes."""
    
    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        super().__init__(model_path=model_path, device=device, threads=threads)
        self._patch_transcriber_for_english()
        
    def _patch_transcriber_for_english(self):
        """Force English language detection to prevent hallucinations."""
        original_transcriber = self.transcriber
        
        def english_transcriber(audio, **kwargs):
            result = original_transcriber(
                audio,
                generate_kwargs={"language": "en", "task": "transcribe"},
                **kwargs
            )
            return result
        
        self.transcriber = english_transcriber
        print("[TTS] Patched transcriber for English-only mode")
    
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
        """Stream speech generation by splitting text into sentences."""
        sentences = split_into_sentences(text)
        
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

def pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate RMS of 16-bit PCM audio."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(audio) == 0:
        return 0
    return int(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def get_best_device() -> str:
    """Auto-detect the best available compute device."""
    try:
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
        return 'cpu'


# =============================================================================
# MumbleVoiceBot - Main bot class
# =============================================================================

class MumbleVoiceBot:
    """A Mumble bot that listens, thinks with an LLM, and responds with TTS."""
    
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
        asr_threshold: int = 2000,
        debug_rms: bool = False,
        voices_dir: str = 'voices',
        # LLM configuration
        llm_endpoint: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        llm_system_prompt: str = None,
        personality: str = None,
        config_file: str = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        self.voices_dir = voices_dir
        
        # VAD settings
        self.asr_threshold = asr_threshold
        self.debug_rms = debug_rms
        self._max_rms = 0
        
        # ASR state per user
        self.audio_buffers = {}  # user_id -> list of PCM bytes
        self.speech_active_until = {}  # user_id -> timestamp
        self.speech_start_time = {}  # user_id -> timestamp
        self.speech_hold_duration = 1.5  # seconds to wait after speech stops
        self.min_speech_duration = 0.8  # minimum seconds to transcribe
        self.max_speech_duration = 30.0  # max seconds before forced processing
        
        # Conversation state per user
        self.conversation_history = {}  # user_id -> list of messages
        self.conversation_timeout = 300.0  # 5 minutes
        self.last_conversation_time = {}  # user_id -> timestamp
        
        # State flags
        self._speaking = threading.Event()
        self._shutdown = threading.Event()
        
        # Threading
        self._asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ASR")
        self._tts_queue = queue.Queue()
        self._tts_lock = threading.Lock()
        
        # Start TTS worker
        self._tts_worker_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTS-Worker"
        )
        self._tts_worker_thread.start()
        
        # Initialize TTS
        print(f"[TTS] Loading model on {device}...")
        self.tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)
        
        # Load voice
        os.makedirs(self.voices_dir, exist_ok=True)
        self._load_reference_voice(reference_audio)
        
        # Initialize LLM
        self.llm = None
        if LLM_AVAILABLE:
            self._init_llm(
                endpoint=llm_endpoint,
                model=llm_model,
                api_key=llm_api_key,
                system_prompt=llm_system_prompt,
                personality=personality,
                config_file=config_file,
            )
        else:
            print("[Warning] LLM not available - bot will only transcribe, not respond")
        
        # Initialize Mumble
        print(f"[Mumble] Connecting to {host}:{port} as '{user}'...")
        self.mumble = pymumble.Mumble(
            host=host,
            user=user,
            port=port,
            password=password,
            reconnect=True,
        )
        
        # Enable audio reception
        self.mumble.set_receive_sound(True)
        print(f"[VAD] Listening for voice (threshold: {asr_threshold})")
        if self.debug_rms:
            print("[VAD] Debug mode: showing audio levels")
        
        # Set up callbacks
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            self.on_message
        )
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED,
            self.on_sound_received
        )
    
    def _load_reference_voice(self, reference_audio: str):
        """Load or encode the reference voice for TTS."""
        reference_name = os.path.splitext(os.path.basename(reference_audio))[0]
        saved_voice_path = os.path.join(self.voices_dir, f"{reference_name}.pt")
        
        if os.path.exists(saved_voice_path):
            print(f"[Voice] Loading cached voice: {saved_voice_path}")
            self.voice_prompt = torch.load(
                saved_voice_path, weights_only=False, map_location=self.device
            )
            self.voice_prompt = self._ensure_voice_on_device(self.voice_prompt)
        else:
            print(f"[Voice] Encoding reference: {reference_audio}")
            self.voice_prompt = self.tts.encode_prompt(reference_audio, rms=0.01)
            # Cache for next time
            torch.save(self.voice_prompt, saved_voice_path)
            print(f"[Voice] Cached as '{reference_name}' for faster startup")
    
    def _ensure_voice_on_device(self, voice_prompt: dict) -> dict:
        """Ensure all tensors in voice_prompt are on the correct device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in voice_prompt.items()
        }
    
    def _init_llm(
        self,
        endpoint: str = None,
        model: str = None,
        api_key: str = None,
        system_prompt: str = None,
        personality: str = None,
        config_file: str = None,
    ):
        """Initialize the LLM provider."""
        # Try config file
        config = None
        if config_file or os.path.exists("config.yaml"):
            try:
                config = load_config(config_file)
                print(f"[LLM] Loaded config from {config_file or 'config.yaml'}")
            except Exception as e:
                print(f"[LLM] Config load failed: {e}")
        
        # CLI args override config
        final_endpoint = endpoint
        final_model = model
        final_api_key = api_key
        final_system_prompt = system_prompt
        
        if config:
            final_endpoint = final_endpoint or config.llm.endpoint
            final_model = final_model or config.llm.model
            final_api_key = final_api_key or config.llm.api_key
            final_system_prompt = final_system_prompt or config.llm.system_prompt
            if hasattr(config, 'bot') and config.bot.conversation_timeout:
                self.conversation_timeout = config.bot.conversation_timeout
        
        # Check environment variables for API key if not set
        if not final_api_key:
            final_api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')
        
        # Defaults
        final_endpoint = final_endpoint or "http://localhost:11434/v1/chat/completions"
        final_model = final_model or "llama3.2:3b"
        final_system_prompt = final_system_prompt or self._load_system_prompt(personality=personality)
        
        self.llm = OpenAIChatLLM(
            endpoint=final_endpoint,
            model=final_model,
            api_key=final_api_key,
            system_prompt=final_system_prompt,
            timeout=30.0,
        )
        print(f"[LLM] Initialized: {final_model} @ {final_endpoint}")
    
    def _load_system_prompt(self, prompt_file: str = None, personality: str = None) -> str:
        """Load system prompt from file, optionally combined with a personality."""
        base_prompt = None
        
        # Try specified file first
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                print(f"[LLM] Loaded prompt from {prompt_file}")
                base_prompt = f.read()
        
        # Try default locations
        if not base_prompt:
            default_paths = [
                os.path.join(_THIS_DIR, "prompts", "default.md"),
                os.path.join(_THIS_DIR, "prompts", "default.txt"),
                "prompts/default.md",
                "prompts/default.txt",
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        print(f"[LLM] Loaded prompt from {path}")
                        base_prompt = f.read()
                        break
        
        # Fallback to inline prompt
        if not base_prompt:
            print("[LLM] Using built-in default prompt")
            base_prompt = self._get_fallback_prompt()
        
        # Load personality if specified
        if personality:
            personality_prompt = self._load_personality(personality)
            if personality_prompt:
                base_prompt = base_prompt + "\n\n" + "=" * 40 + "\n\n" + personality_prompt
        
        return base_prompt
    
    def _load_personality(self, personality: str) -> str:
        """Load a personality file by name."""
        # Check if it's already a path
        if os.path.exists(personality):
            with open(personality, 'r') as f:
                print(f"[LLM] Loaded personality from {personality}")
                return f.read()
        
        # Try personalities directory
        personality_paths = [
            os.path.join(_THIS_DIR, "personalities", f"{personality}.md"),
            os.path.join(_THIS_DIR, "personalities", f"{personality}.txt"),
            os.path.join(_THIS_DIR, "personalities", personality),
            f"personalities/{personality}.md",
            f"personalities/{personality}.txt",
        ]
        
        for path in personality_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    print(f"[LLM] Loaded personality: {personality}")
                    return f.read()
        
        print(f"[LLM] Warning: Personality '{personality}' not found")
        return None
    
    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if no file is found."""
        return """You are a casual voice assistant in a Mumble voice channel.

Your responses will be spoken by TTS. Never use emojis, symbols, or formatting.
Keep responses to 1-2 sentences. Use casual language and contractions.
Sound like a friend chatting, not a corporate assistant.
Write numbers and symbols as words: "about 5 dollars" not "$5"."""
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def _get_history(self, user_id: int) -> list[dict]:
        """Get conversation history for a user, clearing if stale."""
        current_time = time.time()
        
        if user_id in self.last_conversation_time:
            elapsed = current_time - self.last_conversation_time[user_id]
            if elapsed > self.conversation_timeout:
                self.conversation_history.pop(user_id, None)
                print(f"[Chat] Cleared stale history for user {user_id}")
        
        self.last_conversation_time[user_id] = current_time
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        return self.conversation_history[user_id]
    
    def _add_to_history(self, user_id: int, role: str, content: str, user_name: str = None):
        """Add a message to conversation history."""
        history = self._get_history(user_id)
        
        # For user messages, optionally prefix with their name for context
        if role == "user" and user_name:
            # Store the name so LLM knows who's talking
            content = f"[{user_name} says]: {content}"
        
        history.append({"role": role, "content": content})
        
        # Keep last 20 messages
        if len(history) > 20:
            self.conversation_history[user_id] = history[-20:]
    
    async def _generate_response(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response."""
        self._add_to_history(user_id, "user", text, user_name)
        history = self._get_history(user_id)
        
        response = await self.llm.chat(history)
        
        self._add_to_history(user_id, "assistant", response.content)
        return response.content
    
    def _generate_response_sync(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self._generate_response(user_id, text, user_name), loop
                )
                return future.result(timeout=35.0)
            else:
                return loop.run_until_complete(self._generate_response(user_id, text, user_name))
        except RuntimeError:
            return asyncio.run(self._generate_response(user_id, text, user_name))
    
    # =========================================================================
    # Audio Processing
    # =========================================================================
    
    def on_sound_received(self, user, sound_chunk):
        """Handle incoming audio from users."""
        user_id = user['session']
        user_name = user.get('name', 'Unknown')
        
        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)
        
        # Debug display
        if self.debug_rms:
            bar_width = min(rms // 100, 50)
            threshold_pos = min(self.asr_threshold // 100, 50)
            bar = '-' * threshold_pos + '+' * max(0, bar_width - threshold_pos) if rms >= self.asr_threshold else '-' * bar_width
            print(f'\r[{user_name:12}] RMS: {rms:5d} / {self._max_rms:5d}  |{bar:<50}|', end='', flush=True)
        
        # Initialize state for new users
        if user_id not in self.audio_buffers:
            self.audio_buffers[user_id] = []
            self.speech_active_until[user_id] = 0
            self.speech_start_time[user_id] = 0
        
        current_time = time.time()
        
        # Speech detection
        if rms > self.asr_threshold:
            if not self.audio_buffers[user_id]:
                self.speech_start_time[user_id] = current_time
            
            self.audio_buffers[user_id].append(sound_chunk.pcm)
            self.speech_active_until[user_id] = current_time + self.speech_hold_duration
            
            # Check for max duration
            buffer_duration = self._get_buffer_duration(user_id)
            if buffer_duration >= self.max_speech_duration:
                if self.debug_rms:
                    print()
                print(f"[ASR] Max duration reached for {user_name}")
                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                self._asr_executor.submit(self._process_speech, user.copy(), user_id, audio_data)
        else:
            # Below threshold
            if current_time < self.speech_active_until[user_id]:
                self.audio_buffers[user_id].append(sound_chunk.pcm)
            elif self.audio_buffers[user_id]:
                # Speech ended
                if self.debug_rms:
                    print()
                
                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                self._asr_executor.submit(self._process_speech, user.copy(), user_id, audio_data)
    
    def _get_buffer_duration(self, user_id) -> float:
        """Calculate buffered audio duration in seconds."""
        if user_id not in self.audio_buffers:
            return 0
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[user_id])
        return total_bytes / (48000 * 2)  # 48kHz, 16-bit mono
    
    def _process_speech(self, user: dict, user_id: int, audio_chunks: list):
        """Process speech: transcribe -> LLM -> TTS."""
        user_name = user.get('name', 'Unknown')
        
        # Concatenate audio
        pcm_data = b''.join(audio_chunks)
        buffer_duration = len(pcm_data) / (48000 * 2)
        
        if buffer_duration < self.min_speech_duration:
            print(f"[ASR] Skipping short audio from {user_name} ({buffer_duration:.2f}s)")
            return
        
        # Convert and resample
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 0.02:
            print(f"[ASR] Skipping quiet audio from {user_name}")
            return
        
        # Resample 48kHz -> 16kHz
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)
        
        # Normalize for Whisper
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        if rms_16k > 0.001:
            audio_16k = audio_16k * (0.1 / rms_16k)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
        
        # Transcribe
        print(f"[ASR] Transcribing {buffer_duration:.1f}s from {user_name}...")
        start_time = time.time()
        
        try:
            result = self.tts.transcriber(audio_16k)
            text = result.get('text', '').strip()
            transcribe_time = time.time() - start_time
            
            if not text or len(text) < 2:
                print(f"[ASR] No speech detected from {user_name}")
                return
            
            print(f"[ASR] {user_name}: \"{text}\" ({transcribe_time:.2f}s)")
            
            # Generate LLM response if available
            if self.llm:
                # Small thinking delay
                think_delay = random.uniform(0.2, 0.5)
                time.sleep(think_delay)
                
                print(f"[LLM] Generating response...")
                llm_start = time.time()
                response = self._generate_response_sync(user_id, text, user_name)
                llm_time = time.time() - llm_start
                print(f"[LLM] Response: \"{response}\" ({llm_time:.2f}s)")
                
                # Queue TTS
                self._tts_queue.put((response, self.voice_prompt))
            
        except Exception as e:
            print(f"[Error] Processing failed: {e}")
    
    # =========================================================================
    # TTS
    # =========================================================================
    
    def _tts_worker(self):
        """Background worker for TTS."""
        print("[TTS] Worker started")
        while not self._shutdown.is_set():
            try:
                text, voice_prompt = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                self._speak_sync(text, voice_prompt)
            except Exception as e:
                print(f"[TTS] Error: {e}")
            finally:
                self._tts_queue.task_done()
        
        print("[TTS] Worker stopped")
    
    def _speak_sync(self, text: str, voice_prompt: dict):
        """Generate and play speech."""
        self._speaking.set()
        
        try:
            print(f"[TTS] Speaking: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            for wav_chunk in self.tts.generate_speech_streaming(
                text, voice_prompt, num_steps=self.num_steps
            ):
                wav_float = wav_chunk.numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                self.mumble.sound_output.add_sound(pcm.tobytes())
        finally:
            self._speaking.clear()
    
    def speak(self, text: str, blocking: bool = False):
        """Queue text to be spoken."""
        if blocking:
            self._speak_sync(text, self.voice_prompt)
        else:
            self._tts_queue.put((text, self.voice_prompt))
    
    # =========================================================================
    # Text Message Handling
    # =========================================================================
    
    def on_message(self, message):
        """Handle text messages (for TTS)."""
        text = strip_html(message.message)
        if not text.strip():
            return
        
        sender = "Someone"
        if hasattr(message, 'actor') and message.actor in self.mumble.users:
            sender = self.mumble.users[message.actor]['name']
        
        print(f"[Text] {sender}: {text}")
        
        # Speak text messages
        self.speak(text)
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self):
        """Start the bot."""
        self.mumble.start()
        self.mumble.is_ready()
        print("[Mumble] Connected!")
        
        if self.channel:
            try:
                channel = self.mumble.channels.find_by_name(self.channel)
                channel.move_in()
                print(f"[Mumble] Joined channel: {self.channel}")
            except Exception as e:
                print(f"[Mumble] Failed to join channel: {e}")
    
    def run_forever(self):
        """Keep the bot running."""
        print("[Bot] Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Bot] Shutting down...")
            self._shutdown.set()
            self._tts_queue.join()
            self._asr_executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(
        description='Mumble Voice Bot - LLM-powered voice assistant'
    )
    
    # Mumble settings
    parser.add_argument('--host', default='localhost', help='Mumble server')
    parser.add_argument('--port', type=int, default=64738, help='Mumble port')
    parser.add_argument('--user', default='VoiceBot', help='Bot username')
    parser.add_argument('--password', default='', help='Server password')
    parser.add_argument('--channel', default=None, help='Channel to join')
    
    # Voice settings
    parser.add_argument('--reference', default='reference.wav',
                        help='Reference audio for voice cloning')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device')
    parser.add_argument('--steps', type=int, default=4,
                        help='TTS quality (more steps = better quality, slower)')
    parser.add_argument('--voices-dir', default='voices',
                        help='Directory for cached voices')
    
    # VAD settings
    parser.add_argument('--asr-threshold', type=int, default=2000,
                        help='Voice activity threshold (use --debug-rms to tune)')
    parser.add_argument('--debug-rms', action='store_true',
                        help='Show RMS levels for threshold tuning')
    
    # LLM settings
    parser.add_argument('--llm-endpoint', default=None,
                        help='LLM API endpoint (default: Ollama localhost)')
    parser.add_argument('--llm-model', default=None,
                        help='LLM model name')
    parser.add_argument('--llm-api-key', default=None,
                        help='LLM API key (or use LLM_API_KEY env var)')
    parser.add_argument('--llm-system-prompt', default=None,
                        help='System prompt for the assistant')
    parser.add_argument('--personality', default=None,
                        help='Personality to use (e.g., "imperial", or path to file)')
    parser.add_argument('--config', default=None,
                        help='Path to config.yaml')
    
    args = parser.parse_args()
    
    device = args.device if args.device != 'auto' else get_best_device()
    
    bot = MumbleVoiceBot(
        host=args.host,
        user=args.user,
        port=args.port,
        password=args.password,
        channel=args.channel,
        reference_audio=args.reference,
        device=device,
        num_steps=args.steps,
        asr_threshold=args.asr_threshold,
        debug_rms=args.debug_rms,
        voices_dir=args.voices_dir,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_system_prompt=args.llm_system_prompt,
        personality=args.personality,
        config_file=args.config,
    )
    
    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
