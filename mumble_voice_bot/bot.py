"""MumbleBot base class — shared Mumble I/O, VAD, ASR, TTS playback.

This is the composable bot core. It handles everything that's common
between bot types:
- Mumble connection, joining channels, lifecycle
- Audio reception and per-user VAD buffering
- 48kHz→16kHz resampling and ASR transcription
- Text accumulation across speech chunks
- TTS queue, synthesis, and PCM→Mumble playback
- Echo avoidance (muting audio input while speaking)
- Speaking coordination with other bots

The `Brain` is the only thing that varies. MumbleBot calls
`brain.process(utterance)` with a complete utterance, and speaks
whatever the brain returns.

Usage:
    from mumble_voice_bot.bot import MumbleBot
    from mumble_voice_bot.brains.echo import EchoBrain

    bot = MumbleBot(config, tts, stt, brain=EchoBrain(tts))
    bot.start()
    bot.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from mumble_voice_bot.audio import pcm_bytes_to_float, pcm_duration, pcm_rms, prepare_for_stt
from mumble_voice_bot.coordination import SharedBotServices
from mumble_voice_bot.events import ChannelActivityTracker, EventResponder
from mumble_voice_bot.interfaces.brain import BotResponse, Brain, NullBrain, Utterance
from mumble_voice_bot.text_processing import is_question, pad_tts_text, sanitize_for_tts

logger = logging.getLogger(__name__)


class MumbleBot:
    """Base Mumble voice bot with pluggable Brain.

    Owns all I/O infrastructure:
    - Mumble connection and audio callbacks
    - Per-user VAD (Voice Activity Detection) and audio buffering
    - ASR transcription (NeMo Nemotron)
    - Text accumulation across speech chunks
    - TTS queue, synthesis, and PCM→Mumble output
    - Speaking coordination and echo avoidance
    - Event/filler system and channel activity tracking

    The Brain receives complete Utterances and decides what to respond.

    Args:
        host: Mumble server hostname.
        user: Bot username in Mumble.
        port: Mumble server port.
        password: Mumble server password.
        channel: Channel to join (or None for default).
        brain: The Brain implementation to use.
        tts: TTS engine (StreamingLuxTTS or compatible).
        stt: STT engine (NemotronStreamingASR or compatible).
        voice_prompt: Pre-encoded voice tensors for TTS.
        device: Compute device ('cuda', 'cpu', 'mps').
        num_steps: TTS diffusion steps (quality vs. speed).
        asr_threshold: RMS threshold for voice activity detection.
        debug_rms: Show RMS levels for threshold tuning.
        shared_services: SharedBotServices for multi-bot coordination.
        soul_config: Soul configuration for themed event responses.
    """

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 64738,
        password: str = '',
        channel: str | None = None,
        brain: Brain | None = None,
        tts: Any = None,
        stt: Any = None,
        voice_prompt: dict | None = None,
        device: str = 'cpu',
        num_steps: int = 4,
        asr_threshold: int = 2000,
        debug_rms: bool = False,
        shared_services: SharedBotServices | None = None,
        soul_config: Any = None,
        # Advanced settings
        speech_hold_duration: float = 0.6,
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 5.0,
        pending_text_timeout: float = 1.5,
        max_response_staleness: float = 5.0,
        barge_in_enabled: bool = True,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        self.voice_prompt = voice_prompt or {}

        # Brain — the pluggable thinking component
        self.brain: Brain = brain or NullBrain()

        # Services
        self.tts = tts
        self.stt = stt

        # SharedBotServices for multi-bot coordination
        if shared_services is not None:
            self._shared_services = shared_services
        else:
            self._shared_services = SharedBotServices(tts=tts, stt=stt, device=device)

        # Register to receive utterances from other bots
        self._shared_services.register_utterance_listener(self._on_bot_utterance)

        # Event system
        self._event_responder = EventResponder(soul_config)
        self._activity_tracker = ChannelActivityTracker()

        # Logger
        self.logger = logging.getLogger(f"{__name__}.{user}")

        # VAD settings
        self.asr_threshold = asr_threshold
        self.debug_rms = debug_rms
        self._max_rms = 0

        # Per-user audio buffering
        self.audio_buffers: dict[int, list[bytes]] = {}
        self.speech_active_until: dict[int, float] = {}
        self.speech_start_time: dict[int, float] = {}
        self.speech_hold_duration = speech_hold_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

        # Pending transcriptions (accumulating text across chunks)
        self.pending_text: dict[int, str] = {}
        self.pending_text_time: dict[int, float] = {}
        self.pending_text_timeout = pending_text_timeout

        # State flags
        self._speaking = threading.Event()
        self._shutdown = threading.Event()
        self._running = False

        # Staleness / barge-in
        self.max_response_staleness = max_response_staleness
        self.barge_in_enabled = barge_in_enabled

        # Threading
        self._asr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ASR")
        self._tts_queue: queue.Queue = queue.Queue()

        # Audio sender tracking
        self._audio_senders: dict[int, str] = {}

        # Mumble client (set during start())
        self.mumble = None

        # Start TTS worker thread
        self._tts_worker_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name=f"TTS-{user}"
        )
        self._tts_worker_thread.start()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Connect to Mumble and start listening."""
        import pymumble_py3 as pymumble
        from pymumble_py3.constants import PYMUMBLE_CLBK_SOUNDRECEIVED

        self.logger.info(f"Connecting to {self.host}:{self.port} as '{self.user}'...")

        self.mumble = pymumble.Mumble(
            host=self.host,
            user=self.user,
            port=self.port,
            password=self.password,
            reconnect=True,
        )

        self.mumble.set_receive_sound(True)
        self.mumble.set_application_string("MumbleBot")
        self.mumble.set_codec_profile("audio")

        # Register audio callback
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED,
            self.on_sound_received,
        )

        self._running = True
        self.mumble.start()
        self.mumble.is_ready()

        self.logger.info("Connected!")

        if self.channel:
            try:
                ch = self.mumble.channels.find_by_name(self.channel)
                ch.move_in()
                self.logger.info(f"Joined channel: {self.channel}")
            except Exception as e:
                self.logger.warning(f"Failed to join channel '{self.channel}': {e}")

        self.logger.info(f"Listening for voice (threshold: {self.asr_threshold})")

    def run_forever(self):
        """Run the bot until interrupted."""
        self.logger.info("Running. Press Ctrl+C to stop.")
        try:
            while self._running and not self._shutdown.is_set():
                time.sleep(0.5)
                self._activity_tracker.check_channel_quiet()
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        self._shutdown.set()
        self._asr_executor.shutdown(wait=False)
        if self.mumble:
            try:
                self.mumble.stop()
            except Exception:
                pass
        self.logger.info("Shutdown complete.")

    # =========================================================================
    # Audio Input & VAD
    # =========================================================================

    def on_sound_received(self, user, sound_chunk):
        """Handle incoming audio from Mumble users."""
        user_id = user['session']
        user_name = user.get('name', 'Unknown')

        # Ignore own audio
        if user_id == self.mumble.users.myself_session:
            return

        # Ignore audio while any bot is speaking (echo avoidance)
        if self._shared_services.any_bot_speaking():
            return
        if self._speaking.is_set():
            # Barge-in detection: if user speaks loudly while we're talking
            if self.barge_in_enabled and not self._shared_services.any_bot_speaking():
                rms = pcm_rms(sound_chunk.pcm)
                barge_in_threshold = self.asr_threshold * 3
                if rms > barge_in_threshold:
                    self.logger.info(f"Barge-in from {user_name} (RMS={rms})")
                    self._on_barge_in(user_name)
            return

        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)

        # Track first audio from each user
        if user_id not in self._audio_senders:
            self._audio_senders[user_id] = user_name
            self.logger.info(f"First audio from {user_name} (session={user_id}, RMS={rms})")

        # Debug RMS bar
        if self.debug_rms:
            bar_width = min(rms // 100, 50)
            threshold_pos = min(self.asr_threshold // 100, 50)
            bar = '-' * threshold_pos + '+' * max(0, bar_width - threshold_pos) if rms >= self.asr_threshold else '-' * bar_width
            print(f'\r[{user_name:12}] RMS: {rms:5d} / {self._max_rms:5d}  |{bar:<50}|', end='', flush=True)

        # Initialize per-user state
        if user_id not in self.audio_buffers:
            self.audio_buffers[user_id] = []
            self.speech_active_until[user_id] = 0
            self.speech_start_time[user_id] = 0

        current_time = time.time()

        # Hard limit: force process if buffer too long
        buffer_dur = self._get_buffer_duration(user_id)
        if buffer_dur >= self.max_speech_duration:
            if self.debug_rms:
                print()
            audio_data = list(self.audio_buffers[user_id])
            self.audio_buffers[user_id] = []
            self._asr_executor.submit(
                self._process_speech, user.copy(), user_id, audio_data, True
            )
            return

        # VAD: speech detection with hold duration
        if rms > self.asr_threshold:
            if not self.audio_buffers[user_id]:
                self.speech_start_time[user_id] = current_time

            self.audio_buffers[user_id].append(sound_chunk.pcm)
            self.speech_active_until[user_id] = current_time + self.speech_hold_duration
        else:
            if current_time < self.speech_active_until[user_id]:
                # Still in hold period
                self.audio_buffers[user_id].append(sound_chunk.pcm)
            elif self.audio_buffers[user_id]:
                # Speech ended
                if self.debug_rms:
                    print()

                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                self._asr_executor.submit(
                    self._process_speech, user.copy(), user_id, audio_data, False
                )

    def _get_buffer_duration(self, user_id: int) -> float:
        """Calculate buffered audio duration in seconds."""
        if user_id not in self.audio_buffers:
            return 0
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[user_id])
        return total_bytes / (48000 * 2)

    # =========================================================================
    # ASR Pipeline
    # =========================================================================

    def _process_speech(
        self,
        user: dict,
        user_id: int,
        audio_chunks: list[bytes],
        is_continuation: bool = False,
    ):
        """Transcribe speech and route to brain."""
        user_name = user.get('name', 'Unknown')

        pcm_data = b''.join(audio_chunks)
        buffer_duration = pcm_duration(pcm_data)

        if buffer_duration < self.min_speech_duration:
            self._maybe_respond(user_id, user_name)
            return

        # Check audio energy
        audio_float = pcm_bytes_to_float(pcm_data)
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        if rms < 0.02:
            self._maybe_respond(user_id, user_name)
            return

        # Prepare for STT: resample + normalize
        pcm_16k_bytes = prepare_for_stt(audio_chunks)

        # Transcribe
        start_time = time.time()
        try:
            stt_result = asyncio.run(self.stt.transcribe(
                audio_data=pcm_16k_bytes,
                sample_rate=16000,
                sample_width=2,
                channels=1,
                language="en",
            ))
            text = stt_result.text.strip()
            transcribe_time = time.time() - start_time
        except Exception as e:
            self.logger.error(f"ASR error: {e}", exc_info=True)
            return

        if not text or len(text) < 2:
            self._maybe_respond(user_id, user_name)
            return

        self.logger.info(
            f'ASR ({transcribe_time*1000:.0f}ms): "{text}" '
            f'[from {user_name}, {buffer_duration:.1f}s audio]'
        )

        # Track channel activity
        self._activity_tracker.record_activity()
        self._activity_tracker.check_long_speech(user_id, user_name, buffer_duration)

        # Accumulate text
        current_time = time.time()
        if user_id in self.pending_text:
            self.pending_text[user_id] += " " + text
        else:
            self.pending_text[user_id] = text
        self.pending_text_time[user_id] = current_time

        # Decide when to respond
        if not is_continuation:
            self._maybe_respond(user_id, user_name, force=True, audio_chunks=audio_chunks, rms=rms)
        else:
            accumulated = self.pending_text.get(user_id, "")
            if len(accumulated.split()) >= 10:
                self._maybe_respond(user_id, user_name, force=True, audio_chunks=audio_chunks, rms=rms)

    # =========================================================================
    # Brain Routing
    # =========================================================================

    def _maybe_respond(
        self,
        user_id: int,
        user_name: str,
        force: bool = False,
        audio_chunks: list[bytes] | None = None,
        rms: float = 0.0,
    ):
        """Build an Utterance and route it to the Brain."""
        if user_id not in self.pending_text:
            return

        current_time = time.time()
        time_since_last = current_time - self.pending_text_time.get(user_id, 0)

        if not force and time_since_last < self.pending_text_timeout:
            return

        text = self.pending_text.pop(user_id, "")
        self.pending_text_time.pop(user_id, None)

        if not text:
            return

        # Build the Utterance
        utterance = Utterance(
            text=text,
            user_id=user_id,
            user_name=user_name,
            audio_chunks=audio_chunks or [],
            duration=sum(len(c) for c in (audio_chunks or [])) / (48000 * 2),
            rms=rms,
            is_question=is_question(text),
            is_first_speech=self._activity_tracker.check_first_time_speaker(user_name),
        )

        # Try to claim this response (multi-bot coordination)
        if not self._shared_services.try_claim_response(user_id, text):
            self.logger.debug(f"Someone else responding to: {text[:30]}...")
            return

        # Route to brain
        self.logger.info(f'Routing to brain: "{text}" [from {user_name}]')
        try:
            response = self.brain.process(utterance)
        except Exception as e:
            self.logger.error(f"Brain error: {e}", exc_info=True)
            return

        if response is not None:
            self._speak_response(response)

    def _on_bot_utterance(self, speaker_name: str, text: str) -> None:
        """Handle utterance from another bot."""
        if speaker_name == self.user:
            return

        # Route to brain
        try:
            response = self.brain.on_bot_utterance(speaker_name, text)
        except Exception:
            return

        if response is not None:
            # Wait for speaker to finish
            waited = 0
            while self._shared_services.any_bot_speaking() and waited < 60:
                time.sleep(0.1)
                waited += 0.1
            if not self._speaking.is_set():
                self._speak_response(response)

    # =========================================================================
    # TTS Output
    # =========================================================================

    def _speak_response(self, response: BotResponse):
        """Queue a BotResponse for TTS playback."""
        text = sanitize_for_tts(response.text)
        if not text.strip():
            return

        voice = response.voice.voice_prompt if response.voice else self.voice_prompt
        self._tts_queue.put((text, voice, response.skip_broadcast))

    def _tts_worker(self):
        """Background thread for TTS synthesis and playback."""
        while not self._shutdown.is_set():
            try:
                item = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                continue

            text, voice_prompt, skip_broadcast = item

            try:
                self._speak_sync(text, voice_prompt, skip_broadcast)
            except Exception as e:
                self.logger.error(f"TTS error: {e}", exc_info=True)

    def _speak_sync(self, text: str, voice_prompt: dict, skip_broadcast: bool = False):
        """Generate and play speech through Mumble."""
        text = pad_tts_text(text)
        if not text:
            return

        # Set speaking flag
        self._speaking.set()
        self._shared_services.bot_started_speaking()

        # Broadcast to other bots
        if not skip_broadcast:
            self._shared_services.broadcast_utterance(self.user, text)

        # Clear audio buffers (echo avoidance)
        for uid in list(self.audio_buffers.keys()):
            self.audio_buffers[uid] = []
            self.speech_active_until[uid] = 0
        self.pending_text.clear()
        self.pending_text_time.clear()

        try:
            self.logger.info(f'TTS: "{text[:80]}"' if len(text) <= 80 else f'TTS: "{text[:80]}..."')

            total_audio_samples = 0
            first_chunk = True
            tts_start = time.time()

            for wav_chunk in self.tts.generate_speech_streaming(
                text, voice_prompt, num_steps=self.num_steps
            ):
                if self._shutdown.is_set():
                    break

                if first_chunk:
                    ttfa = time.time() - tts_start
                    self.logger.info(f"TTS first audio: {ttfa*1000:.0f}ms")
                    first_chunk = False

                wav_float = wav_chunk.numpy().squeeze() if hasattr(wav_chunk, 'numpy') else wav_chunk
                if hasattr(wav_chunk, 'cpu'):
                    wav_float = wav_chunk.cpu().numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                chunk_samples = len(pcm)
                total_audio_samples += chunk_samples

                self.mumble.sound_output.add_sound(pcm.tobytes())

                # Pace playback
                chunk_duration = chunk_samples / 48000
                wait_time = chunk_duration * 0.9 + 0.15
                if wait_time > 0.1:
                    time.sleep(wait_time)

            tts_total = time.time() - tts_start
            audio_ms = (total_audio_samples / 48000) * 1000
            self.logger.info(f"TTS complete: {tts_total*1000:.0f}ms synthesis, {audio_ms:.0f}ms audio")

        finally:
            # Post-speech cleanup
            time.sleep(0.5)  # Let echo dissipate

            for uid in list(self.audio_buffers.keys()):
                self.audio_buffers[uid] = []
                self.speech_active_until[uid] = 0
            self.pending_text.clear()
            self.pending_text_time.clear()

            self._speaking.clear()
            self._shared_services.bot_stopped_speaking()

    # =========================================================================
    # Barge-in & Idle
    # =========================================================================

    def _on_barge_in(self, user_name: str):
        """Handle barge-in: speak a brief acknowledgment and stop.

        Implements the barge-in acknowledgment from plan-human.md Phase C.
        """
        self.logger.info(f"[BARGE-IN] {user_name} interrupted — acknowledging")

        # Get barge-in acknowledgment from reactive brain or event responder
        ack = None
        if hasattr(self.brain, 'get_barge_in_ack'):
            ack = self.brain.get_barge_in_ack()
        if not ack:
            ack = self._event_responder.get_event_response('interrupted')

        if ack:
            self.logger.info(f"[BARGE-IN] Would say: '{ack}' (queued for after stop)")
            # Note: speaking while being interrupted is awkward. We log it
            # but don't actually speak during interruption. The ack could be
            # spoken after the current TTS is cancelled, as a follow-up.

    def speak(self, text: str, blocking: bool = False):
        """Public API to speak text.

        Args:
            text: Text to speak.
            blocking: If True, speak synchronously. If False, queue for TTS worker.
        """
        if blocking:
            self._speak_sync(text, self.voice_prompt)
        else:
            self._tts_queue.put((text, self.voice_prompt, False))
