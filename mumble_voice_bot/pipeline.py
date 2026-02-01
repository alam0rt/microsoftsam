"""Voice pipeline orchestration for Mumble Voice Bot.

The pipeline coordinates:
1. Speech-to-Text (Whisper) - transcribe incoming audio
2. LLM (Thinking) - generate response from transcription  
3. Text-to-Speech (LuxTTS) - synthesize audio response
"""

import asyncio
import time
from dataclasses import dataclass, field

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse


@dataclass
class PipelineConfig:
    """Configuration for the voice pipeline.
    
    Attributes:
        wake_word: Optional wake word to trigger processing.
        silence_threshold_ms: Milliseconds of silence before processing.
        max_recording_ms: Maximum recording duration.
        max_history_turns: Maximum conversation turns to keep in history.
        history_timeout: Seconds before clearing conversation history.
    """
    wake_word: str | None = None
    silence_threshold_ms: int = 1500
    max_recording_ms: int = 30000
    max_history_turns: int = 10  # Keep last N exchanges (user + assistant = 2 per exchange)
    history_timeout: float = 300.0  # 5 minutes


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription.
    
    Attributes:
        text: The transcribed text.
        language: Detected language (if available).
        duration: Audio duration in seconds.
    """
    text: str
    language: str | None = None
    duration: float = 0.0


@dataclass
class PipelineResult:
    """Result from processing through the full pipeline.
    
    Attributes:
        transcription: The speech-to-text result.
        llm_response: The LLM's text response.
        audio: The synthesized audio bytes (or None if streaming).
        latency: Processing time breakdown.
    """
    transcription: TranscriptionResult
    llm_response: LLMResponse
    audio: bytes | None = None
    latency: dict = field(default_factory=dict)


class VoicePipeline:
    """Orchestrates the voice processing pipeline.
    
    The pipeline takes audio input and produces audio output:
    
        Audio In -> Whisper STT -> LLM -> LuxTTS -> Audio Out
    
    It maintains conversation history per-user for contextual responses.
    
    Attributes:
        whisper: The Whisper transcriber callable.
        llm: The LLM provider for generating responses.
        luxtts: The LuxTTS instance for speech synthesis.
        config: Pipeline configuration.
    """
    
    def __init__(
        self,
        whisper,
        llm: LLMProvider,
        luxtts,
        config: PipelineConfig | None = None,
    ):
        """Initialize the voice pipeline.
        
        Args:
            whisper: Whisper transcriber callable (audio -> text).
            llm: LLM provider implementing the LLMProvider interface.
            luxtts: LuxTTS instance for speech synthesis.
            config: Optional pipeline configuration.
        """
        self.whisper = whisper
        self.llm = llm
        self.luxtts = luxtts
        self.config = config or PipelineConfig()
        
        # Per-user conversation history: user_id -> list of messages
        self._conversation_history: dict[str, list[dict]] = {}
        
        # Track last activity time per user for history timeout
        self._last_activity: dict[str, float] = {}
    
    def _should_respond(self, text: str) -> tuple[bool, str]:
        """Check if we should respond to this transcription.
        
        Args:
            text: The transcribed text.
            
        Returns:
            Tuple of (should_respond, cleaned_text).
        """
        if not text.strip():
            return False, ""
        
        cleaned_text = text.strip()
        
        # If wake word is configured, check for it
        if self.config.wake_word:
            wake_lower = self.config.wake_word.lower()
            text_lower = cleaned_text.lower()
            
            if wake_lower not in text_lower:
                return False, ""
            
            # Remove wake word from text
            import re
            cleaned_text = re.sub(
                re.escape(wake_lower), 
                "", 
                cleaned_text, 
                flags=re.IGNORECASE
            ).strip()
        
        return True, cleaned_text
    
    def _get_history(self, user_id: str) -> list[dict]:
        """Get conversation history for a user, clearing if timed out.
        
        Args:
            user_id: Unique identifier for the user.
            
        Returns:
            List of message dicts for the conversation.
        """
        current_time = time.time()
        
        # Check for timeout
        if user_id in self._last_activity:
            elapsed = current_time - self._last_activity[user_id]
            if elapsed > self.config.history_timeout:
                # Clear stale history
                self._conversation_history.pop(user_id, None)
        
        # Update activity time
        self._last_activity[user_id] = current_time
        
        # Get or create history
        if user_id not in self._conversation_history:
            self._conversation_history[user_id] = []
        
        return self._conversation_history[user_id]
    
    def _add_to_history(self, user_id: str, role: str, content: str) -> None:
        """Add a message to conversation history.
        
        Args:
            user_id: Unique identifier for the user.
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        history = self._get_history(user_id)
        history.append({"role": role, "content": content})
        
        # Trim to max history
        max_messages = self.config.max_history_turns * 2  # 2 messages per turn
        if len(history) > max_messages:
            # Keep the most recent messages
            self._conversation_history[user_id] = history[-max_messages:]
    
    def clear_history(self, user_id: str | None = None) -> None:
        """Clear conversation history.
        
        Args:
            user_id: User to clear history for. If None, clears all history.
        """
        if user_id is None:
            self._conversation_history.clear()
            self._last_activity.clear()
        else:
            self._conversation_history.pop(user_id, None)
            self._last_activity.pop(user_id, None)
    
    async def transcribe(self, audio, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio to text using Whisper.
        
        Args:
            audio: Audio data (numpy array or bytes).
            sample_rate: Audio sample rate.
            
        Returns:
            TranscriptionResult with the transcribed text.
        """
        start_time = time.time()
        
        # Call the whisper transcriber
        # Note: The transcriber is synchronous, so we run it in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.whisper, audio)
        
        elapsed = time.time() - start_time
        
        text = result.get("text", "").strip()
        language = result.get("language")
        
        return TranscriptionResult(
            text=text,
            language=language,
            duration=elapsed,
        )
    
    async def generate_response(
        self, 
        text: str, 
        user_id: str = "default"
    ) -> LLMResponse:
        """Generate an LLM response for the given text.
        
        Args:
            text: The user's input text.
            user_id: Unique identifier for conversation history.
            
        Returns:
            LLMResponse with the generated text.
        """
        # Add user message to history
        self._add_to_history(user_id, "user", text)
        
        # Get full history for context
        history = self._get_history(user_id)
        
        # Get LLM response
        response = await self.llm.chat(history)
        
        # Add assistant response to history
        self._add_to_history(user_id, "assistant", response.content)
        
        return response
    
    async def synthesize(
        self, 
        text: str, 
        voice_prompt: dict,
        num_steps: int = 4,
    ):
        """Synthesize speech from text.
        
        Args:
            text: The text to synthesize.
            voice_prompt: The encoded voice prompt for TTS.
            num_steps: Number of diffusion steps.
            
        Returns:
            Audio tensor from LuxTTS.
        """
        loop = asyncio.get_event_loop()
        
        # Run TTS in a thread since it's CPU/GPU bound
        audio = await loop.run_in_executor(
            None,
            lambda: self.luxtts.generate_speech(
                text, 
                voice_prompt, 
                num_steps=num_steps
            )
        )
        
        return audio
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_prompt: dict,
        num_steps: int = 4,
    ):
        """Synthesize speech from text in streaming fashion.
        
        Yields audio chunks as they are generated for lower latency.
        
        Args:
            text: The text to synthesize.
            voice_prompt: The encoded voice prompt for TTS.
            num_steps: Number of diffusion steps.
            
        Yields:
            Audio tensor chunks from LuxTTS.
        """
        loop = asyncio.get_event_loop()
        
        # Check if streaming method exists
        if hasattr(self.luxtts, 'generate_speech_streaming'):
            # Use a queue to bridge sync generator to async
            import queue
            audio_queue = queue.Queue()
            done_event = asyncio.Event()
            
            def generate_in_thread():
                try:
                    for chunk in self.luxtts.generate_speech_streaming(
                        text, voice_prompt, num_steps=num_steps
                    ):
                        audio_queue.put(chunk)
                finally:
                    audio_queue.put(None)  # Signal completion
            
            # Start generation in thread
            future = loop.run_in_executor(None, generate_in_thread)
            
            # Yield chunks as they become available
            while True:
                try:
                    chunk = await loop.run_in_executor(
                        None, 
                        lambda: audio_queue.get(timeout=0.1)
                    )
                    if chunk is None:
                        break
                    yield chunk
                except:
                    # Timeout - check if still running
                    if future.done():
                        # Drain any remaining items
                        while not audio_queue.empty():
                            chunk = audio_queue.get_nowait()
                            if chunk is not None:
                                yield chunk
                        break
        else:
            # Fall back to non-streaming
            audio = await self.synthesize(text, voice_prompt, num_steps)
            yield audio
    
    async def process_audio(
        self,
        audio,
        sample_rate: int,
        user_id: str = "default",
        voice_prompt: dict = None,
        num_steps: int = 4,
    ) -> PipelineResult | None:
        """Process audio through the full pipeline.
        
        This is the main entry point: audio -> transcription -> LLM -> speech.
        
        Args:
            audio: Input audio data.
            sample_rate: Audio sample rate.
            user_id: Unique identifier for conversation history.
            voice_prompt: Voice prompt for TTS (uses default if None).
            num_steps: Number of TTS diffusion steps.
            
        Returns:
            PipelineResult with transcription, response, and audio.
            Returns None if the input should not be responded to.
        """
        latency = {}
        
        # Step 1: Transcribe
        start = time.time()
        transcription = await self.transcribe(audio, sample_rate)
        latency["transcription"] = time.time() - start
        
        # Check if we should respond
        should_respond, cleaned_text = self._should_respond(transcription.text)
        if not should_respond:
            return None
        
        # Update transcription with cleaned text
        transcription.text = cleaned_text
        
        # Step 2: Get LLM response
        start = time.time()
        llm_response = await self.generate_response(cleaned_text, user_id)
        latency["llm"] = time.time() - start
        
        # Step 3: Synthesize speech
        if voice_prompt is not None:
            start = time.time()
            audio_out = await self.synthesize(
                llm_response.content, 
                voice_prompt, 
                num_steps
            )
            latency["tts"] = time.time() - start
        else:
            audio_out = None
        
        latency["total"] = sum(latency.values())
        
        return PipelineResult(
            transcription=transcription,
            llm_response=llm_response,
            audio=audio_out,
            latency=latency,
        )
