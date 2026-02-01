# Modular Mumble Voice Bot Architecture

## Overview

Transform the Mumble TTS bot into a modular voice assistant with pluggable components for:
1. **Speech-to-Text (STT)** - Whisper via OpenAI-compatible API
2. **Language Model (LLM)** - Any OpenAI-compatible chat endpoint
3. **Text-to-Speech (TTS)** - Qwen3-TTS via vLLM-Omni OpenAI-compatible API

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Mumble    │────▶│   Whisper   │────▶│     LLM     │────▶│  Qwen3-TTS  │
│   Audio     │     │    (STT)    │     │  (Thinking) │     │   (Voice)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      ▲                                                            │
      └────────────────────────────────────────────────────────────┘
                              Audio Response
```

## Architecture

### Component Interfaces

All components communicate via OpenAI-compatible HTTP APIs, making them easily swappable.

#### 1. STT Interface (Whisper)
```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

file: <audio_file>
model: "whisper-1"  # or specific model name
language: "en"      # optional
```

**Compatible Backends:**
- OpenAI Whisper API
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server)
- [whisper.cpp server](https://github.com/ggerganov/whisper.cpp)
- vLLM with Whisper support
- LocalAI

#### 2. LLM Interface (Chat Completions)
```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<transcribed text>"}
  ]
}
```

**Compatible Backends:**
- OpenAI API
- vLLM
- Ollama (with OpenAI compatibility)
- llama.cpp server
- LocalAI
- LiteLLM (proxy to any backend)

#### 3. TTS Interface (Qwen3-TTS via vLLM-Omni)
```
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "<text to speak>",
  "voice": "Vivian",
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "task_type": "CustomVoice",
  "language": "Auto",
  "instructions": "<emotional/tonal guidance from LLM>"
}
```

**Compatible Backends:**
- vLLM-Omni (Qwen3-TTS)
- OpenAI TTS API
- qwen-tts demo server
- Coqui TTS
- Piper TTS

---

## Configuration

### Environment Variables / Config File

```yaml
# config.yaml
stt:
  endpoint: "http://localhost:8001/v1/audio/transcriptions"
  model: "whisper-large-v3"
  api_key: "${STT_API_KEY}"  # optional
  language: "en"  # or "auto"

llm:
  endpoint: "http://localhost:8002/v1/chat/completions"
  model: "Qwen/Qwen3-32B"
  api_key: "${LLM_API_KEY}"
  system_prompt: |
    You are a helpful voice assistant in a Mumble voice chat.
    Keep responses concise and conversational.
    
    When responding, also suggest a tone/emotion for the TTS.
    Format your response as:
    [tone: <emotion>]
    <your response text>
    
    Example tones: cheerful, serious, excited, calm, sympathetic

tts:
  endpoint: "http://localhost:8003/v1/audio/speech"
  model: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
  api_key: "${TTS_API_KEY}"
  voice: "Vivian"
  task_type: "CustomVoice"  # or "VoiceDesign", "Base"
  # For voice cloning (task_type: Base)
  ref_audio: null  # path or URL to reference audio
  ref_text: null   # transcript of reference audio

mumble:
  host: "localhost"
  port: 64738
  user: "VoiceBot"
  password: null
  channel: null
  certfile: null
  keyfile: null

bot:
  wake_word: null  # e.g., "hey bot" - if null, responds to all speech
  silence_threshold_ms: 1500  # silence before processing speech
  max_recording_ms: 30000     # max speech duration
  response_prefix: null       # e.g., "Here's what I think:"
```

---

## Implementation Plan

### Phase 1: Refactor Core Architecture

#### 1.1 Create Abstract Interfaces

```python
# interfaces/stt.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    confidence: float | None

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio to text."""
        pass

# interfaces/llm.py
@dataclass
class LLMResponse:
    content: str
    tone: str | None  # extracted tone hint for TTS

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], context: dict | None = None) -> LLMResponse:
        """Generate response from conversation."""
        pass

# interfaces/tts.py
@dataclass
class SpeechResult:
    audio: bytes
    sample_rate: int
    format: str  # "wav", "mp3", etc.

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None, 
                         instructions: str | None = None) -> SpeechResult:
        """Convert text to speech."""
        pass
```

#### 1.2 Implement OpenAI-Compatible Providers

```python
# providers/openai_stt.py
class OpenAIWhisperSTT(STTProvider):
    def __init__(self, endpoint: str, model: str, api_key: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
    
    async def transcribe(self, audio: bytes, sample_rate: int) -> TranscriptionResult:
        # POST to /v1/audio/transcriptions
        ...

# providers/openai_llm.py
class OpenAIChatLLM(LLMProvider):
    def __init__(self, endpoint: str, model: str, api_key: str | None = None,
                 system_prompt: str | None = None):
        ...
    
    async def chat(self, messages: list[dict], context: dict | None = None) -> LLMResponse:
        # POST to /v1/chat/completions
        # Parse response for [tone: ...] hints
        ...

# providers/openai_tts.py
class OpenAITTS(TTSProvider):
    """Works with OpenAI TTS API and vLLM-Omni Qwen3-TTS."""
    
    def __init__(self, endpoint: str, model: str, api_key: str | None = None,
                 voice: str = "Vivian", task_type: str = "CustomVoice"):
        ...
    
    async def synthesize(self, text: str, voice: str | None = None,
                         instructions: str | None = None) -> SpeechResult:
        # POST to /v1/audio/speech
        # Include instructions for Qwen3-TTS emotional control
        ...
```

#### 1.3 Create Voice Pipeline

```python
# pipeline.py
class VoicePipeline:
    def __init__(self, stt: STTProvider, llm: LLMProvider, tts: TTSProvider,
                 config: PipelineConfig):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.config = config
        self.conversation_history: list[dict] = []
    
    async def process_audio(self, audio: bytes, sample_rate: int) -> SpeechResult:
        """Full pipeline: audio -> transcription -> LLM -> speech."""
        
        # 1. Transcribe
        transcription = await self.stt.transcribe(audio, sample_rate)
        
        # 2. Check wake word (if configured)
        if self.config.wake_word and not self._has_wake_word(transcription.text):
            return None
        
        # 3. Build messages with history
        self.conversation_history.append({
            "role": "user",
            "content": transcription.text
        })
        
        # 4. Get LLM response
        llm_response = await self.llm.chat(self.conversation_history)
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": llm_response.content
        })
        
        # 5. Synthesize speech with tone guidance
        speech = await self.tts.synthesize(
            text=llm_response.content,
            instructions=llm_response.tone  # Pass tone to Qwen3-TTS
        )
        
        return speech
```

### Phase 2: Mumble Integration

#### 2.1 Refactor MumbleBot

```python
# mumble_bot.py
class MumbleVoiceBot:
    def __init__(self, mumble_config: MumbleConfig, pipeline: VoicePipeline):
        self.config = mumble_config
        self.pipeline = pipeline
        self.mumble = None
        self.audio_buffer = AudioBuffer()
    
    async def connect(self):
        """Connect to Mumble server."""
        ...
    
    async def on_audio_received(self, user: User, audio: bytes):
        """Handle incoming audio from users."""
        self.audio_buffer.append(user.name, audio)
        
        # Check for silence / end of speech
        if self.audio_buffer.silence_detected(user.name):
            complete_audio = self.audio_buffer.get_and_clear(user.name)
            
            # Process through pipeline
            response = await self.pipeline.process_audio(
                complete_audio, 
                sample_rate=48000
            )
            
            if response:
                await self.play_audio(response.audio, response.sample_rate)
    
    async def play_audio(self, audio: bytes, sample_rate: int):
        """Play audio to Mumble channel."""
        ...
```

### Phase 3: Service Orchestration

#### 3.1 Docker Compose Setup

```yaml
# docker-compose.yaml
version: '3.8'

services:
  whisper:
    image: fedirz/faster-whisper-server:latest
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  llm:
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen3-8B --port 8000
    ports:
      - "8002:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  tts:
    build:
      context: .
      dockerfile: Dockerfile.tts
    command: >
      vllm-omni serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
      --host 0.0.0.0 --port 8000 --omni
    ports:
      - "8003:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  bot:
    build: .
    depends_on:
      - whisper
      - llm
      - tts
    environment:
      - STT_ENDPOINT=http://whisper:8000
      - LLM_ENDPOINT=http://llm:8000
      - TTS_ENDPOINT=http://tts:8000
    volumes:
      - ./config.yaml:/app/config.yaml
```

#### 3.2 Nix Flake Updates

Add shell options for running individual services:

```nix
# flake.nix additions
devShells.whisper = mkShell { ... };  # faster-whisper-server
devShells.llm = mkShell { ... };      # vLLM for LLM
devShells.tts = mkShell { ... };      # vLLM-Omni for TTS
```

### Phase 4: Advanced Features

#### 4.1 Streaming Support

- Stream LLM responses token-by-token
- Start TTS as soon as first sentence is complete
- Reduce perceived latency

#### 4.2 Voice Cloning Mode

Use Qwen3-TTS Base model to clone a user's voice:
1. Record reference audio from user
2. Store voice profile
3. Use cloned voice for responses to that user

#### 4.3 Multi-User Support

- Track conversation history per user
- Different voice/persona per user
- Concurrent processing

#### 4.4 Interrupt Handling

- Detect when user starts speaking during bot response
- Stop TTS playback
- Process new input

---

## File Structure

```
mumble_voice_bot/
├── __init__.py
├── main.py                 # Entry point
├── config.py               # Configuration loading
├── pipeline.py             # VoicePipeline orchestration
├── interfaces/
│   ├── __init__.py
│   ├── stt.py              # STT abstract interface
│   ├── llm.py              # LLM abstract interface
│   └── tts.py              # TTS abstract interface
├── providers/
│   ├── __init__.py
│   ├── openai_stt.py       # OpenAI-compatible Whisper
│   ├── openai_llm.py       # OpenAI-compatible Chat
│   └── openai_tts.py       # OpenAI-compatible TTS (vLLM-Omni)
├── mumble/
│   ├── __init__.py
│   ├── bot.py              # MumbleVoiceBot
│   ├── audio.py            # Audio buffer/processing
│   └── connection.py       # Mumble connection handling
└── utils/
    ├── __init__.py
    ├── audio.py            # Audio format conversion
    └── tone_parser.py      # Extract tone hints from LLM response
```

---

## API Examples

### Whisper Transcription
```bash
curl -X POST http://localhost:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3"
```

### LLM Chat
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Include [tone: emotion] in responses."},
      {"role": "user", "content": "Tell me a joke"}
    ]
  }'
```

### Qwen3-TTS Speech
```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Why did the chicken cross the road? To get to the other side!",
    "voice": "Vivian",
    "instructions": "cheerful and playful",
    "task_type": "CustomVoice"
  }' --output response.wav
```

---

## Migration Path

1. **Keep existing bot functional** during refactor
2. **Extract interfaces** from current implementation
3. **Implement OpenAI providers** alongside existing code
4. **Add configuration system**
5. **Switch to new pipeline** via feature flag
6. **Remove legacy code** once stable

---

## Dependencies

```toml
[project]
dependencies = [
    "pymumble>=1.0",
    "httpx>=0.27",           # Async HTTP client
    "pydantic>=2.0",         # Config validation
    "pyyaml>=6.0",           # Config files
    "numpy>=1.26",
    "soundfile>=0.12",
    "librosa>=0.10",         # Audio processing
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

---

## Success Metrics

- [ ] Transcription latency < 500ms for 5s audio
- [ ] LLM response latency < 2s
- [ ] TTS synthesis latency < 1s
- [ ] End-to-end latency < 4s
- [ ] Support for 3+ concurrent users
- [ ] Zero-downtime component swapping
