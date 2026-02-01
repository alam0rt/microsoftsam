# Mumble Voice Bot - LLM Thinking Module Integration

## Overview

Add an LLM "Thinking" module to the existing Mumble TTS bot to enable conversational AI capabilities. The bot has:
- ✅ **Speech-to-Text (STT)** - Whisper (built-in)
- ✅ **Text-to-Speech (TTS)** - LuxTTS (built-in, 150x realtime, voice cloning)
- ✅ **Language Model (LLM)** - Any OpenAI-compatible chat endpoint

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Mumble    │────▶│   Whisper   │────▶│     LLM     │────▶│   LuxTTS    │
│   Audio     │     │  (built-in) │     │  (Thinking) │     │ (built-in)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      ▲                                                            │
      └────────────────────────────────────────────────────────────┘
                              Audio Response
```

## Usage

```bash
# Basic usage with Ollama (default)
python mumble_tts_bot.py --host mumble.example.com --reference voice.wav

# With custom LLM endpoint (vLLM, OpenAI, etc.)
python mumble_tts_bot.py --host mumble.example.com --reference voice.wav \
    --llm-endpoint http://localhost:8000/v1/chat/completions \
    --llm-model Qwen/Qwen3-32B

# Debug VAD threshold
python mumble_tts_bot.py --host mumble.example.com --reference voice.wav --debug-rms
```

## Components

### Whisper (STT) - Built-in
- Local Whisper model for speech-to-text
- Processes incoming Mumble audio
- English-only mode to prevent hallucinations

### LuxTTS (TTS) - Built-in
- Lightweight zipvoice-based TTS (~1GB VRAM)
- 150x realtime speed on GPU, realtime on CPU
- High-quality 48kHz voice cloning
- Reference audio for voice matching

### LLM (Thinking) - Pluggable
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

---

## Configuration

### Environment Variables / Config File

```yaml
# config.yaml
llm:
  endpoint: "http://localhost:8002/v1/chat/completions"
  model: "Qwen/Qwen3-32B"  # or any OpenAI-compatible model
  api_key: "${LLM_API_KEY}"
  system_prompt: |
    You are a helpful voice assistant in a Mumble voice chat.
    Keep responses concise and conversational (1-3 sentences).
    Be friendly but not overly verbose - this is voice, not text.

tts:
  # LuxTTS settings (built-in)
  ref_audio: "voice_reference.wav"  # Reference audio for voice cloning
  ref_duration: 5                    # Seconds of reference to use
  num_steps: 4                       # Quality vs speed tradeoff (3-4 recommended)
  speed: 1.0                         # Playback speed

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
```

---

## Implementation Plan

### Phase 1: LLM Provider Interface

#### 1.1 Create LLM Interface

```python
# interfaces/llm.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    model: str | None = None
    usage: dict | None = None

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], context: dict | None = None) -> LLMResponse:
        """Generate response from conversation."""
        pass
```

#### 1.2 Implement OpenAI-Compatible Provider

```python
# providers/openai_llm.py
import httpx

class OpenAIChatLLM(LLMProvider):
    def __init__(self, endpoint: str, model: str, api_key: str | None = None,
                 system_prompt: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
    
    async def chat(self, messages: list[dict], context: dict | None = None) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepend system prompt if configured
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": full_messages,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model"),
            usage=data.get("usage"),
        )
```

### Phase 2: Voice Pipeline Integration

#### 2.1 Create Voice Pipeline

```python
# pipeline.py
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    wake_word: str | None = None
    silence_threshold_ms: int = 1500
    max_recording_ms: int = 30000

class VoicePipeline:
    def __init__(self, whisper, llm: LLMProvider, luxtts, config: PipelineConfig):
        self.whisper = whisper      # Existing Whisper integration
        self.llm = llm              # New LLM provider
        self.luxtts = luxtts        # Existing LuxTTS integration
        self.config = config
        self.conversation_history: list[dict] = []
    
    async def process_audio(self, audio: bytes, sample_rate: int) -> bytes | None:
        """Full pipeline: audio -> transcription -> LLM -> speech."""
        
        # 1. Transcribe with built-in Whisper
        transcription = self.whisper.transcribe(audio, sample_rate)
        
        if not transcription.text.strip():
            return None
        
        # 2. Check wake word (if configured)
        if self.config.wake_word:
            if self.config.wake_word.lower() not in transcription.text.lower():
                return None
            # Remove wake word from text
            text = transcription.text.lower().replace(self.config.wake_word.lower(), "").strip()
        else:
            text = transcription.text
        
        # 3. Build messages with history
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        
        # Keep last 10 exchanges to avoid context overflow
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # 4. Get LLM response
        llm_response = await self.llm.chat(self.conversation_history)
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": llm_response.content
        })
        
        # 5. Synthesize speech with LuxTTS
        audio_output = self.luxtts.generate_speech(
            text=llm_response.content,
            encoded_prompt=self.luxtts.encoded_voice,  # Pre-loaded voice
        )
        
        return audio_output
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
```

### Phase 3: Mumble Bot Integration

#### 3.1 Update MumbleBot to Use Pipeline

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
            
            # Process through pipeline (Whisper -> LLM -> LuxTTS)
            response = await self.pipeline.process_audio(
                complete_audio, 
                sample_rate=48000
            )
            
            if response:
                await self.play_audio(response)
    
    async def play_audio(self, audio: bytes):
        """Play audio to Mumble channel."""
        ...
```

### Phase 4: LLM Server Setup

#### 4.1 Running a Local LLM

**Option A: Ollama (Easiest)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:3b  # Small, fast
ollama pull qwen2.5:7b   # Better quality

# Ollama automatically exposes OpenAI-compatible API at http://localhost:11434/v1
```

**Option B: vLLM (Best for GPU)**
```bash
# Run vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002

# Or with Docker
docker run --gpus all -p 8002:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct
```

**Option C: llama.cpp (CPU-friendly)**
```bash
# Build and run llama.cpp server
./llama-server -m model.gguf --port 8002 --host 0.0.0.0
```

#### 4.2 Using Cloud APIs

```yaml
# config.yaml for OpenAI
llm:
  endpoint: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"

# config.yaml for Anthropic (via LiteLLM proxy)
llm:
  endpoint: "http://localhost:4000/v1/chat/completions"
  model: "claude-3-haiku"
  api_key: "${ANTHROPIC_API_KEY}"
```
        
### Phase 5: Advanced Features (Future)

#### 5.1 Streaming Responses
- Stream LLM responses token-by-token
- Start TTS synthesis as soon as first sentence is complete
- Reduce perceived latency

#### 5.2 Multi-User Support
- Track conversation history per user
- Different system prompts per user
- Concurrent processing

#### 5.3 Interrupt Handling
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
│   └── llm.py              # LLM abstract interface
├── providers/
│   ├── __init__.py
│   └── openai_llm.py       # OpenAI-compatible Chat
├── mumble/
│   ├── __init__.py
│   ├── bot.py              # MumbleVoiceBot
│   ├── audio.py            # Audio buffer/processing
│   └── connection.py       # Mumble connection handling
└── utils/
    ├── __init__.py
    └── audio.py            # Audio format conversion
```

---

## API Examples

### LLM Chat Request
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."},
      {"role": "user", "content": "What is the weather like today?"}
    ]
  }'
```

### LuxTTS Usage (Built-in)
```python
from zipvoice.luxvoice import LuxTTS

# Initialize once at startup
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# Pre-encode voice reference
encoded_voice = lux_tts.encode_prompt('voice_reference.wav', rms=0.01)

# Generate speech from LLM response
audio = lux_tts.generate_speech(llm_response.content, encoded_voice, num_steps=4)
```

---

## Migration Path

1. **Keep existing bot functional** during development
2. **Add LLM interface** as new module
3. **Implement OpenAI provider**
4. **Create pipeline** that wraps existing Whisper + LuxTTS
5. **Add configuration** for LLM endpoint
6. **Test with Ollama** locally
7. **Deploy** with cloud LLM or self-hosted vLLM

---

## Dependencies

```toml
[project]
dependencies = [
    "pymumble>=1.0",
    "httpx>=0.27",           # Async HTTP client for LLM API
    "pydantic>=2.0",         # Config validation
    "pyyaml>=6.0",           # Config files
    "numpy>=1.26",
    "soundfile>=0.12",
    # Existing dependencies for Whisper and LuxTTS
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

---

## Success Metrics

- [ ] LLM response latency < 2s (local) / < 1s (cloud)
- [ ] End-to-end latency < 4s (speech in -> speech out)
- [ ] Conversation context maintained across turns
- [ ] Wake word detection working (optional)
- [ ] Graceful fallback when LLM unavailable
