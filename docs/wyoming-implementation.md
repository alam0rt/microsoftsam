# Wyoming Protocol Integration

This document describes the Wyoming protocol integration in the microsoftsam project.

## Overview

The Wyoming protocol enables modular speech processing by separating STT (Speech-to-Text) 
and TTS (Text-to-Speech) into independent network services that communicate over TCP.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           microsoftsam Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      Wyoming Protocol       ┌────────────────────────┐   │
│  │   Mumble     │◄──────────────────────────►│  wyoming-faster-whisper │   │
│  │    Bot       │         (TCP)               │  (External Server)     │   │
│  │              │                             │  STT Backend           │   │
│  └──────┬───────┘                             └────────────────────────┘   │
│         │                                                                   │
│         │              Wyoming Protocol       ┌────────────────────────┐   │
│         └────────────────────────────────────►│  wyoming-luxtts        │   │
│                        (TCP)                  │  (Our TTS Server)      │   │
│                                               │  Wraps StreamingLuxTTS │   │
│                                               └────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

The Wyoming integration follows the existing modular architecture:

```
mumble_voice_bot/
├── interfaces/
│   ├── __init__.py
│   ├── llm.py          # LLMProvider interface
│   ├── stt.py          # STTProvider interface (NEW)
│   └── tts.py          # TTSProvider interface (NEW)
└── providers/
    ├── __init__.py
    ├── openai_llm.py   # OpenAI-compatible LLM provider
    ├── wyoming_stt.py  # Wyoming STT client (NEW)
    ├── wyoming_tts.py  # Wyoming TTS client (NEW)
    └── wyoming_tts_server.py  # Wyoming TTS server wrapping LuxTTS (NEW)
```

## Usage

### Using Wyoming STT with the Mumble Bot

Start the bot with Wyoming STT instead of local Whisper:

```bash
# Start wyoming-faster-whisper (in another terminal or container)
docker run -p 10300:10300 rhasspy/wyoming-faster-whisper --model base --language en

# Start the bot with Wyoming STT
python mumble_tts_bot.py \
    --host mumble.example.com \
    --reference voice.wav \
    --wyoming-stt-host localhost \
    --wyoming-stt-port 10300
```

### Running the Wyoming LuxTTS Server

Expose LuxTTS as a Wyoming-compatible TTS server:

```bash
# Using the CLI entry point
wyoming-luxtts --reference voice.wav --port 10400

# Or using python module
python -m mumble_voice_bot.providers.wyoming_tts_server \
    --reference voice.wav \
    --port 10400 \
    --device cuda
```

### Using Providers Programmatically

```python
import asyncio
from mumble_voice_bot import WyomingSTT, WyomingTTS

async def main():
    # STT Example
    stt = WyomingSTT(host="localhost", port=10300)
    result = await stt.transcribe(audio_bytes, sample_rate=16000)
    print(f"Transcribed: {result.text}")
    
    # TTS Example
    tts = WyomingTTS(host="localhost", port=10400)
    result = await tts.synthesize("Hello world!")
    # result.audio contains raw PCM bytes at result.sample_rate

asyncio.run(main())
```

### Synchronous Wrappers

For non-async code, use the sync wrappers:

```python
from mumble_voice_bot import WyomingSTTSync, WyomingTTSSync

stt = WyomingSTTSync(host="localhost", port=10300)
result = stt.transcribe(audio_bytes)
print(result.text)
```

## Docker Compose

The included `docker-compose.yml` sets up the full stack:

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f mumble-bot
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WYOMING_STT_HOST` | Wyoming STT server hostname | `localhost` |
| `WYOMING_STT_PORT` | Wyoming STT server port | `10300` |
| `WYOMING_TTS_HOST` | Wyoming TTS server hostname | `localhost` |
| `WYOMING_TTS_PORT` | Wyoming TTS server port | `10400` |
| `LLM_ENDPOINT` | OpenAI-compatible LLM endpoint | `http://localhost:11434/v1/chat/completions` |
| `LLM_MODEL` | LLM model name | `llama3.2:3b` |

### CLI Arguments

```
--wyoming-stt-host HOST    Wyoming STT server host (enables Wyoming STT)
--wyoming-stt-port PORT    Wyoming STT server port (default: 10300)
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=mumble_voice_bot --cov-report=html

# Run integration tests (requires running servers)
RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py
```

## Interfaces

### STTProvider

```python
class STTProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult: ...
    
    @abstractmethod
    async def is_available(self) -> bool: ...
```

### TTSProvider

```python
class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> TTSResult: ...
    
    @abstractmethod
    async def get_voices(self) -> list[TTSVoice]: ...
    
    @abstractmethod
    async def is_available(self) -> bool: ...
```

## References

- [Wyoming Protocol Specification](https://github.com/OHF-Voice/wyoming)
- [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper)
- [wyoming-piper (TTS reference)](https://github.com/rhasspy/wyoming-piper)
