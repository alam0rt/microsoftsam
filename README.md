# microsoftsam

A Mumble voice bot that listens to users via ASR (NeMo Nemotron), thinks via an LLM (OpenAI-compatible), and responds with voice-cloned TTS (LuxTTS).

## Features

- **Voice cloning**: Speaks with any voice using LuxTTS reference audio
- **Streaming ASR**: NeMo Nemotron for low-latency speech recognition
- **LLM integration**: OpenAI-compatible API (Ollama, vLLM, OpenRouter, etc.)
- **Multi-persona**: Run multiple bots sharing TTS/STT/LLM resources
- **Tool calling**: Web search, sound effects, soul/personality switching
- **Barge-in detection**: Users can interrupt the bot mid-speech
- **Echo filtering**: Prevents the bot from responding to its own TTS output
- **Soul system**: Configurable personalities with themed responses

## Architecture

```
MumbleBot (base)
├── AudioInput    — Mumble callback → VAD → per-user buffering → silence detection
├── Transcriber   — 48kHz→16kHz resampling → ASR → text accumulation
├── Brain         — (pluggable) decides what to respond given a transcript
└── Speaker       — TTS queue → PCM generation → Mumble playback
```

Brain implementations:
- **LLMBrain**: Full LLM-powered responses with tool calling
- **EchoBrain**: Clone speaker voice, echo transcript back (parrot mode)
- **ReactiveBrain**: Fillers, echo fragments, deflections (no LLM)
- **AdaptiveBrain**: Routes between LLMBrain and ReactiveBrain based on `brain_power`
- **NullBrain**: Transcribe-only monitoring mode

## Quickstart

### Prerequisites

- Python 3.11 or 3.12
- CUDA-capable GPU (recommended) or CPU
- A Mumble server to connect to
- A reference audio file for voice cloning (`.wav`)

### With Nix (recommended)

```bash
git clone --recurse-submodules https://github.com/alam0rt/microsoftsam.git
cd microsoftsam
nix develop
uv sync
uv run python mumble_tts_bot.py --config config.example.yaml
```

### With uv (manual)

```bash
git clone --recurse-submodules https://github.com/alam0rt/microsoftsam.git
cd microsoftsam

# Install system deps (Ubuntu/Debian)
sudo apt-get install libopus-dev libsndfile1-dev espeak-ng ffmpeg

# Install Python deps
uv sync

# Run
uv run python mumble_tts_bot.py \
    --host mumble.example.com \
    --reference voice.wav \
    --llm-endpoint http://localhost:11434/v1/chat/completions \
    --llm-model llama3.2:3b
```

### With Docker Compose

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
docker-compose up
```

## Configuration

Configuration is loaded from a YAML file. CLI arguments override config values.

```yaml
mumble:
  host: mumble.example.com
  port: 64738
  user: VoiceBot
  channel: General

llm:
  endpoint: http://localhost:11434/v1/chat/completions
  model: llama3.2:3b
  temperature: 0.7
  max_tokens: 150

tts:
  ref_audio: reference.wav
  num_steps: 4
  device: cuda

stt:
  nemotron_model: nvidia/nemotron-speech-streaming-en-0.6b
  nemotron_chunk_ms: 160

bot:
  asr_threshold: 2000
  barge_in_enabled: true
  brain_power: 0.7  # 0.0=reactive only, 1.0=always LLM
```

See `config.example.yaml` for a complete example.

### Multi-Persona Mode

Run multiple bots in the same channel with shared resources:

```yaml
multi_persona: true

shared:
  llm:
    endpoint: http://localhost:8000/v1/chat/completions
    model: Qwen/Qwen3-32B

personas:
  - name: knight
    soul: knight
    mumble:
      channel: Tavern

  - name: potion-seller
    soul: potion-seller
    mumble:
      channel: Tavern
```

See `config.multi-persona.example.yaml` for a complete example.

## Souls System

Souls define bot personalities with voice, system prompt, and themed responses:

```
souls/
├── knight/
│   ├── soul.yaml          # Voice, LLM settings, events
│   ├── personality.md     # System prompt / character description
│   └── audio/
│       └── reference.wav  # Voice clone reference
├── potion-seller/
│   ├── soul.yaml
│   └── personality.md
└── README.md
```

See `souls/README.md` for details on creating custom souls.

## Development

```bash
# Install dev dependencies
make dev

# Run linter
make lint

# Run type checker
make typecheck

# Run all checks
make check

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format
```

## Package Structure

```
mumble_voice_bot/
├── bot.py                  # MumbleBot base (planned)
├── audio.py                # PCM utilities, resampling
├── text_processing.py      # TTS text preparation
├── utils.py                # General helpers
├── coordination.py         # SharedBotServices, multi-bot coordination
├── config.py               # YAML config loading, dataclasses
├── brains/
│   ├── echo.py             # EchoBrain (parrot)
│   ├── llm.py              # LLMBrain (full intelligence)
│   ├── reactive.py         # ReactiveBrain (fillers, no LLM)
│   └── adaptive.py         # AdaptiveBrain (brain_power routing)
├── interfaces/
│   ├── brain.py            # Brain protocol, Utterance, BotResponse
│   ├── llm.py              # LLM provider interface
│   ├── stt.py              # STT provider interface
│   ├── tts.py              # TTS provider interface
│   └── services.py         # Shared services protocols
├── providers/
│   ├── luxtts.py           # StreamingLuxTTS
│   ├── openai_llm.py       # OpenAI-compatible LLM
│   ├── nemotron_stt.py     # NeMo Nemotron ASR
│   └── wyoming_*.py        # Wyoming protocol providers
└── tools/
    ├── web_search.py       # DuckDuckGo web search
    └── sound_effects.py    # Sound effect playback
```

## License

MIT
