# Mumble Voice Bot - AI Coding Instructions

## Project Overview
A Mumble voice chat bot with LLM-powered conversations using voice cloning TTS (LuxTTS), streaming ASR (NeMo Nemotron), and OpenAI-compatible LLM backends. Supports single-bot and multi-persona modes where multiple bot identities share expensive GPU resources.

## Architecture

### Core Pipeline
```
Audio In → Nemotron STT → LLM (OpenAI API) → LuxTTS → Audio Out (Mumble)
```

### Key Components
- **`mumble_tts_bot.py`**: Main entry point, contains `MumbleVoiceBot` class (~3800 lines)
- **`mumble_voice_bot/`**: Library module with pluggable interfaces
  - `interfaces/`: Abstract bases (`LLMProvider`, `STTProvider`, `TTSProvider`, `Tool`)
  - `providers/`: Implementations (OpenAI LLM, Wyoming STT/TTS, Nemotron ASR)
  - `tools/`: LLM tools (web search, sound effects, souls management)
- **`souls/`**: Bot personas with voice cloning references, personalities, and configs
- **`vendor/`**: Vendored dependencies (pymumble, LuxTTS, LinaCodec)

### Multi-Persona Architecture
Expensive services (TTS engine, STT, LLM client) are shared via `SharedServices`. Each persona gets its own Mumble connection, voice prompt, system prompt, and conversation history. See `interfaces/services.py` for the `SharedServices` and `Persona` abstractions.

## Development Commands
```bash
make dev          # Install dev dependencies with uv
make test         # Run pytest (670+ tests)
make test-quick   # Fast test run with -x --tb=short
make lint         # Run ruff linter
make format       # Format with ruff
make typecheck    # Run mypy on mumble_voice_bot/
make check        # lint + typecheck
```

## Configuration
- YAML configs with `${ENV_VAR}` expansion (see `config.example.yaml`)
- Single bot: `python mumble_tts_bot.py --config config.yaml`
- Multi-persona: Use `personas:` section (see `config.multi-persona.example.yaml`)
- Souls: Complete voice identities in `souls/<name>/` with `soul.yaml`, `personality.md`, and `audio/reference.wav`

## Code Patterns

### Interface-First Design
All major components use abstract base classes in `interfaces/`. When adding new providers:
```python
# Example: New STT provider
from mumble_voice_bot.interfaces.stt import STTProvider, STTResult

class MySTT(STTProvider):
    async def transcribe(self, audio_data: bytes, ...) -> STTResult:
        ...
```

### Tool System
Tools extend `mumble_voice_bot.tools.base.Tool` with JSON Schema parameters:
```python
class MyTool(Tool):
    @property
    def name(self) -> str: return "my_tool"
    @property
    def parameters(self) -> dict: return {"type": "object", "properties": {...}}
    async def execute(self, **kwargs) -> str: ...
```

### Event System
Mumble events flow through `EventDispatcher` → `MumbleEventHandler` implementations in `handlers.py`. Events are typed dataclasses in `interfaces/events.py`.

### Async/Threading
- Mumble callbacks run in pymumble thread
- Use `asyncio.run_coroutine_threadsafe()` for async handlers
- TTS playback uses separate worker threads (see `perf.py`)

## Testing Conventions
- Tests in `tests/` mirror source structure
- Use `@pytest.mark.asyncio` for async tests
- Fixtures in `conftest.py` provide test audio (16kHz/48kHz PCM), mock Wyoming info
- Mock external services; tests should not require running Mumble/LLM/TTS servers

## Audio Format Requirements
| Context | Sample Rate | Format |
|---------|-------------|--------|
| Mumble I/O | 48kHz | 16-bit mono PCM |
| STT Input | 16kHz | 16-bit mono PCM |
| LuxTTS Output | 48kHz | 16-bit mono PCM |

## Key Files to Reference
- `config.example.yaml`: All configuration options
- `souls/README.md`: Creating new bot personas
- `docs/perf.md`: Latency optimization architecture
- `interfaces/services.py`: Multi-persona abstractions
- `tests/conftest.py`: Test fixtures and patterns
