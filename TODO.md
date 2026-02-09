# Multi-Persona Bot Implementation

## Overview

Multi-persona support allows running multiple bot personalities that share TTS/STT/LLM resources while maintaining separate Mumble identities.

## Architecture (Simplified)

The architecture is simple: both single-bot and multi-bot modes use the same `MumbleVoiceBot` class. The only difference is whether services are shared.

```
┌─────────────────────────────────────────────────────────────┐
│                  SharedBotServices                          │
│  - Single TTS engine (StreamingLuxTTS)                      │
│  - Single STT engine (Wyoming/Nemotron)                     │
│  - Single LLM client (OpenAI-compatible)                    │
│  - Voice prompt cache (per-persona)                         │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │MumbleVoiceBot │ │MumbleVoiceBot │ │MumbleVoiceBot │
    │  (Knight)     │ │(PotionSeller) │ │   (Custom)    │
    │               │ │               │ │               │
    │ - Mumble conn │ │ - Mumble conn │ │ - Mumble conn │
    │ - Voice prompt│ │ - Voice prompt│ │ - Voice prompt│
    │ - Sys prompt  │ │ - Sys prompt  │ │ - Sys prompt  │
    │ - Conv history│ │ - Conv history│ │ - Conv history│
    └───────────────┘ └───────────────┘ └───────────────┘
```

## Completed

- [x] **Refactored MumbleVoiceBot** for dependency injection
  - [x] Added `shared_tts`, `shared_stt`, `shared_llm`, `voice_prompt` parameters
  - [x] Bot can use provided shared services or create its own
  - [x] Backward compatible - single-bot mode unchanged
  
- [x] **Created SharedBotServices class**
  - [x] Container for shared TTS/STT/LLM
  - [x] `load_voice()` method with caching
  - [x] Device management

- [x] **Created create_shared_services() factory**
  - [x] Initializes TTS/STT/LLM once
  - [x] Auto-detects best device
  - [x] Configurable via parameters

- [x] **Updated multi-persona mode**
  - [x] Auto-detects multi-persona configs
  - [x] Creates shared services once
  - [x] Instantiates N MumbleVoiceBot instances
  - [x] Each bot gets own Mumble connection and system prompt

- [x] **Tests**
  - [x] SharedBotServices tests (6 tests)
  - [x] MumbleVoiceBot shared services tests (2 tests)
  - [x] Multi-persona config tests (3 tests)
  - [x] **670 total tests passing**

## Usage

### Single Bot (unchanged)
```bash
python mumble_tts_bot.py --config config.sauron.yaml
```

### Multiple Bots
```bash
# Config with `personas:` section is auto-detected
python mumble_tts_bot.py --config config.multi-persona.yaml
```

### Example Multi-Persona Config
```yaml
personas:
  - name: knight
    soul: knight
    mumble:
      user: "Sir Knight"
    tts:
      ref_audio: "souls/knight/audio/ref.wav"
      
  - name: seller
    soul: potion-seller
    mumble:
      user: "Potion Seller"
    tts:
      ref_audio: "souls/potion-seller/audio/ref.wav"

shared:
  llm:
    endpoint: "http://localhost:11434/v1/chat/completions"
    model: "llama3.2:3b"
  stt:
    provider: "wyoming"
    wyoming_host: "localhost"

mumble:
  host: "mumble.example.com"
  port: 64738
```

## Files Reference

- `mumble_tts_bot.py` - Main module with `SharedBotServices`, `create_shared_services()`, `MumbleVoiceBot`
- `mumble_voice_bot/multi_persona_config.py` - Config loader
- `config.multi-persona.example.yaml` - Example config
- `tests/test_coordinator.py` - SharedBotServices tests
- `tests/test_multi_persona.py` - Interface tests
- `tests/test_multi_persona_config.py` - Config tests
