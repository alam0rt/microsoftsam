# Multi-Persona Bot Implementation TODO

## Overview

This tracks the implementation of multi-persona bot support, allowing multiple bot personalities to share TTS/STT/LLM resources while maintaining separate Mumble identities.

## Completed

- [x] **Phase 1: Interfaces & Config** (February 2026)
  - [x] Design shared services interfaces (`interfaces/services.py`)
  - [x] Create `VoicePrompt`, `PersonaIdentity`, `PersonaConfig`, `Persona` dataclasses
  - [x] Create `SharedServices` container for TTS/STT/LLM
  - [x] Create `PersonaManager` abstract base class
  - [x] Design `InteractionConfig` for bot-to-bot settings
  - [x] Design `MultiPersonaConfig` schema
  - [x] Write tests for interfaces (44 tests)
  - [x] Create `multi_persona_config.py` config loader
  - [x] Write tests for config loading (29 tests)
  - [x] Create example config (`config.multi-persona.example.yaml`)

## In Progress

- [ ] **Phase 2: MultiPersonaCoordinator**
  - [ ] Create `MultiPersonaCoordinator` class that:
    - [ ] Initializes `SharedServices` from config
    - [ ] Pre-loads voice prompts for all personas
    - [ ] Creates separate Mumble connections per persona
    - [ ] Routes audio to appropriate persona handlers
    - [ ] Manages persona state transitions
  - [ ] Implement bot-to-bot audio routing
  - [ ] Add loop prevention for bot-to-bot conversations

## Not Started

- [ ] **Phase 3: Refactor MumbleVoiceBot**
  - [ ] Extract TTS/STT/LLM initialization into factory functions
  - [ ] Add dependency injection for `SharedServices`
  - [ ] Create lightweight `PersonaBot` class that uses shared services
  - [ ] Maintain backward compatibility with single-bot mode

- [ ] **Phase 4: CLI & Runtime**
  - [ ] Add `--multi-persona` CLI flag
  - [ ] Auto-detect multi-persona configs via `is_multi_persona_config()`
  - [ ] Add graceful startup/shutdown for multiple Mumble connections
  - [ ] Add health monitoring for each persona

- [ ] **Phase 5: Bot-to-Bot Interactions**
  - [ ] Implement `InteractionConfig` logic:
    - [ ] `enable_cross_talk` - bots hear each other
    - [ ] `response_delay_ms` - delay before responding to other bot
    - [ ] `max_chain_length` - prevent infinite loops
    - [ ] `cooldown_after_chain_ms` - cooldown after max chain
  - [ ] Add speaker detection to identify bot vs human audio
  - [ ] Add conversation chain tracking

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    SharedServices                           │
│  - Single TTS engine (multiple voice prompts)               │
│  - Single STT engine                                        │
│  - Single LLM client (different system prompts per persona) │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Persona 1   │ │   Persona 2   │ │   Persona N   │
    │  (Knight)     │ │(PotionSeller) │ │   (Custom)    │
    │               │ │               │ │               │
    │ - Mumble conn │ │ - Mumble conn │ │ - Mumble conn │
    │ - Voice prompt│ │ - Voice prompt│ │ - Voice prompt│
    │ - Sys prompt  │ │ - Sys prompt  │ │ - Sys prompt  │
    │ - Conv history│ │ - Conv history│ │ - Conv history│
    └───────────────┘ └───────────────┘ └───────────────┘
```

## Files Reference

- `mumble_voice_bot/interfaces/services.py` - Core interfaces
- `mumble_voice_bot/multi_persona_config.py` - Config loader
- `config.multi-persona.example.yaml` - Example config
- `tests/test_multi_persona.py` - Interface tests
- `tests/test_multi_persona_config.py` - Config tests
