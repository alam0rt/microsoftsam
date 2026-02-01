# Mumble TTS Bot Plan

A simple Mumble bot that reads text messages aloud using LuxTTS voice cloning.
Packaged as a Nix flake for reproducible builds.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Mumble Server  │◄───►│  pymumble_py3│◄───►│     LuxTTS      │
│                 │     │  (bot core)  │     │  (speech gen)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
        ▲                      │
        │                      ▼
        │              ┌──────────────┐
        └──────────────│  PCM Audio   │
                       │  (48kHz/16b) │
                       └──────────────┘
```

## Project Structure

```
.
├── flake.nix              # Nix flake with packages and devShells
├── pyproject.toml         # Python package metadata
├── nix/
│   └── opuslib-paths.patch # Patch for opuslib to find libopus
├── mumble_tts_bot.py      # Main bot script
├── PLAN.md                # This file
├── LuxTTS/                # Cloned LuxTTS repo (git ignored)
└── reference.wav          # Voice reference audio (user provided)
```

## Nix Integration

### Build the Package

Build the mumble-tts-bot package:
```bash
nix build
./result/bin/mumble-tts-bot --help
```

### Development Shell

Enter the development environment:
```bash
nix develop
```

### Running the Bot

```bash
# Option 1: Run the built package (requires LuxTTS in PYTHONPATH)
nix run -- --host localhost --user "TTS Bot" --reference reference.wav

# Option 2: Inside the devShell (includes LuxTTS setup)
nix develop
python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference reference.wav
```

## Flow

1. **Connect** to Mumble server using `pymumble_py3.Mumble`
2. **Listen** for text messages via `PYMUMBLE_CLBK_TEXTMESSAGERECEIVED` callback
3. **Generate** speech using LuxTTS when a message is received
4. **Send** audio to Mumble using `mumble.sound_output.add_sound()`

## Key Components

| Component | Purpose |
|-----------|---------|
| `pymumble_py3.Mumble` | Main connection class |
| `callbacks.set_callback()` | Listen for text messages |
| `mumble.sound_output.add_sound()` | Send PCM audio |
| `LuxTTS.generate_speech()` | Generate TTS audio |

## Audio Format Requirements

| Property | Value |
|----------|-------|
| Sample Rate | 48000 Hz |
| Bit Depth | 16-bit signed |
| Channels | Mono |
| Encoding | Little-endian PCM |

LuxTTS outputs at 48kHz which matches Mumble's requirements perfectly.

## Dependencies (managed by Nix)

### Nix-provided Python Packages
These are built and provided directly by Nix (no pip):

| Package | Source | Notes |
|---------|--------|-------|
| `pymumble_py3` | botamusique repo | Extracted from algielen/botamusique |
| `opuslib-next` | PyPI + patch | Patched to find libopus via Nix |
| `numpy` | nixpkgs | Standard scientific computing |
| `protobuf` | nixpkgs | Mumble protocol |
| `soundfile` | nixpkgs | Audio file I/O |

### Nix-provided System Libraries
| Library | Purpose |
|---------|---------|
| `libopus` | Opus codec for audio encoding |
| `ffmpeg` | Audio processing |
| `espeak-ng` | Phonemization for LuxTTS |
| `openssl` | TLS for Mumble connection |
| `libsndfile` | Audio file support |

### Pip-installed (in venv, pending nixification)
| Package | Notes |
|---------|-------|
| LuxTTS dependencies | Complex ML stack, uses venv for now |

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Mumble server address | `localhost` |
| `--port` | Mumble server port | `64738` |
| `--user` | Bot username | `TTSBot` |
| `--password` | Server password | (empty) |
| `--channel` | Channel to join | (root) |
| `--reference` | Reference audio for voice | `reference.wav` |
| `--device` | Compute device | `cpu` |

## Implementation Steps

### Step 1: Basic Connection
```python
import pymumble_py3 as pymumble

mumble = pymumble.Mumble(host, user, port=port, password=password)
mumble.start()
mumble.is_ready()  # blocks until connected
```

### Step 2: Text Message Callback
```python
from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED

def on_message(message):
    text = message.message  # HTML may need stripping
    # Generate and play TTS
    
mumble.callbacks.set_callback(PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, on_message)
```

### Step 3: TTS Generation
```python
from zipvoice.luxvoice import LuxTTS

lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu')
encoded_prompt = lux_tts.encode_prompt('reference.wav', rms=0.01)

def generate_speech(text):
    wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)
    return wav.numpy().squeeze()
```

### Step 4: Send Audio to Mumble
```python
import numpy as np

def send_audio(wav_float):
    # Convert float32 [-1, 1] to int16 PCM
    pcm = (wav_float * 32767).astype(np.int16)
    mumble.sound_output.add_sound(pcm.tobytes())
```

### Step 5: HTML Stripping
```python
import re

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text)
```

## Complete Bot Structure

```
mumble_tts_bot.py
├── Imports
├── Argument parsing
├── LuxTTS initialization
├── Message handler (callback)
│   ├── Strip HTML from message
│   ├── Generate speech
│   └── Send audio
├── Mumble connection
├── Channel join (optional)
└── Main loop (keep alive)
```

## Usage Example

```bash
# Enter the Nix development shell
nix develop .#mumblebot

# Run the bot
python mumble_tts_bot.py \
    --host mumble.example.com \
    --user "TTS Bot" \
    --reference voice.wav \
    --channel "General"
```

## Testing

```bash
# Enter devShell
nix develop .#mumblebot

# Test with a local Mumble server (murmur)
# 1. Start murmur on localhost:64738
# 2. Connect with a Mumble client
# 3. Run the bot:
python mumble_tts_bot.py --host localhost --user TestBot --reference reference.wav

# 4. Send a text message in Mumble - the bot should speak it!
```

## Notes

- The bot runs the text message callback in a separate thread (pymumble behavior)
- Audio is queued and sent automatically by pymumble's sound_output
- Reference audio for voice cloning should be at least 3 seconds
- Keep callback processing short to avoid audio jitter
- The Nix flake handles all native dependencies (opus, ssl, etc.)
