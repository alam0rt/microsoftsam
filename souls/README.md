# Souls Directory

A **soul** represents a complete voice identity for the Mumble Voice Bot, combining personality, voice characteristics, and audio assets.

## Directory Structure

```
souls/
├── README.md                 # This file
├── example/                  # Example soul (copy to create new ones)
│   ├── soul.yaml            # Soul configuration
│   ├── personality.md       # Character/personality definition
│   ├── audio/               # Audio assets
│   │   └── reference.wav    # Voice cloning reference audio
│   └── weights/             # Model weights (optional)
│       └── .gitkeep         # Fine-tuned model weights go here
└── imperial-guard/          # Example: Elder Scrolls guard
    ├── soul.yaml
    ├── personality.md
    ├── audio/
    │   └── reference.wav
    └── weights/
```

## Soul Configuration (`soul.yaml`)

```yaml
# Soul metadata
name: "Example Soul"
description: "A helpful voice assistant"
author: "Your Name"
version: "1.0.0"

# Voice settings (TTS)
voice:
  ref_audio: "audio/"  # Directory or file - grabs first audio file found
  ref_duration: 5.0                  # Seconds of reference to use
  num_steps: 4                       # Quality vs speed (3-4 recommended)
  speed: 1.0                         # Playback speed

# Optional: Custom model weights
weights:
  tts_model: null                    # Path to fine-tuned TTS weights
  voice_encoder: null                # Custom voice encoder weights

# LLM behavior overrides (optional)
llm:
  temperature: 0.7
  max_tokens: 150
```

## Creating a New Soul

1. Copy the `example/` directory:
   ```bash
   cp -r souls/example souls/my-new-soul
   ```

2. Edit `soul.yaml` with your configuration

3. Drop your voice sample into the `audio/` directory:
   - Any filename works - the first audio file found will be used
   - Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`
   - 5-15 seconds of clear speech
   - Single speaker, minimal background noise
   - 16kHz+ sample rate recommended

4. Customize `personality.md` with character traits

5. (Optional) Add fine-tuned model weights to `weights/`

## Using a Soul

Reference the soul in your config:

```yaml
# config.yaml
soul: "imperial-guard"  # Load from souls/imperial-guard/
```

Or specify the full path:

```yaml
soul_path: "/path/to/custom/soul"
```

## Audio Guidelines

For best voice cloning results:
- **Duration**: 5-15 seconds of continuous speech
- **Quality**: Clear audio, no background music/noise
- **Format**: WAV (16-bit, mono or stereo)
- **Sample Rate**: 16kHz minimum, 24kHz+ preferred
- **Content**: Natural speaking voice, varied intonation
