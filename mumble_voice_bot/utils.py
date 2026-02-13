"""General utility functions for the Mumble Voice Bot.

Extracted from mumble_tts_bot.py to reduce monolith size.
"""

import os
import re


def strip_html(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Text possibly containing HTML tags.

    Returns:
        Text with all HTML tags removed.
    """
    return re.sub(r'<[^>]+>', '', text)


def get_best_device() -> str:
    """Auto-detect the best available compute device.

    Checks for CUDA (NVIDIA GPU), MPS (Apple Silicon), or falls back to CPU.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA available: {device_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[Device] MPS (Apple Silicon) available")
            return 'mps'
        else:
            print("[Device] Using CPU")
            return 'cpu'
    except ImportError:
        return 'cpu'


def ensure_models_downloaded(device: str = 'cuda') -> None:
    """Pre-download all required models before starting the bot.

    This ensures models are cached locally before connecting to Mumble,
    preventing long delays during the first voice interaction.

    Downloads:
    - LuxTTS model (YatharthS/LuxTTS)
    - Whisper model (openai/whisper-base for GPU, whisper-tiny for CPU)

    Args:
        device: Compute device (affects whisper model selection).
    """
    from huggingface_hub import snapshot_download
    from transformers import pipeline as hf_pipeline

    print("=" * 60)
    print("[Models] Ensuring all required models are downloaded...")
    print("=" * 60)

    # Download LuxTTS model
    print("[Models] Checking LuxTTS model (YatharthS/LuxTTS)...")
    try:
        model_path = snapshot_download("YatharthS/LuxTTS")
        print(f"[Models] LuxTTS ready at: {model_path}")
    except Exception as e:
        print(f"[Models] Failed to download LuxTTS: {e}")
        raise

    # Download Whisper model (used for transcription in TTS pipeline)
    whisper_model = "openai/whisper-base" if device != 'cpu' else "openai/whisper-tiny"
    print(f"[Models] Checking Whisper model ({whisper_model})...")
    try:
        # This will download if not cached, or load from cache
        _ = hf_pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device='cpu'  # Load on CPU just to verify download, actual device set later
        )
        print(f"[Models] Whisper ready: {whisper_model}")
    except Exception as e:
        print(f"[Models] Failed to download Whisper: {e}")
        raise

    print("=" * 60)
    print("[Models] All models ready!")
    print("=" * 60)


def get_vendor_paths() -> list[str]:
    """Get vendor submodule paths relative to the project root.

    Returns:
        List of absolute paths to vendor directories that should be on sys.path.
    """
    this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [
        os.path.join(this_dir, "vendor", "botamusique"),
        os.path.join(this_dir, "vendor", "LuxTTS"),
        os.path.join(this_dir, "vendor", "LinaCodec", "src"),
    ]


def setup_vendor_paths() -> None:
    """Add vendor submodule paths to sys.path.

    Call this early in entry points before importing vendor modules.
    """
    import sys
    for path in get_vendor_paths():
        if path not in sys.path:
            sys.path.insert(0, path)
