#!/usr/bin/env python3
"""
Qwen3-TTS Setup and Launch Script

This script sets up and launches Qwen3-TTS with a web UI for voice generation,
voice design, and voice cloning.

Usage:
    uv run scripts/setup_qwen3_tts.py [--host HOST] [--port PORT] [--model MODEL] [--task TASK]

Examples:
    # Launch with defaults (Base model for voice cloning on 0.0.0.0:9999)
    uv run scripts/setup_qwen3_tts.py

    # Launch CustomVoice model
    uv run scripts/setup_qwen3_tts.py --task CustomVoice

    # Launch VoiceDesign model
    uv run scripts/setup_qwen3_tts.py --task VoiceDesign

    # Custom host/port
    uv run scripts/setup_qwen3_tts.py --host 127.0.0.1 --port 8080
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch Qwen3-TTS Web UI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task Types:
  Base         - Voice cloning from reference audio (default)
  CustomVoice  - Predefined speaker voices with style control
  VoiceDesign  - Natural language voice style description

Available Speakers (CustomVoice):
  Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
  English: Ryan, Aiden
  Japanese: Ono_Anna
  Korean: Sohee

Supported Languages:
  Chinese, English, Japanese, Korean, German, French,
  Russian, Portuguese, Spanish, Italian
        """,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port to run the server on (default: 9999)",
    )
    parser.add_argument(
        "--task",
        choices=["Base", "CustomVoice", "VoiceDesign"],
        default="Base",
        help="Task type / model variant (default: Base for voice cloning)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name (auto-selected based on task if not provided)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="SSL certificate file for HTTPS (recommended for Base model voice cloning)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="SSL key file for HTTPS",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL verification (for self-signed certs)",
    )

    args = parser.parse_args()

    # Map task to model if not explicitly provided
    task_to_model = {
        "Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    model = args.model or task_to_model[args.task]

    # Build command
    cmd = [
        "qwen-tts-demo",
        model,
        "--ip",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.ssl_certfile:
        cmd.extend(["--ssl-certfile", args.ssl_certfile])
    if args.ssl_keyfile:
        cmd.extend(["--ssl-keyfile", args.ssl_keyfile])
    if args.no_ssl_verify:
        cmd.append("--no-ssl-verify")

    print(f"🚀 Launching Qwen3-TTS Web UI")
    print(f"   Model: {model}")
    print(f"   Task: {args.task}")
    print(f"   URL: http{'s' if args.ssl_certfile else ''}://{args.host}:{args.port}")
    print()

    if args.task == "Base" and not args.ssl_certfile:
        print("⚠️  Note: For voice cloning with microphone access from remote browsers,")
        print("   you may need HTTPS. Use --ssl-certfile and --ssl-keyfile options.")
        print("   Generate self-signed certs with:")
        print('   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"')
        print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except FileNotFoundError:
        print("❌ Error: qwen-tts-demo not found.")
        print("   Make sure qwen-tts is installed: uv pip install qwen-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
