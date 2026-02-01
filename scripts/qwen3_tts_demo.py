# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "qwen-tts>=0.1.0",
#     "torch>=2.4.0",
#     "gradio>=4.0.0",
# ]
# [tool.uv]
# find-links = ["https://download.pytorch.org/whl/cu124"]
# extra-index-url = ["https://download.pytorch.org/whl/cu124"]
# ///
"""
Qwen3-TTS Web UI Demo

Launch a Gradio-based web UI for Qwen3-TTS with support for:
- Voice cloning from reference audio
- Custom voice generation with predefined speakers  
- Voice design with natural language descriptions

Requirements:
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended for 1.7B model)
- Flash Attention 2 (optional, for better performance)
  Install separately: MAX_JOBS=4 pip install flash-attn --no-build-isolation

Usage:
    uv run scripts/qwen3_tts_demo.py [HOST:PORT] [--task TASK]

Examples:
    # Launch with defaults (Base model for voice cloning on 0.0.0.0:9999)
    uv run scripts/qwen3_tts_demo.py

    # Specify host:port
    uv run scripts/qwen3_tts_demo.py 0.0.0.0:9999

    # Launch CustomVoice model
    uv run scripts/qwen3_tts_demo.py 0.0.0.0:9999 --task CustomVoice

    # Launch VoiceDesign model  
    uv run scripts/qwen3_tts_demo.py --task VoiceDesign

With Nix:
    nix develop -c uv run scripts/qwen3_tts_demo.py 0.0.0.0:9999
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
        "address",
        nargs="?",
        default="0.0.0.0:9999",
        help="Host:port to bind the server to (default: 0.0.0.0:9999)",
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
        "--ssl",
        action="store_true",
        help="Enable HTTPS with auto-generated self-signed certificate",
    )

    args = parser.parse_args()

    # Parse address
    if ":" in args.address:
        host, port = args.address.rsplit(":", 1)
        port = int(port)
    else:
        host = args.address
        port = 9999

    # Map task to model if not explicitly provided
    # Using 0.6B models (smallest available)
    task_to_model = {
        "Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",  # No 0.6B VoiceDesign available
    }

    model = args.model or task_to_model[args.task]

    # Build command
    cmd = [
        sys.executable, "-m", "qwen_tts.demo",
        model,
        "--ip", host,
        "--port", str(port),
    ]

    # Generate SSL certs if requested
    ssl_files = None
    if args.ssl or args.task == "Base":
        import tempfile
        import os
        
        # Create temp directory for SSL files
        ssl_dir = tempfile.mkdtemp(prefix="qwen-tts-ssl-")
        cert_file = os.path.join(ssl_dir, "cert.pem")
        key_file = os.path.join(ssl_dir, "key.pem")
        
        print("🔐 Generating self-signed SSL certificate for HTTPS...")
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost"
            ], check=True, capture_output=True)
            
            cmd.extend([
                "--ssl-certfile", cert_file,
                "--ssl-keyfile", key_file,
                "--no-ssl-verify",
            ])
            ssl_files = (ssl_dir, cert_file, key_file)
            protocol = "https"
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  Could not generate SSL certificate: {e}")
            print("   Continuing without HTTPS (microphone may not work remotely)")
            protocol = "http"
    else:
        protocol = "http"

    print()
    print("🚀 Launching Qwen3-TTS Web UI")
    print(f"   Model: {model}")
    print(f"   Task: {args.task}")
    print(f"   URL: {protocol}://{host}:{port}")
    print()

    if args.task == "Base" and protocol == "http":
        print("⚠️  Note: For voice cloning with microphone access from remote browsers,")
        print("   HTTPS is required. Use --ssl flag or run with:")
        print(f'   uv run scripts/qwen3_tts_demo.py {args.address} --task Base --ssl')
        print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure qwen-tts is installed")
        sys.exit(1)
    finally:
        # Cleanup SSL files
        if ssl_files:
            import shutil
            shutil.rmtree(ssl_files[0], ignore_errors=True)


if __name__ == "__main__":
    main()
