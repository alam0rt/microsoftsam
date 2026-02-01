# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm-omni",
#     "vllm",
#     "soundfile",
# ]
# ///
"""
Qwen3-TTS Offline Inference Demo (via vllm-omni)

Run Qwen3-TTS models for text-to-speech generation using vLLM Omni.

Requirements:
- NVIDIA GPU with CUDA support (16GB+ VRAM for 1.7B, 8GB for 0.6B)

Usage:
    uv run scripts/qwen3_tts_demo.py [OPTIONS]

Examples:
    # Default: CustomVoice task
    uv run scripts/qwen3_tts_demo.py

    # Voice cloning from reference audio
    uv run scripts/qwen3_tts_demo.py --task Base

    # Voice design from description
    uv run scripts/qwen3_tts_demo.py --task VoiceDesign

    # Custom text input
    uv run scripts/qwen3_tts_demo.py --text "Hello, world!"

References:
    https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts
"""

import argparse
import os
from typing import NamedTuple

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import soundfile as sf
from vllm import SamplingParams
from vllm_omni import Omni


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""
    inputs: dict
    model_name: str


# Task to model mapping (0.6B for faster inference, 1.7B for VoiceDesign)
TASK_MODELS = {
    "Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Available speakers for CustomVoice
SPEAKERS = {
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "English": ["Ryan", "Aiden"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}


def build_prompt(text: str) -> str:
    """Build the standard Qwen3-TTS prompt format."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def get_custom_voice_query(
    text: str = "Hello! This is a test of the Qwen3 text to speech system.",
    speaker: str = "Ryan",
    language: str = "English",
    instruct: str = "",
) -> QueryResult:
    """Build CustomVoice inputs - generate speech with a predefined speaker."""
    return QueryResult(
        inputs={
            "prompt": build_prompt(text),
            "additional_information": {
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "instruct": [instruct],
                "max_new_tokens": [2048],
            },
        },
        model_name=TASK_MODELS["CustomVoice"],
    )


def get_voice_design_query(
    text: str = "Hello! This is a test of the Qwen3 text to speech system.",
    instruct: str = "Speak in a warm, friendly tone with clear enunciation.",
    language: str = "English",
) -> QueryResult:
    """Build VoiceDesign inputs - generate speech with a voice designed from description."""
    return QueryResult(
        inputs={
            "prompt": build_prompt(text),
            "additional_information": {
                "task_type": ["VoiceDesign"],
                "text": [text],
                "language": [language],
                "instruct": [instruct],
                "max_new_tokens": [2048],
                "non_streaming_mode": [True],
            },
        },
        model_name=TASK_MODELS["VoiceDesign"],
    )


def get_base_query(
    text: str = "Hello! This is a test of voice cloning with Qwen3 TTS.",
    ref_audio: str = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav",
    ref_text: str = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    language: str = "Auto",
) -> QueryResult:
    """Build Base inputs - voice cloning using reference audio."""
    return QueryResult(
        inputs={
            "prompt": build_prompt(text),
            "additional_information": {
                "task_type": ["Base"],
                "ref_audio": [ref_audio],
                "ref_text": [ref_text],
                "text": [text],
                "language": [language],
                "x_vector_only_mode": [False],
                "max_new_tokens": [2048],
            },
        },
        model_name=TASK_MODELS["Base"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Text-to-Speech Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task Types:
  CustomVoice  - Predefined speaker voices (default)
  VoiceDesign  - Design voice from natural language description
  Base         - Voice cloning from reference audio

Available Speakers (CustomVoice):
  English: Ryan, Aiden
  Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
  Japanese: Ono_Anna
  Korean: Sohee
        """,
    )

    parser.add_argument(
        "--task", "-t",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        default="CustomVoice",
        help="Task type (default: CustomVoice)",
    )
    parser.add_argument(
        "--text",
        default="Hello! This is a test of the Qwen3 text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speaker", "-s",
        default="Ryan",
        help="Speaker name for CustomVoice (default: Ryan)",
    )
    parser.add_argument(
        "--language", "-l",
        default="English",
        help="Language (default: English)",
    )
    parser.add_argument(
        "--instruct", "-i",
        default="",
        help="Style instruction (for CustomVoice/VoiceDesign)",
    )
    parser.add_argument(
        "--ref-audio",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav",
        help="Reference audio URL/path for Base (voice cloning)",
    )
    parser.add_argument(
        "--ref-text",
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        help="Transcript of reference audio for Base",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )

    args = parser.parse_args()

    # Build query based on task
    if args.task == "CustomVoice":
        query = get_custom_voice_query(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
        )
    elif args.task == "VoiceDesign":
        instruct = args.instruct or "Speak in a warm, friendly tone with clear enunciation."
        query = get_voice_design_query(
            text=args.text,
            instruct=instruct,
            language=args.language,
        )
    else:  # Base
        query = get_base_query(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
        )

    print(f"🎤 Qwen3-TTS Demo")
    print(f"   Task: {args.task}")
    print(f"   Model: {query.model_name}")
    print(f"   Text: {args.text[:50]}{'...' if len(args.text) > 50 else ''}")
    print()
    print("⏳ Loading model (this may take a minute on first run)...")

    # Initialize Omni
    omni = Omni(
        model=query.model_name,
        stage_init_timeout=300,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )

    print("🔊 Generating speech...")

    # Generate audio
    for stage_outputs in omni.generate(query.inputs, [sampling_params]):
        for output in stage_outputs.request_output:
            audio_tensor = output.multimodal_output["audio"]
            sample_rate = output.multimodal_output["sr"].item()

            # Convert to numpy
            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            # Save
            sf.write(args.output, audio_numpy, samplerate=sample_rate, format="WAV")
            print(f"✅ Saved audio to: {args.output}")


if __name__ == "__main__":
    main()
