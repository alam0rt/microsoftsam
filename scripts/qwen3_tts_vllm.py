# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm==0.14.0",
#     "vllm-omni>=0.14.0",
#     "torch>=2.4.0",
#     "soundfile",
# ]
# [tool.uv]
# find-links = ["https://download.pytorch.org/whl/cu124"]
# extra-index-url = ["https://download.pytorch.org/whl/cu124"]
# ///
"""
Qwen3-TTS vLLM-Omni Offline Inference

Generate speech using Qwen3-TTS models via vLLM-Omni.
Supports CustomVoice (predefined speakers), VoiceDesign (natural language),
and Base (voice cloning) tasks.

Based on: https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts

Requirements:
- NVIDIA GPU with CUDA support (8GB+ VRAM for 0.6B, 16GB+ for 1.7B)

Usage:
    uv run scripts/qwen3_tts_vllm.py [--task TASK] [--text TEXT] [--output FILE]

Examples:
    # CustomVoice with predefined speaker
    uv run scripts/qwen3_tts_vllm.py --task CustomVoice --text "Hello world" --speaker Vivian

    # Voice cloning (Base model)
    uv run scripts/qwen3_tts_vllm.py --task Base --text "Hello world" --ref-audio reference.wav

    # VoiceDesign with natural language description
    uv run scripts/qwen3_tts_vllm.py --task VoiceDesign --text "Hello" --instruct "Young female voice"

With Nix:
    nix develop -c uv run scripts/qwen3_tts_vllm.py --task CustomVoice --text "Hello"
"""

import argparse
import os
import sys
from typing import NamedTuple

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""
    inputs: dict | list[dict]
    model_name: str


def build_custom_voice_query(
    text: str,
    language: str = "Auto",
    speaker: str = "Vivian",
    instruct: str = "",
    use_small_model: bool = True,
) -> QueryResult:
    """Build CustomVoice query."""
    task_type = "CustomVoice"
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = {
        "prompt": prompt,
        "additional_information": {
            "task_type": [task_type],
            "text": [text],
            "language": [language],
            "speaker": [speaker],
            "instruct": [instruct],
            "max_new_tokens": [2048],
        },
    }
    model = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" if use_small_model else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    return QueryResult(inputs=inputs, model_name=model)


def build_voice_design_query(
    text: str,
    language: str = "Auto",
    instruct: str = "A warm, friendly female voice",
) -> QueryResult:
    """Build VoiceDesign query (only 1.7B available)."""
    task_type = "VoiceDesign"
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = {
        "prompt": prompt,
        "additional_information": {
            "task_type": [task_type],
            "text": [text],
            "language": [language],
            "instruct": [instruct],
            "max_new_tokens": [2048],
            "non_streaming_mode": [True],
        },
    }
    # VoiceDesign only available in 1.7B
    return QueryResult(inputs=inputs, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")


def build_base_query(
    text: str,
    ref_audio: str,
    ref_text: str = "",
    language: str = "Auto",
    x_vector_only: bool = False,
    use_small_model: bool = True,
) -> QueryResult:
    """Build Base (voice clone) query."""
    task_type = "Base"
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = {
        "prompt": prompt,
        "additional_information": {
            "task_type": [task_type],
            "ref_audio": [ref_audio],
            "ref_text": [ref_text],
            "text": [text],
            "language": [language],
            "x_vector_only_mode": [x_vector_only],
            "max_new_tokens": [2048],
        },
    }
    model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if use_small_model else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    return QueryResult(inputs=inputs, model_name=model)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Offline Inference via vLLM-Omni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task Types:
  CustomVoice  - Predefined speaker voices (0.6B/1.7B)
  VoiceDesign  - Natural language voice description (1.7B only)
  Base         - Voice cloning from reference audio (0.6B/1.7B)

Available Speakers (CustomVoice):
  Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
  English: Ryan, Aiden
  Japanese: Ono_Anna
  Korean: Sohee
        """,
    )

    parser.add_argument("--task", choices=["CustomVoice", "VoiceDesign", "Base"],
                        default="CustomVoice", help="Task type (default: CustomVoice)")
    parser.add_argument("--text", default="Hello, this is a test of Qwen3 TTS.",
                        help="Text to synthesize")
    parser.add_argument("--language", default="Auto",
                        help="Language: Auto, Chinese, English, Japanese, Korean, etc.")
    parser.add_argument("--speaker", default="Vivian",
                        help="Speaker name for CustomVoice")
    parser.add_argument("--instruct", default="",
                        help="Style instruction for CustomVoice/VoiceDesign")
    parser.add_argument("--ref-audio", help="Reference audio for voice cloning (Base task)")
    parser.add_argument("--ref-text", default="", help="Reference audio transcript (Base task)")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument("--large-model", action="store_true",
                        help="Use 1.7B model instead of 0.6B")

    args = parser.parse_args()

    # Build query based on task
    use_small = not args.large_model

    if args.task == "CustomVoice":
        query = build_custom_voice_query(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            use_small_model=use_small,
        )
    elif args.task == "VoiceDesign":
        if not args.instruct:
            args.instruct = "A warm, friendly voice"
        query = build_voice_design_query(
            text=args.text,
            language=args.language,
            instruct=args.instruct,
        )
    elif args.task == "Base":
        if not args.ref_audio:
            print("Error: --ref-audio is required for Base (voice clone) task")
            sys.exit(1)
        query = build_base_query(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            use_small_model=use_small,
        )

    print("🚀 Qwen3-TTS via vLLM-Omni")
    print(f"   Model: {query.model_name}")
    print(f"   Task: {args.task}")
    print(f"   Text: {args.text[:50]}{'...' if len(args.text) > 50 else ''}")
    print()

    # Import vLLM-Omni
    try:
        from vllm import SamplingParams
        from vllm_omni import Omni
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Install with: uv pip install vllm vllm-omni")
        sys.exit(1)

    # Initialize Omni
    omni = Omni(
        model=query.model_name,
        stage_init_timeout=300,
    )

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )

    # Generate
    print("⏳ Generating speech...")
    import soundfile as sf

    omni_generator = omni.generate(query.inputs, [sampling_params])
    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            audio_tensor = output.multimodal_output["audio"]
            audio_samplerate = output.multimodal_output["sr"].item()

            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            sf.write(args.output, audio_numpy, samplerate=audio_samplerate, format="WAV")
            print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
