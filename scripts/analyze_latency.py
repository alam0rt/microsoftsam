#!/usr/bin/env python3
"""Analyze latency logs and compute percentiles."""

import json
import sys
from pathlib import Path


def analyze(path: Path):
    """Analyze latency data from a JSONL file.

    Args:
        path: Path to the latency.jsonl file.
    """
    ttfa_values = []
    asr_values = []
    llm_values = []
    llm_ttft_values = []
    tts_values = []
    total_turn_values = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            metrics = data.get("metrics", {})
            if metrics.get("total_ttfa_ms"):
                ttfa_values.append(metrics["total_ttfa_ms"])
                asr_values.append(metrics.get("asr_ms", 0))
                llm_values.append(metrics.get("llm_total_ms", 0))
                llm_ttft_values.append(metrics.get("llm_ttft_ms", 0))
                tts_values.append(metrics.get("tts_ttfa_ms", 0))
                if metrics.get("total_turn_ms"):
                    total_turn_values.append(metrics["total_turn_ms"])

    if not ttfa_values:
        print("No latency data found")
        return

    def stats(values, name):
        if len(values) < 2:
            print(f"{name}: insufficient data ({len(values)} samples)")
            return

        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Manual percentile calculation for small datasets
        p50_idx = int(n * 0.50)
        p90_idx = int(n * 0.90)
        p99_idx = min(int(n * 0.99), n - 1)

        p50 = sorted_values[p50_idx]
        p90 = sorted_values[p90_idx]
        p99 = sorted_values[p99_idx]
        avg = sum(values) / len(values)

        print(f"{name}:")
        print(f"  avg: {avg:.0f}ms")
        print(f"  p50: {p50:.0f}ms")
        print(f"  p90: {p90:.0f}ms")
        print(f"  p99: {p99:.0f}ms")
        print(f"  min: {min(values):.0f}ms")
        print(f"  max: {max(values):.0f}ms")
        print()

    print("=" * 50)
    print(f"Latency Analysis: {path}")
    print(f"Analyzed {len(ttfa_values)} turns")
    print("=" * 50)
    print()

    stats(ttfa_values, "Total TTFA (Time-to-First-Audio)")
    stats(asr_values, "ASR (Speech-to-Text)")
    stats(llm_ttft_values, "LLM TTFT (Time-to-First-Token)")
    stats(llm_values, "LLM Total")
    stats(tts_values, "TTS TTFA (Time-to-First-Audio)")

    if total_turn_values:
        stats(total_turn_values, "Total Turn (end-to-end)")

    # Summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    avg_ttfa = sum(ttfa_values) / len(ttfa_values)
    avg_asr = sum(asr_values) / len(asr_values)
    avg_llm = sum(llm_values) / len(llm_values)
    avg_tts = sum(tts_values) / len(tts_values)

    print("\nAverage TTFA breakdown:")
    print(f"  ASR:  {avg_asr:6.0f}ms ({avg_asr/avg_ttfa*100:.1f}%)")
    print(f"  LLM:  {avg_llm:6.0f}ms ({avg_llm/avg_ttfa*100:.1f}%)")
    print(f"  TTS:  {avg_tts:6.0f}ms ({avg_tts/avg_ttfa*100:.1f}%)")
    print("  ------------------")
    print(f"  TTFA: {avg_ttfa:6.0f}ms")

    # Identify bottleneck
    bottleneck = max([("ASR", avg_asr), ("LLM", avg_llm), ("TTS", avg_tts)], key=lambda x: x[1])
    print(f"\n  Biggest bottleneck: {bottleneck[0]} ({bottleneck[1]:.0f}ms)")


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("latency.jsonl")

    if not path.exists():
        print(f"Error: {path} not found")
        print("Usage: python analyze_latency.py [path/to/latency.jsonl]")
        sys.exit(1)

    analyze(path)


if __name__ == "__main__":
    main()
