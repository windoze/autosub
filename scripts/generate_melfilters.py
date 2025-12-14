#!/usr/bin/env python3
"""
Generate mel filter bank files for Whisper models.

This script generates the pre-computed mel filter banks used by Whisper
for converting FFT spectrograms to mel-scale spectrograms.

Requirements:
    pip install openai-whisper numpy

Usage:
    python scripts/generate_melfilters.py

Output:
    src/melfilters.bytes    - 80 mel bins (for tiny, base, small, medium models)
    src/melfilters128.bytes - 128 mel bins (for large-v3 model)
"""

import numpy as np

try:
    from whisper.audio import mel_filters
except ImportError:
    print("Error: openai-whisper is required")
    print("Install with: pip install openai-whisper")
    exit(1)


def generate_mel_filters(n_mels: int, output_path: str):
    """Generate mel filter bank and save as raw f32 bytes."""
    # Get mel filters from whisper (returns torch tensor)
    filters = mel_filters(device="cpu", n_mels=n_mels)

    # Convert to numpy float32 array
    filters_np = filters.numpy().astype(np.float32)

    # Save as raw little-endian bytes
    filters_np.tofile(output_path)

    print(f"Generated {output_path}")
    print(f"  Shape: {filters_np.shape}")
    print(f"  Size: {filters_np.nbytes} bytes")


def main():
    # 80 mel bins - used by tiny, base, small, medium models
    generate_mel_filters(80, "src/melfilters.bytes")

    # 128 mel bins - used by large-v3 model
    generate_mel_filters(128, "src/melfilters128.bytes")

    print("\nDone! Mel filter files have been generated.")


if __name__ == "__main__":
    main()
