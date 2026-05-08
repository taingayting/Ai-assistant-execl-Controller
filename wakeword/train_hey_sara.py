"""Train a custom "Hey Sara" wake-word model using openwakeword.

This script generates synthetic training clips and trains a small
.onnx model that WakeDetector can load.

Prerequisites:
    pip install openwakeword[full]   # includes training dependencies

Usage:
    cd d:\\year4\\speech_to_text
    python wakeword/train_hey_sara.py

Output:
    wakeword/models/hey_sara.onnx

After training, set this env var (or update .env):
    WAKE_MODEL_PATH=wakeword/models/hey_sara.onnx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "wakeword" / "models"
MODEL_OUTPUT = MODEL_DIR / "hey_sara.onnx"


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from openwakeword.train import train_model
    except ImportError:
        print(
            "ERROR: openwakeword training extras not installed.\n"
            "Run:  pip install openwakeword[full]\n"
            "Then retry this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[train_hey_sara] Training 'Hey Sara' wake-word model...")
    print(f"[train_hey_sara] Output: {MODEL_OUTPUT}")

    # openwakeword's train_model accepts:
    # - target_phrase: the wake phrase
    # - output_path: where to save the .onnx/.tflite
    # - n_samples: number of synthetic clips to generate (more = better accuracy)
    # - epochs: training epochs
    # - n_background: number of negative (noise) samples
    try:
        train_model(
            target_phrase="hey sara",
            output_path=str(MODEL_OUTPUT),
            n_samples=5000,
            n_background=5000,
            epochs=50,
        )
        print(f"[train_hey_sara] Model saved to {MODEL_OUTPUT}")
        print(f"[train_hey_sara] Set WAKE_MODEL_PATH={MODEL_OUTPUT}")
    except Exception as exc:
        print(f"[train_hey_sara] Training failed: {exc}", file=sys.stderr)
        print(
            "\n[train_hey_sara] ALTERNATIVE: You can use the pre-built 'hey_jarvis' model\n"
            "while you work on training data. Just set WAKE_MODEL_PATH to empty\n"
            "and Sara will use 'hey_jarvis' as a demo wake word.\n"
            "\nOr download a pre-trained model from:\n"
            "  https://github.com/dscripka/openwakeword#pre-trained-models",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    train()
