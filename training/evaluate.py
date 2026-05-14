#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the test split.

Usage:
  python training/evaluate.py --checkpoint checkpoints/resnet18_best.pth \
                               --data /path/to/dataset --split test

Placeholder — full implementation in Phase 3.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to dataset root directory")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu, mps")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data} (split={args.split})")
    print("Full evaluation logic will be implemented in Phase 3.")


if __name__ == "__main__":
    main()
