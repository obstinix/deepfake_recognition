#!/usr/bin/env python3
"""
Entry point for training deepfake detection models.

Usage:
  python training/train.py --config training/configs/resnet18.yaml --data /path/to/dataset
  python training/train.py --config training/configs/efficientnet_b3.yaml --data /path/to/dataset --resume checkpoints/last.pth

Placeholder — full implementation in Phase 3.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deepfake detection models")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", required=True, help="Path to dataset root directory")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu, mps")
    parser.add_argument("--wandb-offline", action="store_true", help="Run wandb in offline mode")
    args = parser.parse_args()

    print(f"Training with config: {args.config}")
    print(f"Dataset: {args.data}")
    print("Full training loop will be implemented in Phase 3.")


if __name__ == "__main__":
    main()
