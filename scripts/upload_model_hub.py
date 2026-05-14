#!/usr/bin/env python3
"""Upload trained checkpoint to HuggingFace Hub.

Usage:
  python scripts/upload_model_hub.py --checkpoint checkpoints/resnet18_best.pth \
                                      --repo-id obstinix/deepfake-resnet18 \
                                      --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
from pathlib import Path


def upload(checkpoint_path: str, repo_id: str, token: str) -> None:
    """Upload a checkpoint file to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=f"checkpoints/{Path(checkpoint_path).name}",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {checkpoint_path} to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    upload(args.checkpoint, args.repo_id, token)


if __name__ == "__main__":
    main()
