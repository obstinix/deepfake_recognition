#!/usr/bin/env python3
"""
Dataset download helper and frame extractor.

Usage examples:
  # Extract frames from FF++ (after manual download)
  python scripts/download_faceforensics.py extract-frames \
      --source data/FaceForensics --output data/ff_frames --fps 1

  # Extract frames from Celeb-DF-v2
  python scripts/download_faceforensics.py extract-celebdf \
      --source data/Celeb-DF-v2 --output data/celebdf_frames --fps 1

  # Extract frames from DFDC
  python scripts/download_faceforensics.py extract-dfdc \
      --source data/DFDC --output data/dfdc_frames --fps 1

  # Verify a dataset folder structure
  python scripts/download_faceforensics.py verify --path data/ff_frames
"""

import argparse
import cv2
import json
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path: Path, output_dir: Path,
                                fps: float = 1.0, max_frames: int = 32) -> int:
    """
    Extract frames from a video file at given FPS.
    Returns number of frames extracted.
    Saves as: output_dir/VIDEONAME_NNNN.jpg
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(video_fps / fps))
    count = 0
    saved = 0
    stem = video_path.stem

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = output_dir / f"{stem}_{saved:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        count += 1

    cap.release()
    return saved


def extract_ff_frames(source: Path, output: Path, fps: float):
    """Extract frames from FaceForensics++ directory structure."""
    real_dirs = list((source / "original_sequences").rglob("videos"))
    fake_dirs = list((source / "manipulated_sequences").rglob("videos"))

    print(f"Found {len(real_dirs)} real dirs, {len(fake_dirs)} fake dirs")

    # Real frames
    real_out = output / "real"
    for d in tqdm(real_dirs, desc="Real videos"):
        for vid in d.glob("*.mp4"):
            extract_frames_from_video(vid, real_out, fps=fps)

    # Fake frames
    fake_out = output / "fake"
    for d in tqdm(fake_dirs, desc="Fake videos"):
        for vid in d.glob("*.mp4"):
            extract_frames_from_video(vid, fake_out, fps=fps)

    _print_stats(output)


def extract_celebdf_frames(source: Path, output: Path, fps: float):
    """Extract frames from Celeb-DF-v2."""
    real_out = output / "real"
    fake_out = output / "fake"

    real_dirs = ["Celeb-real", "YouTube-real"]
    fake_dirs = ["Celeb-synthesis"]

    for d in real_dirs:
        p = source / d
        if not p.exists():
            continue
        for vid in tqdm(list(p.glob("*.mp4")), desc=f"Real ({d})"):
            extract_frames_from_video(vid, real_out, fps=fps)

    for d in fake_dirs:
        p = source / d
        if not p.exists():
            continue
        for vid in tqdm(list(p.glob("*.mp4")), desc=f"Fake ({d})"):
            extract_frames_from_video(vid, fake_out, fps=fps)

    _print_stats(output)


def extract_dfdc_frames(source: Path, output: Path, fps: float):
    """Extract frames from DFDC using metadata.json for labels."""
    meta_path = source / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {source}")

    with open(meta_path) as f:
        metadata = json.load(f)

    videos_dir = source / "videos"
    real_out = output / "real"
    fake_out = output / "fake"

    for fname, info in tqdm(metadata.items(), desc="DFDC videos"):
        vid_path = videos_dir / fname
        if not vid_path.exists():
            continue
        label = info.get("label", "").upper()
        out_dir = real_out if label == "REAL" else fake_out
        extract_frames_from_video(vid_path, out_dir, fps=fps)

    _print_stats(output)


def verify_dataset(path: Path):
    """Verify a prepared dataset directory has real/ and fake/ with images."""
    real_count = len(list((path / "real").glob("*.jpg"))) if (path / "real").exists() else 0
    fake_count = len(list((path / "fake").glob("*.jpg"))) if (path / "fake").exists() else 0
    total = real_count + fake_count
    balance = real_count / total if total > 0 else 0

    print(f"\nDataset at: {path}")
    print(f"  Real images : {real_count:,}")
    print(f"  Fake images : {fake_count:,}")
    print(f"  Total       : {total:,}")
    print(f"  Balance     : {balance:.1%} real / {1-balance:.1%} fake")
    if balance < 0.3 or balance > 0.7:
        print("  WARNING: Dataset is imbalanced. Trainer will use class weights.")
    else:
        print("  OK: Dataset is reasonably balanced.")


def _print_stats(output: Path):
    print("\nExtraction complete.")
    verify_dataset(output)


def main():
    parser = argparse.ArgumentParser(description="Deepfake dataset frame extractor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ff = sub.add_parser("extract-frames", help="Extract from FaceForensics++")
    p_ff.add_argument("--source", required=True, type=Path)
    p_ff.add_argument("--output", required=True, type=Path)
    p_ff.add_argument("--fps", type=float, default=1.0)

    p_cd = sub.add_parser("extract-celebdf", help="Extract from Celeb-DF-v2")
    p_cd.add_argument("--source", required=True, type=Path)
    p_cd.add_argument("--output", required=True, type=Path)
    p_cd.add_argument("--fps", type=float, default=1.0)

    p_df = sub.add_parser("extract-dfdc", help="Extract from DFDC")
    p_df.add_argument("--source", required=True, type=Path)
    p_df.add_argument("--output", required=True, type=Path)
    p_df.add_argument("--fps", type=float, default=1.0)

    p_v = sub.add_parser("verify", help="Verify prepared dataset")
    p_v.add_argument("--path", required=True, type=Path)

    args = parser.parse_args()

    if args.cmd == "extract-frames":
        extract_ff_frames(args.source, args.output, args.fps)
    elif args.cmd == "extract-celebdf":
        extract_celebdf_frames(args.source, args.output, args.fps)
    elif args.cmd == "extract-dfdc":
        extract_dfdc_frames(args.source, args.output, args.fps)
    elif args.cmd == "verify":
        verify_dataset(args.path)


if __name__ == "__main__":
    main()
