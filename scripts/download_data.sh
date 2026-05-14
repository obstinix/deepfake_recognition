#!/usr/bin/env bash
# Helper to download FaceForensics++ (requires user to have access)
# See: https://github.com/ondyari/FaceForensics

set -euo pipefail

echo "=== Deepfake Dataset Download Helper ==="
echo ""
echo "FaceForensics++ (recommended):"
echo "  1. Request access at: https://github.com/ondyari/FaceForensics"
echo "  2. Run: python faceforensics_download_v4.py <output_dir> -d all -c c23 -t videos"
echo ""
echo "Celeb-DF-v2:"
echo "  https://github.com/yuezunli/celeb-deepfakeforensics"
echo ""
echo "Expected directory structure after download:"
echo "  data/"
echo "    real/    (real face images)"
echo "    fake/    (deepfake images)"
echo ""
echo "For video datasets, run frame extraction first:"
echo "  python -c 'from deepfake_recognition.inference.video import extract_frames; ...'"
