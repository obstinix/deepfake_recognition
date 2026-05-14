# 🧠 Deepfake Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade deepfake detection system using fine-tuned deep learning models (ResNet-18, EfficientNet-B3), complete with a training pipeline, dataset downloaders, and a FastAPI inference server.

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/obstinix/deepfake_recognition.git
cd deepfake_recognition

# 2. Setup environment
pip install -r requirements.txt
cp .env.example .env

# 3. Download synthetic dataset for smoke-testing
python scripts/download_dataset.py synthetic --output data/frames --n-images 400

# 4. Train a fast model
python training/train.py --config training/configs/resnet18.yaml --data data/frames --max-samples 100

# 5. Start API server
uvicorn api.main:app --reload
```

## 📊 Model Performance

*Model performance will be updated once fully trained on a large dataset like FaceForensics++ or DFDC.*

| Model | Accuracy | Latency |
|-------|----------|---------|
| ResNet-18 (fine-tuned) | TBD | ~50ms |
| EfficientNet-B3 (fine-tuned) | TBD | ~100ms |
| **Ensemble** | **TBD** | **~120ms** |

## 💾 Dataset

**Automated download (recommended):**
```bash
# Downloads best available dataset automatically
# Requires free Kaggle account: https://www.kaggle.com/settings → API → Create Token
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python scripts/download_dataset.py auto --output data/frames
```

**No Kaggle account? Use synthetic data for testing:**
```bash
python scripts/download_dataset.py synthetic --output data/frames --n-images 400
```

**Manual dataset options:**
| Dataset | Size | Requires | Command |
|---------|------|----------|---------|
| 140k Real/Fake Faces | ~2GB | Free Kaggle account | `kaggle-140k` |
| Real and Fake Detection | ~500MB | Free Kaggle account | `kaggle-real-fake` |
| DFDC Sample | ~10GB | Join free competition | `dfdc` |
| FaceForensics++ | ~100GB | Email approval | see scripts/download_faceforensics.py |

**Verify your dataset:**
```bash
python scripts/download_dataset.py verify --path data/frames
```

## 🏋️ Training

The project includes configs for multiple architectures.

```bash
# ResNet-18
python training/train.py --config training/configs/resnet18.yaml --data data/frames

# EfficientNet-B3
python training/train.py --config training/configs/efficientnet_b3.yaml --data data/frames

# Evaluate
python training/evaluate.py --checkpoint checkpoints/resnet18/best.pth --config training/configs/resnet18.yaml --data data/frames
```

## 🌐 Inference API

The FastAPI server provides endpoints for image and video prediction.

```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Predict Image
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Predict Video
curl -X POST "http://localhost:8000/predict/video?sample_frames=16" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_video.mp4"
```

## 🐳 Docker

Run the entire inference stack using Docker.

```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f api
```

## 🧪 Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run pytest
pytest tests/ -v
```

## 📁 Project Structure

```text
deepfake_recognition/
├── api/                   # FastAPI inference server
│   └── main.py
├── scripts/               # Utilities (dataset downloaders)
│   ├── download_dataset.py
│   └── download_faceforensics.py
├── src/
│   └── deepfake_recognition/
│       ├── data/          # PyTorch datasets and transforms
│       ├── inference/     # Predictor wrapper classes
│       ├── models/        # ResNet, EfficientNet definitions
│       └── training/      # Trainer loops, metrics, callbacks
├── tests/                 # Pytest suite
├── training/              # CLI entry points
│   ├── configs/           # YAML hyperparameters
│   ├── evaluate.py
│   └── train.py
├── .github/workflows/     # CI pipelines
├── Dockerfile
├── docker-compose.yml
└── requirements.txt       # Pinned dependencies
```
