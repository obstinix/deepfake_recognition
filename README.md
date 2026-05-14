# 🧠 Deepfake Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![React](https://img.shields.io/badge/React-18-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade deepfake detection system using an ensemble of fine-tuned deep learning models (ResNet-18, EfficientNet-B3, ViT-Base), a FastAPI backend, and a modern React frontend.

## Architecture

```
User → React Frontend → FastAPI (BackgroundTasks) → ML Ensemble → Results
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/obstinix/deepfake_recognition.git
cd deepfake_recognition
cp .env.example .env

# Run with Docker
docker-compose up -d

# Access
# Frontend: http://localhost:3000
# API:      http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Project Structure

| Directory | Description |
|-----------|-------------|
| `backend/` | FastAPI application, models, services |
| `frontend/` | React + TypeScript web interface |
| `training/` | Model training scripts and utilities |
| `models/` | Trained model checkpoints |
| `k8s/` | Kubernetes deployment manifests |

## Model Performance

| Model | Accuracy | Latency |
|-------|----------|---------|
| ResNet-18 (fine-tuned) | 92% | 50ms |
| EfficientNet-B3 (fine-tuned) | 94% | 100ms |
| ViT-Base (fine-tuned) | 91% | 150ms |
| **Ensemble** | **95%+** | **120ms** |

## Documentation

- [Backend README](./backend/README.md)
- [Frontend README](./frontend/README.md)
- [Training README](./training/README.md)
