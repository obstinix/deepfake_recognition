# Deepfake Recognition Training Pipeline

This directory contains the machine learning pipeline for training, evaluating, and preprocessing data for the deepfake detection ensemble (ResNet18, EfficientNet-B3, ViT).

## Requirements

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## 1. Preprocessing Data

Place your raw image or video data in `data/raw/real/` and `data/raw/fake/`. The preprocessing script will split them into train, validation, and test sets, and automatically extract frames from `.mp4` files.

```bash
python scripts/preprocess_data.py --raw_dir data/raw --processed_dir data --splits 0.7 0.15 0.15
```

## 2. Training

Modify `configs/default.yaml` to adjust hyperparameters, batch size, learning rate, or to select which models to train.

```bash
python scripts/train.py --config configs/default.yaml --data_dir data
```

Checkpoints will be automatically saved to `models/checkpoints/`.

## 3. Evaluation

To evaluate a specific trained model checkpoint on the test set:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --data_dir data \
  --model resnet18 \
  --checkpoint ../models/checkpoints/resnet18_best.pth
```
