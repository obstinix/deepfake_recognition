"""Model architectures for deepfake detection."""

from deepfake_recognition.models.resnet import DeepfakeResNet18
from deepfake_recognition.models.efficientnet import DeepfakeEfficientNetB3
from deepfake_recognition.models.vit import DeepfakeViT
from deepfake_recognition.models.ensemble import DeepfakeEnsemble

__all__ = [
    "DeepfakeResNet18",
    "DeepfakeEfficientNetB3",
    "DeepfakeViT",
    "DeepfakeEnsemble",
]
