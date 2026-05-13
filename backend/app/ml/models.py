import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Dict
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b3(num_classes: int = 2) -> nn.Module:
    try:
        import timm
        model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
        return model
    except Exception as e:
        logger.warning(f"Could not load EfficientNet: {e}. Falling back to ResNet50.")
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


def build_vit(num_classes: int = 2) -> nn.Module:
    try:
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
        return model
    except Exception as e:
        logger.warning(f"Could not load ViT: {e}. Falling back to ResNet18.")
        return build_resnet18(num_classes)


def load_ensemble(device: torch.device) -> Dict[str, nn.Module]:
    """Load all three models onto device."""
    logger.info(f"Loading ensemble models on {device}")
    models = {
        "resnet18": build_resnet18().to(device).eval(),
        "efficientnet_b3": build_efficientnet_b3().to(device).eval(),
        "vit_base": build_vit().to(device).eval(),
    }
    logger.info(f"Ensemble loaded: {list(models.keys())}")
    return models
