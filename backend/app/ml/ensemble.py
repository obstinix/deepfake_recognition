import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple
import torch.nn as nn
from app.utils.logger import get_logger

logger = get_logger(__name__)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)


def ensemble_predict(
    models: Dict[str, nn.Module],
    image_path: str,
    device: torch.device
) -> Dict:
    """Run ensemble prediction and return verdict + confidence."""
    img_tensor = preprocess_image(image_path).to(device)

    all_probs = []
    with torch.no_grad():
        for name, model in models.items():
            try:
                out = model(img_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
                logger.info(f"{name}: real={probs[0]:.3f} fake={probs[1]:.3f}")
            except Exception as e:
                logger.error(f"Model {name} inference failed: {e}")

    if not all_probs:
        raise RuntimeError("All models failed during inference")

    avg_probs = np.mean(all_probs, axis=0)
    verdict = "fake" if avg_probs[1] > 0.5 else "real"
    confidence = float(max(avg_probs))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_real": float(avg_probs[0]),
        "confidence_fake": float(avg_probs[1]),
        "models_used": list(models.keys()),
    }
