import torch
import numpy as np
import cv2
import base64
from PIL import Image
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_gradcam(model, image_path: str, device: torch.device) -> Optional[str]:
    """Generate Grad-CAM heatmap. Returns base64 encoded PNG."""
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        # Find last conv layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            return None

        activations, gradients = [], []
        hooks = [
            target_layer.register_forward_hook(lambda m, i, o: activations.append(o)),
            target_layer.register_backward_hook(lambda m, gi, go: gradients.append(go[0]))
        ]

        output = model(img_tensor)
        model.zero_grad()
        output[0, output.argmax()].backward()

        for h in hooks:
            h.remove()

        if not activations or not gradients:
            return None

        act = activations[0].detach().cpu().numpy()[0]
        grad = gradients[0].detach().cpu().numpy()[0]
        weights = grad.mean(axis=(1, 2))
        cam = np.sum(weights[:, None, None] * act, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        orig = np.array(img_pil.resize((224, 224)))
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        return None
