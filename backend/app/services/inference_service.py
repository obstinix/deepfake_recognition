import time
import torch
from app.config import settings
from app.ml.models import load_ensemble
from app.ml.ensemble import ensemble_predict
from app.ml.heatmap import generate_gradcam
from app.services.task_service import update_task_status
from app.utils.logger import get_logger

logger = get_logger(__name__)

_device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
_models = None


def get_models():
    global _models
    if _models is None:
        _models = load_ensemble(_device)
    return _models


def run_inference(task_id: str, file_path: str, file_type: str) -> None:
    """Main inference entry point — called by Celery worker."""
    start = time.time()
    try:
        update_task_status(task_id, "processing")
        models = get_models()
        result = ensemble_predict(models, file_path, _device)

        # Heatmap (use first model)
        first_model = next(iter(models.values()))
        heatmap = generate_gradcam(first_model, file_path, _device)

        elapsed_ms = int((time.time() - start) * 1000)
        update_task_status(
            task_id,
            "completed",
            result={
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "confidence_real": result["confidence_real"],
                "confidence_fake": result["confidence_fake"],
                "heatmap_data": heatmap,
                "models_used": result["models_used"],
                "processing_time_ms": elapsed_ms,
            }
        )
        logger.info(f"Task {task_id} completed in {elapsed_ms}ms → {result['verdict']}")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        update_task_status(task_id, "failed", error_message=str(e))
