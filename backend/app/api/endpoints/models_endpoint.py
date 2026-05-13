from fastapi import APIRouter

router = APIRouter(prefix="/models", tags=["models"])

AVAILABLE_MODELS = [
    {"name": "resnet18", "version": "1.0.0", "accuracy": 0.92, "latency_ms": 50, "is_active": True},
    {"name": "efficientnet_b3", "version": "1.0.0", "accuracy": 0.94, "latency_ms": 100, "is_active": True},
    {"name": "vit_base", "version": "1.0.0", "accuracy": 0.91, "latency_ms": 150, "is_active": True},
    {"name": "ensemble", "version": "1.0.0", "accuracy": 0.95, "latency_ms": 120, "is_active": True},
]


@router.get("")
async def list_models():
    return {"models": AVAILABLE_MODELS}
