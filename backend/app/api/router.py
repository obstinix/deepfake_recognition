from fastapi import APIRouter
from app.api.endpoints import analyze, models_endpoint, health

router = APIRouter()
router.include_router(health.router)
router.include_router(analyze.router)
router.include_router(models_endpoint.router)
