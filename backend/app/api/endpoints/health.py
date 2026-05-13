from fastapi import APIRouter
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "env": settings.APP_ENV
    }
