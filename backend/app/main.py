from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.router import router
from app.middleware.monitoring import add_monitoring, metrics_app
from app.db.session import create_tables
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(add_monitoring)
app.mount("/metrics", metrics_app)
app.include_router(router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup():
    create_tables()
    logger.info(f"Deepfake Recognition API started — env={settings.APP_ENV}")


@app.get("/")
async def root():
    return {"message": "Deepfake Recognition API", "docs": "/docs", "health": "/health"}
