from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    APP_SECRET_KEY: str = "dev-secret-key-change-in-production"
    API_PREFIX: str = "/api/v1"
    API_TITLE: str = "Deepfake Recognition API"
    API_VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = "postgresql://deepfake_user:deepfake_pass@localhost:5432/deepfake_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"


    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 100
    UPLOAD_DIR: str = "./uploads"

    # ML
    MODEL_DIR: str = "./models/checkpoints"
    DEVICE: str = "cpu"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
