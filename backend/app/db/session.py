from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.db.models import Base
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy engine — initialized on first use, not at import time
_engine = None
_SessionLocal = None


def _get_engine():
    """Create engine lazily, falling back to SQLite if PostgreSQL is unavailable."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        # Verify connection
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL")
    except Exception as e:
        fallback_url = "sqlite:///./deepfake_dev.db"
        if logger:
            logger.warning(f"PostgreSQL unavailable ({e}), using SQLite fallback")
        _engine = create_engine(fallback_url)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


def create_tables():
    Base.metadata.create_all(bind=_get_engine())


def get_db():
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
