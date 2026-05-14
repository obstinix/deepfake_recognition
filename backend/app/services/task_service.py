import json
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

TASK_TTL = 60 * 60 * 24  # 24 hours

# Lazy connection — initialized on first use, not at import time
_redis_client = None
_fallback_store: dict[str, str] = {}
_using_fallback = False


def _get_redis():
    """Get Redis client, falling back to in-memory dict if unavailable."""
    global _redis_client, _using_fallback
    if _redis_client is not None:
        return _redis_client
    if _using_fallback:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), using in-memory task store")
        _using_fallback = True
        return None


def set_task(task_id: str, data: dict) -> None:
    key = f"task:{task_id}"
    value = json.dumps(data)
    client = _get_redis()
    if client:
        client.setex(key, TASK_TTL, value)
    else:
        _fallback_store[key] = value


def get_task(task_id: str) -> Optional[dict]:
    key = f"task:{task_id}"
    client = _get_redis()
    if client:
        raw = client.get(key)
    else:
        raw = _fallback_store.get(key)
    return json.loads(raw) if raw else None


def update_task_status(task_id: str, status: str, **kwargs) -> None:
    task = get_task(task_id) or {}
    task.update({"status": status, **kwargs})
    set_task(task_id, task)
