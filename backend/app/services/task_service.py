import json
import redis
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
TASK_TTL = 60 * 60 * 24  # 24 hours


def set_task(task_id: str, data: dict) -> None:
    _redis.setex(f"task:{task_id}", TASK_TTL, json.dumps(data))


def get_task(task_id: str) -> dict | None:
    raw = _redis.get(f"task:{task_id}")
    return json.loads(raw) if raw else None


def update_task_status(task_id: str, status: str, **kwargs) -> None:
    task = get_task(task_id) or {}
    task.update({"status": status, **kwargs})
    set_task(task_id, task)
