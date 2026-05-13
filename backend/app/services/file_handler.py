import os
import aiofiles
from fastapi import UploadFile, HTTPException
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "video/mp4", "video/quicktime", "video/avi"
}
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


async def save_upload(file: UploadFile, task_id: str) -> str:
    """Validate, save, and return path for uploaded file."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(400, f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")

    ext = os.path.splitext(file.filename or "file")[1] or ".bin"
    dest_dir = os.path.join(settings.UPLOAD_DIR, task_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"input{ext}")

    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(content)

    logger.info(f"Saved upload to {dest_path} ({len(content)} bytes)")
    return dest_path


def get_file_type(content_type: str) -> str:
    return "video" if content_type.startswith("video/") else "image"
