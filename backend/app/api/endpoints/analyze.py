import uuid
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from app.services.file_handler import save_upload, get_file_type
from app.services.task_service import set_task, get_task
from app.services.inference_service import run_inference
from app.schemas.response import TaskResponse, SubmitResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=SubmitResponse)
async def submit(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    task_id = str(uuid.uuid4())
    try:
        file_path = await save_upload(file, task_id)
        file_type = get_file_type(file.content_type or "image/jpeg")
        set_task(task_id, {
            "task_id": task_id,
            "filename": file.filename,
            "status": "processing",
            "file_type": file_type
        })
        background_tasks.add_task(run_inference, task_id, file_path, file_type)
        return SubmitResponse(
            task_id=task_id,
            status="processing",
            filename=file.filename or "upload",
            message="File received. Poll /analyze/{task_id} for results."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit error: {e}")
        raise HTTPException(500, "Internal server error")


@router.get("/{task_id}", response_model=dict)
async def get_result(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task
