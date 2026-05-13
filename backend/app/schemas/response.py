from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class FrameAnalysis(BaseModel):
    frame_num: int
    confidence: float
    verdict: str


class AnalysisResult(BaseModel):
    verdict: str
    confidence: float
    confidence_real: float
    confidence_fake: float
    heatmap_data: Optional[str] = None
    frame_analysis: Optional[List[FrameAnalysis]] = None
    processing_time_ms: int
    models_used: List[str]


class TaskResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    progress: Optional[int] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None


class SubmitResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    message: str


class ModelInfo(BaseModel):
    name: str
    version: str
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    is_active: bool
