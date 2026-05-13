from pydantic import BaseModel
from typing import Optional


class AnalysisOptions(BaseModel):
    detailed_analysis: bool = True
    return_heatmap: bool = True
    frame_sampling: int = 10  # Every N frames for video


class BatchAnalysisRequest(BaseModel):
    callback_url: Optional[str] = None
