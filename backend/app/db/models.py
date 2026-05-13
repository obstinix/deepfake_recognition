from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, JSON, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(20))  # 'image' | 'video'
    status = Column(String(20), default="processing")  # processing | completed | failed

    # Results
    verdict = Column(String(10))  # 'real' | 'fake'
    confidence = Column(Float)
    confidence_real = Column(Float)
    confidence_fake = Column(Float)

    # Details
    frame_analysis = Column(JSON)
    heatmap_path = Column(String(512))
    error_message = Column(Text)

    # Meta
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    processing_time_ms = Column(Integer)
    models_used = Column(JSON)

    __table_args__ = (
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_created_at", "created_at"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    accuracy = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    model_path = Column(String(512))
    framework = Column(String(50), default="pytorch")
    input_size = Column(Integer, default=224)
    avg_latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    tags = Column(JSON, default=list)
