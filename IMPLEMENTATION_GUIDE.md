# Deepfake Recognition - Implementation Guide
**Quick Start & Code Templates**

---

## Table of Contents
1. [Repository Restructuring](#repository-restructuring)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Model Training Setup](#model-training-setup)
5. [Quick Code Examples](#quick-code-examples)
6. [Common Commands](#common-commands)

---

## Repository Restructuring

### Step 1: Clean Up Current Repository

```bash
# Clone the current repo
git clone https://github.com/obstinix/deepfake_recognition.git
cd deepfake_recognition

# Create new branches to preserve history
git checkout -b backup/old-structure
git push origin backup/old-structure

# Return to main
git checkout main

# Archive old files (don't delete, just move)
mkdir -p archive/notebooks archive/old-scripts
git mv *.ipynb archive/notebooks/
git mv app_enhanced.py trydeepfake.py server.py archive/old-scripts/
git mv index.html index.css archive/old-scripts/
git mv *.mp4.zip archive/

# Commit this cleanup
git add .
git commit -m "chore: archive legacy code, restructure for production"
```

### Step 2: Create New Directory Structure

```bash
# Create the new structure
mkdir -p backend/{app,tests}
mkdir -p frontend/src
mkdir -p training/{scripts,configs}
mkdir -p models/{checkpoints,onnx}
mkdir -p k8s
mkdir -p docker
mkdir -p docs
mkdir -p data/{raw,processed}

# Create placeholder files
touch README.md
touch .env.example
touch docker-compose.yml
touch requirements.txt

# Create backend
cat > backend/requirements.txt << 'EOF'
# Core
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.0
python-dotenv==1.0.0

# ML
torch==2.0.1
torchvision==0.15.2
timm==0.9.10
opencv-python==4.8.1.78
numpy==1.24.3
albumentations==1.3.1

# Database
sqlalchemy==2.0
psycopg2-binary==2.9.9
alembic==1.12.0

# Async & Queue
redis==5.0
celery==5.3.4
aiofiles==23.2.1

# Monitoring
prometheus-client==0.18.0
python-json-logger==2.0.7

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.1

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.0
EOF

git add .
git commit -m "feat: initialize project structure"
```

---

## Backend Setup

### Step 1: FastAPI Application Structure

```python
# backend/app/__init__.py
"""Deepfake Recognition Backend API"""
__version__ = "1.0.0"

# backend/app/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API
    API_TITLE: str = "Deepfake Recognition API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/deepfake"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "./uploads"
    
    # Models
    MODEL_DIR: str = "./models"
    DEVICE: str = "cuda"  # or "cpu"
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "https://yourdomain.com"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.api import router
from app.middleware import add_monitoring
import logging

# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Deepfake detection API with ensemble models"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring middleware
app.middleware("http")(add_monitoring)

# Include routers
app.include_router(router.router, prefix=settings.API_PREFIX)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.API_VERSION}

@app.on_event("startup")
async def startup():
    logger.info("Application startup")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 2: API Endpoints

```python
# backend/app/api/router.py
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.api.endpoints import analyze, batch, models, health
from app.config import settings

router = APIRouter()

# Include endpoint modules
router.include_router(analyze.router)
router.include_router(batch.router)
router.include_router(models.router)
router.include_router(health.router)

# backend/app/api/endpoints/analyze.py
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import aiofiles
from app.services import inference_service, task_service
from app.schemas import AnalysisRequest, AnalysisResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])

class AnalysisOptions(BaseModel):
    detailed_analysis: bool = True
    return_heatmap: bool = True
    frame_sampling: int = 10

@router.post("", response_model=dict)
async def submit_analysis(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Submit a file for deepfake analysis.
    Returns a task ID for async processing.
    """
    try:
        # Validate file
        if file.size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(400, "File too large")
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Save file
        file_path = await save_upload_file(file, task_id)
        
        # Queue inference task
        background_tasks.add_task(
            inference_service.process_file,
            task_id,
            file_path,
            file.content_type
        )
        
        logger.info(f"Task {task_id} submitted for {file.filename}")
        
        return {
            "task_id": task_id,
            "status": "processing",
            "filename": file.filename
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, "Analysis failed")

@router.get("/{task_id}")
async def get_analysis_result(task_id: str):
    """Retrieve analysis results by task ID."""
    try:
        result = await task_service.get_task_result(task_id)
        
        if not result:
            raise HTTPException(404, "Task not found")
        
        return result
    
    except Exception as e:
        logger.error(f"Error retrieving result: {e}")
        raise HTTPException(500, "Error retrieving result")

async def save_upload_file(upload_file: UploadFile, task_id: str) -> str:
    """Save uploaded file to disk."""
    import os
    from app.config import settings
    
    file_ext = os.path.splitext(upload_file.filename)[1]
    file_path = os.path.join(settings.UPLOAD_DIR, f"{task_id}{file_ext}")
    
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)
    
    return file_path
```

### Step 3: Database Models

```python
# backend/app/db/models.py
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # 'image' or 'video'
    status = Column(String, default="processing")  # processing, completed, failed
    
    # Results
    verdict = Column(String)  # 'real' or 'fake'
    confidence = Column(Float)
    confidence_real = Column(Float)
    confidence_fake = Column(Float)
    
    # Detailed results
    frame_analysis = Column(JSON)  # For videos
    heatmap_data = Column(Text)  # Base64 encoded
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    processing_time_ms = Column(Integer)
    
    # Model info
    model_versions = Column(JSON)  # List of model versions used
    
    # User/API info
    user_id = Column(String, nullable=True)
    api_key = Column(String, nullable=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)  # 'resnet18', 'efficientnet', etc.
    version = Column(String, nullable=False)  # '1.0.0', etc.
    accuracy = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # File paths
    model_path = Column(String, nullable=False)
    config_path = Column(String)
    
    # Metadata
    framework = Column(String)  # 'pytorch', 'onnx'
    input_size = Column(Integer)
    latency_ms = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # Boolean-like
    tags = Column(JSON)  # ['production', 'ensemble', 'latest']

# backend/app/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## Frontend Setup

### Step 1: React Project with TypeScript

```bash
# Create React app
cd frontend
npm create vite@latest . -- --template react-ts

# Install dependencies
npm install react-router-dom axios zustand tailwindcss
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Step 2: Core Components

```typescript
// frontend/src/types/index.ts
export interface AnalysisResult {
  task_id: string;
  verdict: 'real' | 'fake';
  confidence: number;
  confidence_real: number;
  confidence_fake: number;
  processing_time_ms: number;
  heatmap_data?: string;
  frame_analysis?: FrameAnalysis[];
}

export interface FrameAnalysis {
  frame_num: number;
  confidence: number;
  heatmap?: string;
}

export interface Task {
  task_id: string;
  filename: string;
  status: 'processing' | 'completed' | 'failed';
  result?: AnalysisResult;
  created_at: string;
}

// frontend/src/services/api.ts
import axios from 'axios';
import { Task, AnalysisResult } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Submit file for analysis
  submitAnalysis: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    
    return response.data;
  },
  
  // Get analysis results
  getResult: async (taskId: string): Promise<Task> => {
    const response = await api.get(`/analyze/${taskId}`);
    return response.data;
  },
  
  // Poll for results
  pollResult: async (taskId: string, interval = 2000, timeout = 300000) => {
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const result = await apiService.getResult(taskId);
          
          if (result.status === 'completed') {
            resolve(result);
          } else if (result.status === 'failed') {
            reject(new Error('Analysis failed'));
          } else {
            // Continue polling
            if (Date.now() - startTime > timeout) {
              reject(new Error('Request timeout'));
            } else {
              setTimeout(poll, interval);
            }
          }
        } catch (error) {
          reject(error);
        }
      };
      
      poll();
    });
  },
};

// frontend/src/components/FileUpload.tsx
import React, { useRef, useState } from 'react';
import { apiService } from '../services/api';

export const FileUpload: React.FC<{ onSubmit: (taskId: string) => void }> = ({ onSubmit }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await handleFile(files[0]);
    }
  };

  const handleFile = async (file: File) => {
    // Validate file
    const validTypes = ['image/png', 'image/jpeg', 'video/mp4', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
      alert('Invalid file type. Please upload PNG, JPG, MP4, or MOV');
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      alert('File too large. Maximum 100MB');
      return;
    }

    try {
      setIsLoading(true);
      const response = await apiService.submitAnalysis(file);
      onSubmit(response.task_id);
    } catch (error) {
      alert('Upload failed: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        hidden
        onChange={(e) => e.target.files && handleFile(e.target.files[0])}
        accept="image/*,video/*"
      />
      
      {isLoading ? (
        <div className="animate-spin text-blue-500 text-2xl">⟳</div>
      ) : (
        <>
          <div className="text-4xl mb-2">📁</div>
          <p className="text-gray-600 mb-2">Drag file here or click to select</p>
          <p className="text-sm text-gray-400">PNG, JPG, MP4, MOV • Max 100MB</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Select File
          </button>
        </>
      )}
    </div>
  );
};

// frontend/src/components/ResultsDisplay.tsx
import React from 'react';
import { Task } from '../types';

export const ResultsDisplay: React.FC<{ task: Task }> = ({ task }) => {
  if (!task.result) return null;

  const { verdict, confidence, confidence_real, confidence_fake } = task.result;
  const isReal = verdict === 'real';

  return (
    <div className="bg-white rounded-lg shadow-lg p-8">
      <h2 className="text-2xl font-bold mb-6">Analysis Results</h2>

      {/* Verdict */}
      <div className="mb-8 text-center">
        <div className={`text-6xl font-bold mb-2 ${isReal ? 'text-green-600' : 'text-red-600'}`}>
          {isReal ? '✓ REAL' : '✗ FAKE'}
        </div>
        <div className="text-4xl font-bold text-gray-800">
          {(confidence * 100).toFixed(1)}% Confidence
        </div>
      </div>

      {/* Confidence Breakdown */}
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          <span>Real Probability</span>
          <span className="font-bold">{(confidence_real * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-green-600 h-3 rounded-full"
            style={{ width: `${confidence_real * 100}%` }}
          />
        </div>
      </div>

      <div className="mb-8">
        <div className="flex justify-between mb-2">
          <span>Fake Probability</span>
          <span className="font-bold">{(confidence_fake * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-red-600 h-3 rounded-full"
            style={{ width: `${confidence_fake * 100}%` }}
          />
        </div>
      </div>

      {/* Heatmap */}
      {task.result.heatmap_data && (
        <div className="mb-8">
          <h3 className="text-lg font-bold mb-4">Attention Heatmap</h3>
          <img
            src={`data:image/png;base64,${task.result.heatmap_data}`}
            alt="Attention heatmap"
            className="w-full rounded-lg"
          />
        </div>
      )}

      {/* Metadata */}
      <div className="text-sm text-gray-600 border-t pt-4">
        <p>Processing time: {task.result.processing_time_ms}ms</p>
        <p>Filename: {task.filename}</p>
        <p>Analysis completed: {new Date(task.created_at).toLocaleString()}</p>
      </div>
    </div>
  );
};
```

---

## Model Training Setup

### Step 1: Training Script Structure

```python
# training/scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import json

from training.data_loader import DeepfakeDataset
from training.models import build_ensemble
from training.utils import EarlyStopping, setup_logging

logger = logging.getLogger(__name__)

def main(args):
    setup_logging(args.log_level)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Data loaders
    train_dataset = DeepfakeDataset(
        root_dir=args.data_dir,
        split='train',
        transform=transform,
        augment=True
    )
    val_dataset = DeepfakeDataset(
        root_dir=args.data_dir,
        split='val',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    models = build_ensemble(device)
    
    # Optimizer & scheduler
    params = []
    for model in models.values():
        params.extend(model.parameters())
    
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t_0,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True
    )
    
    # Training loop
    best_metrics = {}
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            models,
            train_loader,
            criterion,
            optimizer,
            device,
            args
        )
        
        # Validate
        val_metrics = validate(
            models,
            val_loader,
            criterion,
            device
        )
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Loss: {train_loss:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )
        
        # Scheduler
        scheduler.step()
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
        
        # Save best
        if not best_metrics or val_metrics['auc'] > best_metrics.get('auc', 0):
            best_metrics = val_metrics
            save_checkpoint(models, epoch, val_metrics, args.ckpt_dir)
    
    logger.info(f"Best metrics: {best_metrics}")

def train_epoch(models, dataloader, criterion, optimizer, device, args):
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through ensemble
        outputs = []
        for model in models.values():
            out = model(images)
            outputs.append(out)
        
        # Average predictions
        avg_output = torch.stack(outputs).mean(dim=0)
        
        loss = criterion(avg_output, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in models.values() for p in m.parameters()],
            args.grad_clip
        )
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(models, dataloader, criterion, device):
    with torch.no_grad():
        all_preds = []
        all_labels = []
        total_loss = 0
        
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = []
            for model in models.values():
                out = model(images)
                outputs.append(out)
            
            avg_output = torch.stack(outputs).mean(dim=0)
            loss = criterion(avg_output, labels)
            
            all_preds.extend(avg_output.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

def save_checkpoint(models, epoch, metrics, ckpt_dir):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        path = Path(ckpt_dir) / f"{name}_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'metrics': metrics
        }, path)
    
    logger.info(f"Checkpoint saved to {ckpt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--t-0", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)
```

### Step 2: Run Training

```bash
# Prepare data
python training/scripts/download_datasets.py \
  --datasets faceforensics dfdc \
  --output-dir data/raw

python training/scripts/preprocess_data.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --extract-faces

# Train models
python training/scripts/train.py \
  --data-dir data/processed \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --num-workers 8 \
  --ckpt-dir models/checkpoints
```

---

## Quick Code Examples

### Using Pre-trained Models

```python
# training/models.py
import torch
import torch.nn as nn
import torchvision.models as models
import timm

def build_ensemble(device):
    """Build ensemble of pre-trained models"""
    
    models_dict = {}
    
    # ResNet-18
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # Binary classification
    models_dict['resnet18'] = resnet.to(device)
    
    # EfficientNet-B3
    efficientnet = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2)
    models_dict['efficientnet_b3'] = efficientnet.to(device)
    
    # Vision Transformer
    vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    models_dict['vit_base'] = vit.to(device)
    
    return models_dict

# Inference
def predict(image_path, models, device, transform):
    """Ensemble prediction"""
    import cv2
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    predictions = []
    with torch.no_grad():
        for model in models.values():
            model.eval()
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)
            predictions.append(probs.cpu().numpy())
    
    # Average ensemble
    avg_probs = np.mean(predictions, axis=0)[0]
    
    return {
        'verdict': 'fake' if avg_probs[1] > 0.5 else 'real',
        'confidence': float(max(avg_probs)),
        'confidence_real': float(avg_probs[0]),
        'confidence_fake': float(avg_probs[1])
    }
```

---

## Common Commands

### Backend Development

```bash
# Setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with environment file
export $(cat .env | xargs)
uvicorn app.main:app --reload

# Database migrations
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

# Tests
pytest tests/ -v --cov=app --cov-report=html

# Code quality
black backend/
flake8 backend/
mypy backend/
```

### Frontend Development

```bash
# Setup
cd frontend
npm install

# Development server
npm run dev

# Build
npm run build

# Preview production build
npm run preview

# Lint
npm run lint

# Type check
npm run type-check
```

### Docker

```bash
# Build images
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Clean everything
docker-compose down -v
```

### Git Workflow

```bash
# Feature branch
git checkout -b feature/add-websocket-updates
git add .
git commit -m "feat: add websocket for real-time updates"
git push origin feature/add-websocket-updates
# Create PR on GitHub

# Main deployment
git checkout main
git pull
git merge feature/add-websocket-updates
git push origin main
# GitHub Actions runs tests & deploys
```

---

## Resource Links

- **PyTorch**: https://pytorch.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev/
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **timm (Pytorch Image Models)**: https://github.com/huggingface/pytorch-image-models
- **MLflow**: https://mlflow.org/
- **Docker**: https://docs.docker.com/

---

**Next Steps:**
1. Start with Phase 1: Repository restructuring
2. Run setup commands for backend and frontend
3. Implement core training pipeline
4. Deploy locally with Docker Compose
5. Begin integration testing

Good luck! 🚀
