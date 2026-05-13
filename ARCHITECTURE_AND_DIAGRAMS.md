# Deepfake Recognition - System Architecture & Diagrams

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Web Browser    │  │  Mobile App      │  │  API Clients     │  │
│  │   (React SPA)    │  │  (React Native)  │  │  (CLI/Script)    │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                     │             │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            │         HTTPS/WebSocket                   │
            └─────────────────────┼─────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────┐
│                      API GATEWAY LAYER                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  FastAPI + Uvicorn                                               │
│  - CORS Middleware                                               │
│  - Rate Limiting (100 req/hour per IP)                          │
│  - Authentication/Authorization                                 │
│  - Request Validation                                            │
│  - Error Handling                                                │
│  - Monitoring/Metrics                                            │
│                                                                  │
│  Endpoints:                                                      │
│  POST   /api/v1/analyze          (single file analysis)         │
│  GET    /api/v1/analyze/{task_id} (get results)                 │
│  POST   /api/v1/batch             (batch processing)            │
│  GET    /api/v1/models            (list models)                 │
│  GET    /health                   (health check)                │
│  GET    /metrics                  (prometheus metrics)          │
│                                                                  │
└──────────┬──────────────────────┬──────────────────────┬────────┘
           │                      │                      │
     ┌─────▼────┐         ┌──────▼────┐         ┌──────▼──────┐
     │  FILES   │         │  QUEUE    │         │  DATABASE  │
     │ STORAGE  │         │ (Celery)  │         │(PostgreSQL)│
     └──────────┘         └───┬──────┘         └────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                     PROCESSING LAYER                           │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Worker Processes (Celery)                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  1. Extract frames (videos)                             │ │
│  │  2. Face detection & alignment (MTCNN)                  │ │
│  │  3. Preprocess images (resize, normalize)               │ │
│  │  4. Run inference on ensemble models                    │ │
│  │  5. Generate heatmaps/visualizations                    │ │
│  │  6. Format and store results                            │ │
│  │                                                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML MODEL LAYER                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Ensemble of Fine-tuned Models                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  ResNet-18       │  │ EfficientNet-B3  │  │  ViT-Base  │ │
│  │  Acc: 92%        │  │ Acc: 94%         │  │ Acc: 91%   │ │
│  │  Latency: 50ms   │  │ Latency: 100ms   │  │ Latency:150ms
│  └──────────────────┘  └──────────────────┘  └────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Ensemble Voting (Majority + Confidence)       │   │
│  │  Final Output: Verdict + Confidence Score            │   │
│  │  Accuracy: 95%+ | Latency: 120ms                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                  STORAGE & CACHE LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐│
│  │   Redis      │  │ S3/Cloud     │  │  PostgreSQL        ││
│  │   (Cache)    │  │  Storage     │  │  (Results DB)      ││
│  │  - Results   │  │  - Models    │  │  - Tasks           ││
│  │  - Models    │  │  - Uploads   │  │  - Model metrics   ││
│  │  - Sessions  │  │  - Heatmaps  │  │  - User data       ││
│  └──────────────┘  └──────────────┘  └────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### User Upload to Result Flow

```
USER UPLOADS FILE
        │
        ▼
┌──────────────────────────────────┐
│  File Validation                 │
│  - Check file type               │
│  - Check file size (<100MB)      │
│  - Scan for malware              │
└─────┬────────────────────────────┘
      │
      ├─ Invalid → Return 400 Error
      │
      └─ Valid
         │
         ▼
┌──────────────────────────────────┐
│  Save Uploaded File              │
│  - Store in ./uploads/{task_id}  │
│  - Generate unique task_id       │
│  - Log upload metadata           │
└─────┬────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  Queue Task                      │
│  - Add to Celery queue           │
│  - Return task_id to client      │
│  - Start async processing        │
└─────┬────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────┐
│  ASYNC PROCESSING (Worker)                   │
├──────────────────────────────────────────────┤
│                                              │
│  Step 1: Pre-processing                      │
│  ├─ If video: Extract frames every 10 frames│
│  ├─ Detect faces using MTCNN                │
│  └─ Crop and align faces                    │
│                                              │
│  Step 2: Normalization                       │
│  ├─ Resize to 224x224                       │
│  ├─ Normalize (ImageNet stats)              │
│  └─ Convert to tensors                      │
│                                              │
│  Step 3: Run Ensemble Inference              │
│  ├─ ResNet-18 prediction                    │
│  ├─ EfficientNet-B3 prediction              │
│  ├─ ViT-Base prediction                     │
│  └─ Ensemble voting (average confidence)    │
│                                              │
│  Step 4: Generate Visualizations             │
│  ├─ Compute attention heatmap (Grad-CAM)    │
│  ├─ Create confidence gauge                 │
│  └─ Frame-by-frame breakdown (videos)       │
│                                              │
│  Step 5: Store Results                       │
│  ├─ Save to database                        │
│  ├─ Cache in Redis                          │
│  └─ Upload heatmaps to S3                   │
│                                              │
└──────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│  Task Completed                  │
│  - Status: 'completed'           │
│  - Results ready for retrieval   │
│  - Client receives via polling   │
└──────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  CLIENT RETRIEVES RESULTS                            │
│  GET /api/v1/analyze/{task_id}                       │
│                                                      │
│  Response:                                           │
│  {                                                  │
│    "task_id": "uuid-xxx",                          │
│    "status": "completed",                          │
│    "verdict": "fake",                              │
│    "confidence": 0.95,                             │
│    "confidence_real": 0.05,                        │
│    "confidence_fake": 0.95,                        │
│    "heatmap": "base64-image-data",                 │
│    "frame_analysis": [...],                        │
│    "processing_time_ms": 3420                      │
│  }                                                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Database Schema

### Core Tables

```sql
-- Tasks (submitted analysis jobs)
CREATE TABLE tasks (
    id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    file_type VARCHAR(20),  -- 'image' or 'video'
    status VARCHAR(20),     -- 'processing', 'completed', 'failed'
    
    -- Results
    verdict VARCHAR(10),    -- 'real' or 'fake'
    confidence FLOAT,
    confidence_real FLOAT,
    confidence_fake FLOAT,
    
    -- Detailed results
    frame_analysis JSON,
    heatmap_data LONGTEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time_ms INT,
    
    -- Model info
    model_versions JSON,
    
    -- User info
    user_id VARCHAR(36),
    api_key VARCHAR(255),
    
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Model versions (track all models)
CREATE TABLE model_versions (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,         -- 'resnet18'
    version VARCHAR(20) NOT NULL,       -- '1.0.0'
    accuracy FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    
    model_path VARCHAR(512),
    config_path VARCHAR(512),
    
    framework VARCHAR(50),              -- 'pytorch', 'onnx'
    input_size INT,
    latency_ms FLOAT,
    
    created_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    tags JSON,
    
    UNIQUE KEY unique_model_version (name, version)
);

-- Model metrics (track performance over time)
CREATE TABLE model_metrics (
    id VARCHAR(36) PRIMARY KEY,
    model_version_id VARCHAR(36),
    metric_date DATE,
    
    -- Performance metrics
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    
    -- Detailed breakdown
    true_positives INT,
    true_negatives INT,
    false_positives INT,
    false_negatives INT,
    
    -- Inference metrics
    avg_latency_ms FLOAT,
    throughput_requests_per_hour INT,
    
    -- Errors
    error_count INT,
    
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id),
    INDEX idx_metric_date (metric_date)
);

-- Audit logs
CREATE TABLE audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    action VARCHAR(100),               -- 'upload', 'analyze', 'download'
    resource_type VARCHAR(50),
    resource_id VARCHAR(36),
    user_id VARCHAR(36),
    ip_address VARCHAR(45),
    status VARCHAR(20),
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);
```

### ER Diagram

```
┌─────────────────────┐         ┌────────────────────┐
│     tasks           │         │  model_versions    │
├─────────────────────┤         ├────────────────────┤
│ PK id               │         │ PK id              │
│    filename         │         │    name            │
│    file_path        │         │    version         │
│    file_type        │    ┌────│    accuracy        │
│    status           │    │    │    model_path      │
│    verdict          │    │    │    is_active       │
│    confidence       │    │    └────────────────────┘
│    frame_analysis   │    │
│    heatmap_data     │    │ 1:N
│ FK model_versions   ├────┘
│    created_at       │
│    processing_time  │
└─────────────────────┘
         │
         │ 1:N
         │
┌────────▼──────────────┐
│   model_metrics       │
├───────────────────────┤
│ PK id                 │
│ FK model_version_id   │
│    metric_date        │
│    accuracy           │
│    precision          │
│    recall             │
│    avg_latency_ms     │
│    throughput         │
└───────────────────────┘


┌─────────────────────┐
│   audit_logs        │
├─────────────────────┤
│ PK id               │
│    action           │
│    resource_type    │
│    resource_id      │
│    user_id          │
│    ip_address       │
│    status           │
│    details          │
│    created_at       │
└─────────────────────┘
```

---

## API Request/Response Flow

### Example: Image Analysis

```
REQUEST:
POST /api/v1/analyze
Content-Type: multipart/form-data

file: <binary-image-data>

─────────────────────────────────────────────────────

RESPONSE (202 Accepted):
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "filename": "sample_image.jpg"
}

─────────────────────────────────────────────────────

POLLING REQUEST (Client polls every 2-5 seconds):
GET /api/v1/analyze/550e8400-e29b-41d4-a716-446655440000

─────────────────────────────────────────────────────

RESPONSE (While processing):
HTTP 200 OK
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45
}

─────────────────────────────────────────────────────

RESPONSE (When completed):
HTTP 200 OK
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "sample_image.jpg",
  
  "result": {
    "verdict": "fake",
    "confidence": 0.95,
    "confidence_real": 0.05,
    "confidence_fake": 0.95,
    "processing_time_ms": 3420,
    "models_used": ["resnet18-v1.0", "efficientnet-v1.0", "vit-v1.0"],
    
    "heatmap_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf...",
    "heatmap_type": "grad_cam",
    
    "frame_analysis": null  -- null for images, populated for videos
  },
  
  "metadata": {
    "created_at": "2026-05-13T10:30:00Z",
    "completed_at": "2026-05-13T10:30:03.42Z",
    "file_size_bytes": 245820,
    "file_type": "image/jpeg"
  }
}
```

---

## Deployment Architecture

### Docker Compose (Local Development)

```
┌─────────────────────────────────────────────────────────┐
│          Docker Compose (docker-compose.yml)            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│ ┌─────────────────┐  ┌──────────────┐  ┌────────────┐   │
│ │  Backend        │  │  Frontend    │  │  Database  │   │
│ │  Service        │  │  Service     │  │  Service   │   │
│ │  Port: 8000     │  │  Port: 3000  │  │  Port:5432 │   │
│ │  PyTorch        │  │  Node.js     │  │  Postgres  │   │
│ │  FastAPI        │  │  React       │  │            │   │
│ └────────┬────────┘  └──────┬───────┘  └────┬───────┘   │
│          │                  │               │            │
│ ┌────────▼──────────────────▼───────────────▼────────┐   │
│ │           Shared Networks & Volumes              │   │
│ │  - Network: deepfake-net                         │   │
│ │  - Volumes: postgres_data, models, uploads       │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ ┌──────────────┐  ┌────────────┐  ┌──────────────┐    │
│ │   Redis      │  │  Celery    │  │  Nginx       │    │
│ │  Port: 6379  │  │  Worker    │  │  Port: 80    │    │
│ │  Cache       │  │  Async Job │  │  Reverse     │    │
│ │              │  │  Queue     │  │  Proxy       │    │
│ └──────────────┘  └────────────┘  └──────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Kubernetes (Production)

```
┌──────────────────────────────────────────────────────────┐
│            Kubernetes Cluster                            │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │            Ingress                               │    │
│  │  (HTTPS, TLS, Rate Limiting)                    │    │
│  └────────────┬─────────────────────────────────────┘    │
│               │                                           │
│  ┌────────────▼────────────────────────────────────┐    │
│  │     API Service (LoadBalancer)                  │    │
│  └────────────┬────────────────────────────────────┘    │
│               │                                           │
│  ┌────────────▼────────────────────────────────────┐    │
│  │    Backend Pods (Replicas: 3+)                 │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐    │    │
│  │  │Backend    │ │Backend    │ │Backend    │    │    │
│  │  │Pod #1     │ │Pod #2     │ │Pod #N     │    │    │
│  │  └───────────┘ └───────────┘ └───────────┘    │    │
│  └────────────┬────────────────────────────────────┘    │
│               │                                           │
│  ┌────────────▼────────────────────────────────────┐    │
│  │  Worker Pods (Replicas: 2+)                    │    │
│  │  ┌───────────┐ ┌───────────┐                   │    │
│  │  │Worker     │ │Worker     │                   │    │
│  │  │Pod #1     │ │Pod #2     │                   │    │
│  │  └───────────┘ └───────────┘                   │    │
│  └────────────┬────────────────────────────────────┘    │
│               │                                           │
│  ┌────────────▼────────────────────────────────────┐    │
│  │   Stateful Services                            │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │    │
│  │  │PostgreSQL│ │  Redis   │ │RabbitMQ      │   │    │
│  │  │StatefulS │ │  Cache   │ │Job Broker    │   │    │
│  │  └──────────┘ └──────────┘ └──────────────┘   │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Storage Services                              │    │
│  │  ┌───────────────┐  ┌──────────────────────┐  │    │
│  │  │  S3 Bucket    │  │  Persistent Volume   │  │    │
│  │  │  (Models,     │  │  (Logs, Backups)     │  │    │
│  │  │   Uploads)    │  │                      │  │    │
│  │  └───────────────┘  └──────────────────────┘  │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Monitoring & Logging                          │    │
│  │  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │ Prometheus   │  │  ELK / Loki Stack    │   │    │
│  │  │ (Metrics)    │  │  (Logs)              │   │    │
│  │  └──────────────┘  └──────────────────────┘   │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Monitoring & Observability Stack

### Metrics Collection (Prometheus)

```
Application Metrics:
├── Request Metrics
│   ├── http_requests_total (counter)
│   ├── http_request_duration_seconds (histogram)
│   └── http_requests_in_progress (gauge)
│
├── Model Metrics
│   ├── inference_duration_seconds (histogram)
│   ├── model_predictions (counter)
│   ├── model_accuracy (gauge)
│   └── model_errors (counter)
│
├── Queue Metrics
│   ├── celery_task_total (counter)
│   ├── celery_task_duration (histogram)
│   ├── celery_queue_length (gauge)
│   └── celery_worker_pool_size (gauge)
│
├── Database Metrics
│   ├── db_connections (gauge)
│   ├── db_query_duration_seconds (histogram)
│   └── db_errors (counter)
│
└── System Metrics
    ├── process_resident_memory_bytes
    ├── process_cpu_seconds_total
    └── process_start_time_seconds
```

### Logging Architecture

```
Application
    │
    ├── Structured JSON Logs
    │   └── Correlation ID
    │
    ├── Multiple Outputs
    │   ├── Console (development)
    │   ├── File (backup)
    │   └── Syslog (production)
    │
    └── Log Aggregation
        └── ELK Stack or Loki
            ├── Elasticsearch (storage)
            ├── Logstash (processing)
            └── Kibana (visualization)
            
            OR
            
            ├── Loki (storage)
            ├── Promtail (collector)
            └── Grafana (visualization)
```

---

## Security Architecture

### Data Flow Security

```
Internet
    │
    ├─ TLS 1.3 (HTTPS only)
    │
    ▼
API Gateway (Nginx/ALB)
    │
    ├─ CORS validation
    ├─ Rate limiting (100 req/hour)
    ├─ Request size limits
    │
    ▼
Application Authentication
    │
    ├─ API Key validation (if needed)
    ├─ Session verification
    ├─ User authorization checks
    │
    ▼
Input Validation
    │
    ├─ File type validation
    ├─ File size validation
    ├─ Malware scanning
    ├─ SQL injection prevention (ORM)
    │
    ▼
Database Encryption
    │
    ├─ Data at rest (encrypted)
    ├─ Data in transit (TLS)
    ├─ Sensitive data masking in logs
    │
    ▼
Secure File Storage
    │
    ├─ S3/Cloud with encryption
    ├─ Access control (IAM)
    ├─ Signed URLs for downloads
    └─ Automatic cleanup of old files
```

---

## Performance Optimization Strategies

### Caching Layers

```
Level 1: CDN Cache (Frontend)
├─ Serve static assets (HTML, CSS, JS)
├─ TTL: 1 hour
└─ Reduces bandwidth

Level 2: Redis Cache (API Results)
├─ Cache analysis results (task_id → result)
├─ TTL: 24 hours
├─ Saves re-computation
└─ Very fast (<1ms lookup)

Level 3: Database Cache (Query Results)
├─ Connection pooling
├─ Query result caching
└─ Reduces database load

Level 4: Model Inference Cache
├─ Cache model outputs for identical inputs
├─ Use file hash as key
├─ TTL: 7 days
└─ Very useful for repeated analyses
```

### Load Balancing

```
Incoming Requests
        │
        ▼
Load Balancer (AWS ALB / GCP LB)
├─ Distributes traffic
├─ Health checks
├─ Auto-scaling trigger
│
┌───────────┬───────────┬───────────┐
│           │           │           │
▼           ▼           ▼           ▼
Backend    Backend    Backend    Backend
Pod #1     Pod #2     Pod #3     Pod #N
(8 workers (8 workers (8 workers (8 workers
per pod)   per pod)   per pod)   per pod)
```

### Horizontal Scaling

```
Traffic Increases
        │
        ▼
Prometheus detects high latency
        │
        ▼
Kubernetes HPA (Horizontal Pod Autoscaler)
├─ CPU > 70%
├─ Memory > 80%
├─ Requests per second > 1000
│
└─ Triggers scale-up
   └─ Spin up new pods
   └─ Add to load balancer
   └─ Route new requests
```

---

## Disaster Recovery Plan

### Backup Strategy

```
Database Backups
├─ Daily full backups
├─ Hourly incremental backups
├─ Cross-region replication
├─ 30-day retention
└─ Automated backup tests

Model Backups
├─ Version control (Git)
├─ S3 with versioning
├─ Automatic daily snapshots
└─ Signed backups with checksums

Application Backups
├─ Docker images in registry
├─ Infrastructure as Code (Terraform/Helm)
├─ Configuration management
└─ Automated deployment scripts
```

### Disaster Recovery (RTO/RPO)

```
Recovery Time Objective (RTO): < 1 hour
Recovery Point Objective (RPO): < 15 minutes

Plan:
1. Detect failure (automated alerting)
   └─ 1-2 minutes

2. Failover to backup
   └─ Database: 5-10 minutes (replication)
   └─ Application: 10-15 minutes (k8s redeploy)

3. Restore from backup
   └─ Full restore: 20-30 minutes
   └─ Partial restore: 5-15 minutes

Total RTO: < 60 minutes
Total RPO: < 15 minutes (replicated data)
```

---

## Scaling Considerations

### Vertical Scaling (Single Server)
```
Max capacity: ~1000 req/hour
Solutions:
├─ Increase CPU cores
├─ Increase RAM
├─ Use GPU for inference
└─ Limited by hardware constraints
```

### Horizontal Scaling (Multiple Servers)
```
Recommended approach:

Load: 1,000-10,000 req/hour
├─ 2-5 backend pods
├─ 2-3 worker pods
├─ 1 database instance
└─ Distributed cache (Redis)

Load: 10,000-100,000 req/hour
├─ 10-20 backend pods
├─ 5-10 worker pods
├─ Replicated database
├─ Distributed cache cluster
└─ CDN for frontend

Load: 100,000+ req/hour
├─ Multi-region deployment
├─ Database sharding
├─ Dedicated GPU cluster
├─ Advanced caching strategies
└─ Requires significant infrastructure
```

---

## Quick Reference: Key Files & Services

```
BACKEND SERVICES:
├─ FastAPI Application (Port 8000)
├─ Celery Workers (Async Processing)
├─ PostgreSQL (Port 5432)
├─ Redis (Port 6379)
└─ RabbitMQ (Port 5672)

FRONTEND SERVICES:
├─ React SPA (Port 3000)
├─ Nginx Reverse Proxy (Port 80/443)
└─ CDN (Optional, production)

MONITORING SERVICES:
├─ Prometheus (Port 9090)
├─ Grafana (Port 3000)
├─ ELK/Loki (Logs)
└─ Jaeger (Tracing, optional)

MODEL STORAGE:
├─ Local: ./models/checkpoints/
├─ Production: S3/GCS/Azure Blob
└─ Registry: MLflow or DVC (optional)
```

---

This comprehensive guide should help you visualize and understand the entire system architecture. Good luck with your implementation! 🚀
