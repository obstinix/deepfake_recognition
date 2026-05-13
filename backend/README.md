# Deepfake Recognition Backend

This is the FastAPI backend for the deepfake recognition system.

## Stack
- **FastAPI**: High-performance async web framework
- **PyTorch & timm**: ML inference engine for ResNet18, EfficientNet-B3, ViT
- **PostgreSQL**: Relational database for storing tasks and model metadata
- **SQLAlchemy**: ORM for PostgreSQL interactions
- **Redis**: In-memory data store for queuing and caching task states
- **Prometheus**: Application metrics monitoring

## Architecture

1. **Endpoints**: Exposes REST APIs for file analysis (`/api/v1/analyze`), polling (`/api/v1/analyze/{task_id}`), and model health (`/api/v1/models`).
2. **Services**:
   - `InferenceService`: Handles pre-processing, ensemble prediction (softmax averaging), and Grad-CAM heatmap generation.
   - `TaskService`: Manages task state in Redis and persists results to PostgreSQL.
   - `FileHandler`: Securely handles asynchronous file uploads and validations.
3. **Database**: Tracks history of all deepfake analyses and currently active model versions.

## Setup & Running

It is highly recommended to run the backend via Docker Compose from the root directory:

```bash
cd ..
docker-compose up --build backend worker db redis
```

### Running Locally (Without Docker)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables (see `.env.example`).
4. Run the API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Documentation

Once running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
