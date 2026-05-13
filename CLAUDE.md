# CLAUDE.md — Deepfake Recognition Project Instructions

## Git Auto-Commit Rules (CRITICAL — NEVER SKIP)

After **every single file creation or modification**, you MUST immediately run:

```bash
git add .
git commit -m "<type>(<scope>): <short description>"
git push origin main
```

**Commit types:** feat | fix | chore | docs | style | refactor | test | ci

**Example commits:**
- `feat(backend): add FastAPI main application`
- `feat(frontend): create FileUpload React component`
- `feat(ml): add ResNet18 training script`
- `chore(repo): initialize project structure`
- `fix(api): correct CORS middleware config`

**NEVER batch multiple files into one commit.**  
**NEVER wait until a phase is complete to commit.**  
**One logical change = one commit = one push.**

---

## GitHub Setup

Remote URL (with PAT already embedded):
```
https://<GITHUB_PAT>@github.com/obstinix/deepfake_recognition.git
```

Run this once at the start of the session:
```bash
git remote set-url origin https://<GITHUB_PAT>@github.com/obstinix/deepfake_recognition.git
git config user.email "obstinix@gmail.com"
git config user.name "obstinix-"
```

---

## Project: Deepfake Recognition System

**Goal:** Transform the current scattered proof-of-concept into a production-grade deepfake detection system.

**Stack:**
- Backend: FastAPI + Python 3.10
- Frontend: React 18 + TypeScript + Tailwind CSS
- ML: PyTorch + timm (ResNet-18, EfficientNet-B3, ViT)
- Database: PostgreSQL + SQLAlchemy
- Queue: Celery + Redis
- Monitoring: Prometheus + Grafana
- Deploy: Docker + Docker Compose
- CI/CD: GitHub Actions

---

## Current State (What's Broken)

The repo `obstinix/deepfake_recognition` has:
- Loose Python files in root (app_enhanced.py, server.py, trydeepfake.py)
- Basic HTML/CSS frontend (index.html, index.css)
- Jupyter notebooks instead of proper training scripts
- No proper API, database, or structure
- Old files to archive: `archive/` subdirectory

---

## Build Order (Execute in This Exact Sequence)

### PHASE 1 — Repo Structure
1. Create all directories
2. Move old files to archive/
3. Commit each directory/file creation

### PHASE 2 — Backend
4. FastAPI app skeleton
5. Config and environment
6. Database models + migrations
7. API endpoints (analyze, batch, models, health)
8. Services (file handler, inference, task queue)
9. Middleware (CORS, rate limit, monitoring)

### PHASE 3 — ML Pipeline
10. Dataset loader with augmentation
11. Model definitions (ResNet18, EfficientNet, ViT ensemble)
12. Training script
13. Inference service
14. Grad-CAM heatmap generation

### PHASE 4 — Frontend
15. Vite + React + TypeScript scaffold
16. Tailwind + design tokens
17. FileUpload component
18. ResultsDisplay component
19. ConfidenceGauge component
20. Pages: Home, Detect, Results, Dashboard
21. API service layer

### PHASE 5 — DevOps
22. Dockerfile (backend)
23. Dockerfile (frontend)
24. docker-compose.yml
25. .env.example
26. GitHub Actions CI/CD workflow
27. requirements.txt (backend)
28. package.json (frontend)

### PHASE 6 — Documentation
29. README.md (root)
30. backend/README.md
31. frontend/README.md
32. training/README.md

---

## File Structure to Create

```
deepfake_recognition/
├── archive/                          ← Move old files here
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py
│   │   │   └── endpoints/
│   │   │       ├── analyze.py
│   │   │       ├── batch.py
│   │   │       ├── models_endpoint.py
│   │   │       └── health.py
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── session.py
│   │   │   └── crud.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── file_handler.py
│   │   │   ├── inference_service.py
│   │   │   ├── task_service.py
│   │   │   └── video_processor.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── ensemble.py
│   │   │   ├── models.py
│   │   │   └── heatmap.py
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── monitoring.py
│   │   │   └── rate_limit.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_analyze.py
│   │   └── test_models.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload/
│   │   │   │   └── index.tsx
│   │   │   ├── ResultsDisplay/
│   │   │   │   └── index.tsx
│   │   │   ├── ConfidenceGauge/
│   │   │   │   └── index.tsx
│   │   │   ├── HeatmapViewer/
│   │   │   │   └── index.tsx
│   │   │   └── Navigation/
│   │   │       └── index.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Detect.tsx
│   │   │   ├── Results.tsx
│   │   │   └── Dashboard.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── hooks/
│   │   │   ├── useFileUpload.ts
│   │   │   └── usePolling.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── Dockerfile
│   └── README.md
├── training/
│   ├── scripts/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── preprocess_data.py
│   ├── configs/
│   │   └── default.yaml
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── augmentation.py
│   │   ├── metrics.py
│   │   └── early_stopping.py
│   ├── requirements.txt
│   └── README.md
├── models/
│   └── .gitkeep
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── docker/
│   └── nginx.conf
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```
