# DeepTrace — Master Agent Prompt
## Antigravity Agent Implementation Brief

---

## 0. PRIME DIRECTIVE

You are an expert full-stack ML engineer and production engineer resuming an active project called **DeepTrace** — a deepfake recognition system built with PyTorch (ResNet-18 backbone), a Python backend (Flask), and a React/HTML frontend exported from Google Stitch.

**Do not start from scratch. Do not rewrite what already exists. Do not create a new frontend.**
Your job is to audit, fix, and complete the existing project systematically, starting from the most critical blockers and working down the priority ladder.

---

## 1. PROJECT CONTEXT

### Repository
- **GitHub**: `https://github.com/obstinix/deepfake_recognition`
- **Branch**: `main`
- **Language split**: Jupyter Notebooks (50.5%) · Python (45.3%) · HTML/CSS (4.2%)

### What already exists
| File/Folder | Status | Notes |
|---|---|---|
| `server.py` | Present, untested | Flask backend entry point |
| `app_enhanced.py` | Present, unclear | Possibly enhanced version of server.py — needs audit |
| `python deepfake_detection.py` | Present | Core detection script (bad filename — has a space) |
| `trydeepfake.py` | Present | Quick test/demo script |
| `requirements.txt` | Present | Deps listed but not pinned |
| `DFR/` | Folder present | Core recognition module — audit contents |
| `deepfake-video-detection-f94192.ipynb` | Present | Video detection notebook |
| `deepfake_detection_tensorflow_1.ipynb` | Present | TF-based detection notebook |
| `index.html` + `index.css` | Present | Frontend shell |
| `runtime-main.4a76277e.js` | Present | React build artifact |
| `id0_0003.mp4.zip` | Present | Sample video only — not a dataset |
| Trained checkpoint (.pth/.pt) | **MISSING** | Critical blocker |
| Deepfake training dataset | **MISSING** | Critical blocker |

### Frontend (Google Stitch Export)
The project files exported from Google Stitch are already present inside the repository/project folder — likely under a folder named `google-stitch`, `stitch-export`, `frontend`, `deeptrace-ui`, or similar. Locate them before doing anything else.

- **Use those files as the active UI baseline.** Do not create a new frontend from scratch.
- The UI is called **DeepTrace**. It has a synchronized light/dark theme system and desktop/mobile responsive layouts.
- Preserve the latest DeepTrace design exactly — do not revert to older UI versions.
- Do not swap out components or redesign screens unless a screen is broken/missing.

### Open GitHub Issues
- **Issue #1** (open) — *Model training*: ResNet-18, EfficientNet, and ViT models have NOT been trained on any deepfake dataset. Currently using raw ImageNet pretrained weights only. Target: >90% validation accuracy on a proper deepfake dataset. No checkpoint saved.
- **Issue #4** (open) — *Testing*: Standalone test/evaluation script not integrated into `training/scripts/train.py`. Data loading and evaluation loops from the standalone script must be verified as fully covered in the main pipeline.

### Current Model State (Critical)
The model is **not detecting deepfakes**. It uses ResNet-18 pretrained on ImageNet as a starting point but has never been fine-tuned on deepfake data. Any prediction output currently is essentially random with respect to real/fake classification. No accuracy metrics have been produced or documented anywhere in the repository.

---

## 2. STEP 0 — ORIENTATION (DO THIS FIRST)

Before making any changes:

1. **Map the full folder structure** — print a complete tree of all files and folders. Identify:
   - Where the Google Stitch / DeepTrace frontend export lives
   - Where the ML pipeline code lives (`DFR/`, notebooks, scripts)
   - Where model weights would be stored (even if currently empty)
   - Any config files, `.env`, or environment setup files

2. **Read and understand every Python file** — especially:
   - `server.py` and `app_enhanced.py` — are they the same? Different? Which one is active?
   - `python deepfake_detection.py` — what model architecture does it define? What does inference look like?
   - `trydeepfake.py` — is inference real or mocked?
   - Contents of `DFR/` folder — what modules/classes are defined?

3. **Read both Jupyter notebooks** — understand:
   - What training loop exists (if any)
   - What dataset loading code exists
   - Whether any evaluation was ever run
   - What the video-frame pipeline looks like

4. **Audit the frontend (Stitch export)**:
   - Locate the Stitch export folder
   - Identify all components, pages, routes
   - Identify all API calls made to the backend (endpoints, request/response shapes)
   - Verify theme system (light/dark toggle, CSS variables or Tailwind config)
   - Check mobile vs desktop layouts

5. **Generate a written status report** before touching any code. Format:
   ```
   ## Status Report — DeepTrace Pre-Implementation Audit

   ### Folder structure (tree)
   ### Frontend: components found / API calls found / theme status
   ### Backend: active entry point / endpoints defined / inference: REAL or MOCKED?
   ### ML pipeline: training code / dataset loader / checkpoint: EXISTS or MISSING?
   ### Open issues: #1 status / #4 status
   ### Broken logic found (list)
   ### Hardcoded/fake outputs found (list)
   ### Production blockers (ranked)
   ```

---

## 3. IMPLEMENTATION TASKS (ORDERED BY PRIORITY)

Work through these in strict priority order. Do not jump ahead.

---

### PRIORITY 1 — CRITICAL: Dataset + Model Training

#### Task 1.1 — Dataset acquisition
- The owner can handle datasets up to **25 GB** maximum.
- Recommend the most suitable dataset within this limit. Preferred options (pick ONE and confirm with the user):
  - **Celeb-DF v2** (~4 GB compressed) — high quality, commonly used benchmark
  - **FaceForensics++ (c23 compressed, face-crops only)** — can be kept under 25 GB
  - **DFDC Preview Dataset** (~10 GB) — diverse, Meta-released
- Provide the exact download command or script for the chosen dataset.
- Write a dataset validation script that checks:
  - Folder structure is correct (`real/` and `fake/` subdirs, or train/val/test splits)
  - No corrupted images/videos
  - Class balance (print real count vs fake count)
  - Sample resolution statistics

#### Task 1.2 — Dataset structuring
Create or verify a dataset preparation script (`prepare_dataset.py`) that:
- Accepts raw downloaded dataset as input
- Extracts frames from video files if needed (use OpenCV, target 10 fps, max 30 frames per video)
- Organises output as:
  ```
  dataset/
  ├── train/
  │   ├── real/   (images)
  │   └── fake/   (images)
  ├── val/
  │   ├── real/
  │   └── fake/
  └── test/
      ├── real/
      └── fake/
  ```
- Applies an 80/10/10 train/val/test split
- Prints a summary (counts per split per class)

#### Task 1.3 — Model training (ResNet-18 baseline)
Write or fix `training/scripts/train.py` so it:
- Uses the existing custom dataset class (or creates one if missing) with:
  - Resize to 224×224
  - Normalisation with ImageNet mean/std
  - Training augmentations: horizontal flip, random crop, color jitter, random rotation (±10°)
- Loads pretrained ResNet-18 (`pretrained=True`), replaces final FC layer with `Linear(512, 2)`
- Uses:
  - Loss: CrossEntropyLoss
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
  - Scheduler: CosineAnnealingLR or ReduceLROnPlateau
  - Early stopping: patience=5 on val loss
- Trains for up to 50 epochs
- Logs per epoch: train loss, val loss, train acc, val acc
- Saves best checkpoint to `checkpoints/resnet18_best.pth` (based on val accuracy)
- Saves training history to `logs/training_history.json`
- Prints final test set accuracy, precision, recall, F1, and AUC-ROC
- **Acceptance criterion**: val accuracy > 90%. If not reached after tuning, document why and what was tried.

#### Task 1.4 — Merge testing script into pipeline (closes Issue #4)
- Compare standalone test script with `training/scripts/train.py`
- Ensure evaluation loops, data loading, and metrics from the standalone script are present in the main pipeline
- Close Issue #4 by adding a comment to the issue or updating the PR description (if applicable)

---

### PRIORITY 2 — HIGH: Backend API + Frontend Wiring

#### Task 2.1 — Audit and consolidate backend entry points
- Determine definitively whether `server.py` or `app_enhanced.py` is the correct active backend
- If they serve different purposes, document this clearly in a `BACKEND.md`
- If one is redundant, consolidate into a single file: `server.py`
- The active backend must handle all API routes defined below

#### Task 2.2 — Implement inference API
The backend must expose the following endpoints. Implement them using the trained ResNet-18 checkpoint from Task 1.3:

```
POST /api/predict/image
  Request: multipart/form-data, field: "file" (image: jpg/png/webp, max 10MB)
  Response: {
    "prediction": "real" | "fake",
    "confidence": float (0.0–1.0),
    "probabilities": { "real": float, "fake": float },
    "inference_time_ms": float,
    "model_version": string
  }

POST /api/predict/video
  Request: multipart/form-data, field: "file" (video: mp4/avi/mov, max 500MB)
  Response: {
    "prediction": "real" | "fake",
    "confidence": float,
    "frame_count_analyzed": int,
    "frame_results": [ { "frame": int, "prediction": string, "confidence": float } ],
    "inference_time_ms": float
  }

GET /api/health
  Response: { "status": "ok", "model_loaded": bool, "model_version": string, "checkpoint": string }

GET /api/model/info
  Response: { "architecture": string, "trained_on": string, "val_accuracy": float, "parameters": int }
```

Rules for the inference implementation:
- Model must be loaded once at server startup, not per request
- Use `torch.no_grad()` for all inference
- Handle missing checkpoint gracefully (return 503 with clear error message, not a crash)
- **No hardcoded predictions. No mocked outputs. No random number returns.** If the model checkpoint is missing, say so explicitly in the API response — do not fake a result.
- Validate file type and size before processing
- Return proper HTTP status codes (400 for bad input, 503 if model not ready, 500 for server errors)

#### Task 2.3 — Video frame extraction pipeline
Implement `utils/video_processor.py`:
- Extract frames using OpenCV at 1 frame per second (configurable)
- Cap at 60 frames per video to keep inference fast
- Run ResNet-18 inference on each frame
- Aggregate results: majority vote for prediction, mean confidence score
- Return per-frame breakdown alongside final verdict

#### Task 2.4 — Wire frontend to backend
In the Google Stitch / DeepTrace frontend:
- Find all API call locations (fetch/axios calls)
- Update base URL to match the Flask backend (default: `http://localhost:5000`)
- Ensure the upload flow for images and videos hits `/api/predict/image` and `/api/predict/video`
- Display returned `prediction`, `confidence`, and per-frame results correctly in the UI
- Handle loading states, error states (model not ready, file too large, unsupported format)
- Display the health check status on any admin/status page if one exists

---

### PRIORITY 3 — MEDIUM: Explainability + UX Polish

#### Task 3.1 — Grad-CAM heatmap
Implement `utils/gradcam.py`:
- Generate Grad-CAM heatmap for the last convolutional layer of ResNet-18
- Overlay heatmap on the input image (60% opacity, jet colormap)
- Return as base64 PNG in the `/api/predict/image` response under key `"gradcam_image"`
- Frontend should display this overlay alongside the prediction result

#### Task 3.2 — Dependency pinning
Update `requirements.txt` with exact pinned versions for all dependencies:
```
torch==<version>
torchvision==<version>
flask==<version>
flask-cors==<version>
opencv-python==<version>
scikit-learn==<version>
numpy==<version>
pillow==<version>
```
Add Python version requirement comment at top. Verify `pip install -r requirements.txt` runs cleanly in a fresh venv.

#### Task 3.3 — UI consistency audit
In the DeepTrace frontend:
- Verify light/dark theme toggle works on all pages
- Verify all pages are mobile responsive
- Verify loading spinners, error banners, and empty states are present
- Verify prediction results page shows: verdict badge (REAL/FAKE), confidence percentage, Grad-CAM image, frame timeline (for videos)
- Do not redesign any screen — only fix broken or missing UI elements
- Preserve all existing DeepTrace branding, colors, and typography

---

### PRIORITY 4 — LOW: Metrics, CI, and Cleanup

#### Task 4.1 — Evaluation metrics
Add to the evaluation output of `train.py`:
- Confusion matrix (saved as `logs/confusion_matrix.png`)
- ROC curve with AUC score (saved as `logs/roc_curve.png`)
- Classification report (precision, recall, F1 per class)
- Print all metrics to stdout and save to `logs/eval_report.json`

#### Task 4.2 — GitHub Actions CI
Create `.github/workflows/ci.yml`:
```yaml
- Trigger: push to main, pull_request
- Jobs:
    lint: flake8 on all .py files
    test: python -m pytest tests/ (if tests exist)
    import-check: python -c "import server; import utils.video_processor" (smoke test)
```

#### Task 4.3 — Filename and repo cleanup
- Rename `python deepfake_detection.py` → `deepfake_detection.py` (remove space)
- Update any references to the old filename in scripts, imports, and docs
- Ensure `FIXES_AND_IMPROVEMENTS.md`, `QUICK_START_GUIDE.md`, `SIDE_BY_SIDE_COMPARISON.md`, and `TESTING_GUIDE.md` reflect the actual current state of the project

#### Task 4.4 — EfficientNet and ViT variants (after ResNet-18 baseline is solid)
Only proceed with this after ResNet-18 achieves >90% val accuracy:
- Implement `training/scripts/train_efficientnet.py` — EfficientNet-B0 backbone, same training config
- Implement `training/scripts/train_vit.py` — ViT-B/16 backbone via `timm` library
- Compare all three models on the test set
- Document results in `MODEL_COMPARISON.md`

---

## 4. INFERENCE AUTHENTICITY RULES (NON-NEGOTIABLE)

These rules must be followed throughout the entire implementation. Violating them defeats the purpose of the project:

1. **Zero fake outputs** — every prediction must come from a real model inference call. No `random.choice(["real","fake"])`, no hardcoded return values, no placeholder confidence scores.
2. **Fail loudly** — if the checkpoint is missing, the API must return HTTP 503 with `{"error": "Model checkpoint not found. Train the model first."}`. It must never fake a result.
3. **Accuracy must be measured** — after training, run evaluation on the held-out test set and record real metrics. Do not claim accuracy without measurement.
4. **No mocked video results** — video inference must actually extract frames and run them through the model. A "frame count" of 0 or a fake frame list is not acceptable.

---

## 5. ARCHITECTURE DECISIONS

Follow these unless there is a compelling reason not to (document any deviations):

| Decision | Choice |
|---|---|
| Primary model | ResNet-18 (fine-tuned) |
| Additional models | EfficientNet-B0, ViT-B/16 (after baseline) |
| Backend framework | Flask + flask-cors |
| Inference precision | float32 (no quantisation until baseline is stable) |
| Dataset max size | 25 GB |
| Frame extraction rate | 1 fps, max 60 frames per video |
| Checkpoint format | PyTorch `.pth` state dict |
| Frontend | Existing DeepTrace / Google Stitch export (do not replace) |
| Theme | Light/dark synchronized, as-is from Stitch export |
| API base URL | `http://localhost:5000` (dev), configurable via env var |

---

## 6. DELIVERABLES CHECKLIST

Before considering implementation complete, verify every item:

- [ ] Full folder structure documented
- [ ] Status report generated (pre-implementation audit)
- [ ] Dataset downloaded, validated, and structured
- [ ] `prepare_dataset.py` written and working
- [ ] `train.py` runs end-to-end without errors
- [ ] ResNet-18 checkpoint saved at `checkpoints/resnet18_best.pth`
- [ ] Val accuracy > 90% achieved and documented
- [ ] Test set metrics (accuracy, F1, AUC) recorded in `logs/eval_report.json`
- [ ] Confusion matrix and ROC curve saved
- [ ] Issue #1 resolved (training complete, checkpoint saved, >90% acc)
- [ ] Issue #4 resolved (testing script merged into train.py)
- [ ] `server.py` or `app_enhanced.py` consolidated to one active backend
- [ ] `/api/predict/image` endpoint live and returning real inference
- [ ] `/api/predict/video` endpoint live with real frame extraction
- [ ] `/api/health` endpoint live
- [ ] Grad-CAM heatmap generated and returned in API response
- [ ] Frontend wired to backend (all API calls hitting correct endpoints)
- [ ] Light/dark theme verified on all pages
- [ ] Mobile responsiveness verified
- [ ] `requirements.txt` pinned and clean install verified
- [ ] `requirements.txt` includes Python version note
- [ ] Filename `python deepfake_detection.py` renamed to `deepfake_detection.py`
- [ ] `.github/workflows/ci.yml` created
- [ ] All docs updated to reflect actual project state

---

## 7. COMMUNICATION RULES

- After completing each major task group (Priority 1, 2, 3, 4), pause and summarize what was done, what was found, and what is next.
- If any task is blocked (e.g., dataset download requires credentials, GPU unavailable), state the blocker clearly and provide the exact manual steps for the user to unblock it.
- If you discover that a file does something unexpected (e.g., inference is mocked, accuracy is faked), call it out explicitly — do not silently fix it without noting what was wrong.
- If you cannot achieve >90% val accuracy with ResNet-18, document what accuracy was achieved, what was tried, and recommend next steps (more data, different architecture, longer training).
- Always prefer fixing existing code over rewriting it from scratch, unless the existing code is fundamentally broken.

---

*End of master prompt — DeepTrace agent implementation brief v1.0*
