# Quick Start Guide - Deepfake Detection Application

## 📋 Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- 4GB RAM minimum (8GB recommended)
- GPU (optional, but recommended for faster inference)

## 🚀 Installation & Setup

### 1. Clone/Download the Project
```bash
cd /path/to/deepfake-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**For GPU Support (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Prepare Model File
```bash
mkdir -p model/
# Place your trained df_model.pt in the model/ directory
```

### 5. Create Upload Directory (Auto-created, but verify)
```bash
mkdir -p Uploaded_Files
```

## 🏃 Running the Application

### Development Mode
```bash
python app_enhanced.py
```

Server will start at: `http://localhost:3000`

### Production Mode (Gunicorn)
```bash
# Install gunicorn if not already installed
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:3000 --timeout 120 app_enhanced:app
```

### Production Mode (uWSGI)
```bash
pip install uwsgi

uwsgi --http :3000 --wsgi-file app_enhanced.py --callable app --processes 4 --threads 2
```

## 🔧 Configuration

### Via Environment Variables (Optional)
Create a `.env` file:
```
MODEL_PATH=model/df_model.pt
UPLOAD_FOLDER=/tmp/deepfake_uploads
LOG_LEVEL=INFO
LOG_FILE=deepfake_detection.log
```

Then load in Python (modify app_enhanced.py):
```python
from dotenv import load_dotenv
load_dotenv()
config = Config(
    MODEL_PATH=os.getenv('MODEL_PATH', 'model/df_model.pt'),
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'Uploaded_Files'),
    LOG_LEVEL=os.getenv('LOG_LEVEL', 'INFO'),
    LOG_FILE=os.getenv('LOG_FILE', 'deepfake_detection.log')
)
```

### Via Code Modifications
Edit the `Config` class in `app_enhanced.py`:
```python
@dataclass
class Config:
    # File handling
    UPLOAD_FOLDER: str = '/custom/path/uploads'
    MAX_CONTENT_LENGTH: int = 32 * 1024 * 1024  # 32MB
    
    # Model settings
    MODEL_PATH: str = '/custom/path/model.pt'
    SEQUENCE_LENGTH: int = 20
    IMAGE_SIZE: int = 112
```

## 📝 API Endpoints

### 1. Homepage
```
GET /
Returns: HTML homepage
```

### 2. Detect Deepfake
```
POST /detect
Content-Type: multipart/form-data

Request Body:
- video: (file) Video file to analyze

Response (Success - 200):
{
    "output": "REAL",
    "confidence": 95.23,
    "class": 1
}

Response (Error - 400):
{
    "error": "Invalid video: Video too short: 15 frames (minimum 20)"
}

Response (Error - 503):
{
    "error": "Model not available. Please deploy model file."
}
```

### 3. Health Check
```
GET /health

Response (200):
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda"
}
```

## 🧪 Testing the API

### Using cURL
```bash
# Health check
curl http://localhost:3000/health

# Detect deepfake
curl -X POST -F "video=@sample_video.mp4" http://localhost:3000/detect
```

### Using Python
```python
import requests

# Health check
response = requests.get('http://localhost:3000/health')
print(response.json())

# Detect deepfake
with open('sample_video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:3000/detect', files=files)
    print(response.json())
```

### Using JavaScript/Node.js
```javascript
// Using fetch API
const formData = new FormData();
const videoFile = document.getElementById('video-input').files[0];
formData.append('video', videoFile);

fetch('http://localhost:3000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## 📊 Monitoring

### Check Logs (Development)
```bash
tail -f deepfake_detection.log
```

### View Real-time Logs (Production)
```bash
# With gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 app_enhanced:app --access-logfile - --error-logfile -
```

### Parse Structured Logs (Optional)
```bash
# Extract only errors
grep "ERROR" deepfake_detection.log

# Extract detection results
grep "Detection complete" deepfake_detection.log

# Count by level
grep -o "ERROR\|WARNING\|INFO" deepfake_detection.log | sort | uniq -c
```

## 🐛 Troubleshooting

### Issue: "Model not found at model/df_model.pt"
**Solution:**
1. Verify model file exists: `ls -la model/df_model.pt`
2. Check path in Config: `MODEL_PATH: str = 'model/df_model.pt'`
3. Use absolute path if relative fails: `MODEL_PATH: str = '/absolute/path/to/model.pt'`

### Issue: "CUDA out of memory"
**Solution:**
1. Reduce video resolution (edit `_process_frame`)
2. Use CPU: Set `USE_GPU: bool = False` in Config
3. Reduce batch size
4. Close other GPU-using applications

### Issue: "No faces detected in video"
**Solution:**
1. Video quality too low - use clearer video
2. Face too small - use video with larger face
3. Extreme angles - use frontal video
4. Model will use full frame as fallback (less accurate)

### Issue: "Request timeout"
**Solution:**
1. Increase timeout in gunicorn: `--timeout 300`
2. Use faster GPU/hardware
3. Reduce sequence length (less accurate): `SEQUENCE_LENGTH: int = 10`
4. Enable GPU if available

### Issue: "Permission denied" on upload folder
**Solution:**
```bash
# Fix permissions
chmod -R 755 Uploaded_Files/
```

## 📈 Performance Optimization

### 1. Enable GPU
```python
@dataclass
class Config:
    USE_GPU: bool = True  # Requires CUDA
```

### 2. Increase Worker Processes (Production)
```bash
gunicorn -w 8 -b 0.0.0.0:3000 app_enhanced:app  # Use 2x CPU cores
```

### 3. Reduce Sequence Length (Faster, less accurate)
```python
SEQUENCE_LENGTH: int = 10  # Default: 20
```

### 4. Use HTTP Caching
Add to Flask app in production:
```python
@app.after_request
def add_headers(response):
    response.cache_control.max_age = 3600
    return response
```

## 🔒 Security Best Practices

### 1. Disable Debug Mode
```python
# ✅ Always False in production
app.run(debug=False)
```

### 2. Enable HTTPS
```bash
# Use reverse proxy like Nginx with SSL certificate
# Or use Flask-SSL:
pip install pyopenssl
```

### 3. Add Authentication
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@app.route('/detect', methods=['POST'])
@auth.login_required
def detect():
    ...
```

### 4. Limit Upload Size
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### 5. Validate File Types
```python
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
```

## 📊 Sample Deployment Architecture

```
┌─────────────────┐
│   Client (Web)  │
└────────┬────────┘
         │ HTTP/HTTPS
         ▼
┌─────────────────┐
│  Nginx/Apache   │ (Reverse Proxy)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gunicorn x4    │ (Workers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Flask App       │
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
 [Model]  [Logs]     [Uploads]
```

## 🚀 Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_enhanced.py .
COPY templates/ templates/
COPY model/ model/

EXPOSE 3000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:3000", "app_enhanced:app"]
```

Build and run:
```bash
docker build -t deepfake-detector .
docker run -p 3000:3000 -v /path/to/uploads:/app/Uploaded_Files deepfake-detector
```

## 📚 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Flask Documentation: https://flask.palletsprojects.com/
- Face Recognition Library: https://github.com/ageitgey/face_recognition
- OpenCV Documentation: https://docs.opencv.org/

## 🆘 Support & Issues

If you encounter issues:
1. Check the logs: `cat deepfake_detection.log`
2. Verify health endpoint: `curl http://localhost:3000/health`
3. Test with curl: `curl -X POST -F "video=@test.mp4" http://localhost:3000/detect`
4. Check system resources: `top` or Task Manager

## ✅ Quick Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip list | grep torch`
- [ ] Model file in place: `ls -la model/df_model.pt`
- [ ] Upload folder exists: `ls -la Uploaded_Files`
- [ ] App starts without errors: `python app_enhanced.py`
- [ ] Health check works: `curl http://localhost:3000/health`
- [ ] Can detect with test video
- [ ] Logs being written: `cat deepfake_detection.log`

---

**Last Updated:** January 2025
**Version:** 1.0
**Compatibility:** Python 3.8+, PyTorch 2.0+
