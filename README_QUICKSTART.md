# DeepTrace Quickstart

1. `git clone https://github.com/obstinix/deepfake_recognition.git`
2. `cd deepfake_recognition`
3. `pip install -r requirements.txt`
4. `bash start.sh` (or `start.bat` on Windows)
5. Open `http://localhost:8000` in your browser
6. (Optional) Drop trained checkpoint into `checkpoints/resnet18/best.pth` and hit the reload API:
   ```bash
   curl -X POST http://localhost:8000/api/model/reload \
        -H "Content-Type: application/json" \
        -d '{"checkpoint_path": "checkpoints/resnet18/best.pth"}'
   ```
