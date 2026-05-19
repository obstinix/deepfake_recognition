@echo off
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
