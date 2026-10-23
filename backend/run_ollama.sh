#!/usr/bin/env bash
# Translation API on :10083 (Ollama backend; override the model with MODEL_NAME=...)
cd /home/cloudadmin/welsh-mt
export MODEL_NAME=${MODEL_NAME:-qwen3.6:27b}
exec .venv/bin/uvicorn server_ollama:app --host 0.0.0.0 --port 10083 --workers 1
