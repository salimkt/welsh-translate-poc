#!/usr/bin/env bash
cd /home/cloudadmin/welsh-mt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export QUANT=${QUANT:-8bit}
export MODEL_PATH=${MODEL_PATH:-/home/cloudadmin/gemma4_31b}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10042 --workers 1
