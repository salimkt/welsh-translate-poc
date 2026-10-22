#!/usr/bin/env bash
cd /home/cloudadmin/welsh-mt
exec .venv/bin/uvicorn server_tg:app --host 0.0.0.0 --port 10083 --workers 1
