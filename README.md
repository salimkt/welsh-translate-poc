# Cyfieithu – Welsh ⇄ English translation POC

Static single-page UI (`index.html`, served by GitHub Pages) with live speech transcription, an editable source text box and auto-translation.

- **Translation:** `google/translategemma-27b-it` (Ollama `translategemma:27b-it-q8_0`, ~32 GB VRAM) behind `backend/server_tg.py` (FastAPI, port 10083).
- **Speech:** vLLM `distil-whisper/distil-medium.en` (English speech only), proxied by the same backend under `/asr/v1/...`.
- Default backend: `https://dev.studio.bluemesh.ai/welsh`. Override with `?api=https://host/path&asr=https://host/path/asr` or in *Backend settings*.

## Backend API

| Method | Path | Body |
|---|---|---|
| GET | `/health` | – |
| POST | `/translate` | `{"text": "...", "source": "en\|cy\|auto", "target": "cy\|en"}` |
| POST | `/v1/chat/completions` | OpenAI-style messages |
| GET | `/asr/v1/models` | proxied to Whisper |
| POST | `/asr/v1/audio/transcriptions` | multipart, proxied to Whisper |

## Run the backend

```bash
cd ~/welsh-mt
uv venv .venv && uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" requests python-multipart
ollama pull translategemma:27b-it-q8_0
screen -dmS welsh-tg ./run_tg.sh   # port 10083
```

`backend/server.py` + `welsh-mt.service` is the alternative pure-transformers backend (Gemma 4 31B-it, bitsandbytes 8-bit, port 10042). `bench.py` / `score.py` run the FLORES-200 en↔cy comparison (BLEU, chrF++, COMET).
