# Cyfieithu – Welsh ⇄ English translation POC

Static single-page UI (`index.html`, served by GitHub Pages) with live speech transcription, an editable source text box and auto-translation.

- **Translation:** Qwen 3.6 27B (Ollama `qwen3.6:27b`, Q4_K_M, ~22 GB VRAM) behind `backend/server_ollama.py` (FastAPI, port 10083). Chosen from the benchmark below; set `MODEL_NAME` to serve another Ollama model (TranslateGemma models get their own prompt template).
- **Speech:** vLLM `distil-whisper/distil-medium.en` (English speech only), proxied by the same backend under `/asr/v1/...`.
- Default backend: `https://dev.studio.bluemesh.ai/welsh`. Override with `?api=https://host/path&asr=https://host/path/asr` or in *Backend settings*.

## Benchmark

[`report/welsh_mt_benchmark.pdf`](report/welsh_mt_benchmark.pdf) compares 8 systems on FLORES-200 devtest (100 sentences per direction; COMET, chrF++, BLEU, paired-bootstrap significance).

| System | VRAM | COMET en→cy | COMET cy→en |
|---|---|---|---|
| Techiaith NLLB 1.3B fine-tunes (CC-BY-NC) | 3 GB | 87.3 | 88.4 |
| NLLB-200 3.3B (CC-BY-NC) | 7 GB | 86.8 | 88.8 |
| **Qwen 3.6 27B** (deployed) | 22 GB | 85.5 | 89.9 |
| TranslateGemma 27B | 32 GB | 84.3 | 89.0 |
| Gemma 4 31B-it (int8 / Q4) | 32 / 25 GB | 77.2 / 76.7 | 88.4 / 88.6 |
| TranslateGemma 12B / 4B | 10 / 4.4 GB | 76.6 / 48.9 | 87.1 / 80.5 |

Reproduce: `backend/bench.py <system> 100` per system, `backend/score.py <systems...>`, then `report/build_report.py --data <dir>` (needs reportlab + matplotlib and the `hyps_*.json` / `scores.json` outputs).

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
ollama pull qwen3.6:27b
screen -dmS welsh-api ./run_ollama.sh   # port 10083; MODEL_NAME=translategemma:27b-it-q8_0 ./run_ollama.sh to switch back
```

`backend/server.py` + `welsh-mt.service` is the alternative pure-transformers backend (Gemma 4 31B-it, bitsandbytes 8-bit, port 10042).
