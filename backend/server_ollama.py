"""Welsh <-> English translation API backed by an Ollama model (default Qwen 3.6 27B).

MODEL_NAME picks the model; TranslateGemma models get their own prompt template, other
models the generic translator prompt used in the benchmark (backend/bench.py).

Same API as server.py:
POST /translate        {"text": "...", "source": "en"|"cy"|"auto", "target": "cy"|"en"}
POST /v1/chat/completions   minimal OpenAI-compatible passthrough
GET  /health
/asr/v1/...            proxied to the vLLM Whisper server (ASR_URL)
"""
import os
import time

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.6:27b")
IS_TG = MODEL_NAME.startswith("translategemma")
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
ASR_URL = os.environ.get("ASR_URL", "http://localhost:3242")

LANGS = {"en": "English", "cy": "Welsh"}

app = FastAPI(title="Welsh-English Translator")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

WELSH_WORDS = set(
    "yn y yr mae roedd bydd byddwn ac a'r i'r o'r ddim dw dwi ydy ydw gyda ond hefyd eich ein ei eu "
    "chi ni nhw fi ti hi fe fo efo am ar wedi bod sut beth pam ble pryd diolch fawr iawn bore da nos "
    "croeso hwn hon hyn yma yno neu os oes gan wrth rhaid cael gwneud mynd dod".split()
)


def detect(text):
    # cheap heuristic: share of very common Welsh function words, plus Welsh-only letters
    words = [w.strip(".,!?;:\"()'’") for w in text.lower().replace("’", "'").split()]
    if any(c in text.lower() for c in "ŵŷ"):
        return "cy"
    hits = sum(w in WELSH_WORDS for w in words)
    return "cy" if words and hits / len(words) >= 0.15 else "en"


def ollama_chat(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0):
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,  # stay loaded
        "options": {"temperature": temperature, "num_predict": max_new_tokens, "num_ctx": NUM_CTX},
    }
    if not IS_TG:
        body["think"] = False  # reasoning adds latency and didn't help in the benchmark
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"].strip(), d.get("eval_count", 0)


def tg_prompt(text, src, tgt):
    # TranslateGemma's documented prompt; the blank lines before the text matter
    s, t = LANGS[src], LANGS[tgt]
    return (
        f"You are a professional {s} ({src}) to {t} ({tgt}) translator. Your goal is to accurately convey "
        f"the meaning and nuances of the original {s} text while adhering to {t} grammar, vocabulary, "
        f"and cultural sensitivities.\nProduce only the {t} translation, without any additional explanations "
        f"or commentary. Please translate the following {s} text into {t}:\n\n\n{text}"
    )


def translate_messages(text, src, tgt):
    if IS_TG:
        return [{"role": "user", "content": tg_prompt(text, src, tgt)}]
    s, t = LANGS[src], LANGS[tgt]
    return [
        {"role": "system", "content": (
            f"You are a professional {s} to {t} translator. Translate the user's text into natural, fluent, "
            f"grammatically correct {t}, preserving meaning, tone, names and numbers. "
            "Output only the translation, with no notes or explanations.")},
        {"role": "user", "content": text},
    ]


class TranslateReq(BaseModel):
    text: str
    source: str = "auto"
    target: str | None = None
    max_new_tokens: int = MAX_NEW_TOKENS


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/translate")
def translate(req: TranslateReq):
    src = detect(req.text) if req.source == "auto" else req.source
    tgt = req.target or ("en" if src == "cy" else "cy")
    if src not in LANGS or tgt not in LANGS or src == tgt:
        raise HTTPException(400, "source/target must be 'en' or 'cy' and differ")
    t0, total = time.time(), 0
    # translate paragraph by paragraph: keeps formatting and matches the benchmarked setup
    out = []
    for para in req.text.split("\n"):
        if not para.strip():
            out.append(para)
            continue
        text, n = ollama_chat(translate_messages(para.strip(), src, tgt), req.max_new_tokens)
        out.append(text)
        total += n
    return {
        "source": src,
        "target": tgt,
        "translation": "\n".join(out),
        "model": MODEL_NAME,
        "tokens": total,
        "seconds": round(time.time() - t0, 2),
    }


class ChatReq(BaseModel):
    messages: list[dict]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = 0.0


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    text, n = ollama_chat(req.messages, req.max_tokens or MAX_NEW_TOKENS, req.temperature or 0.0)
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": n},
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


# ---- speech: pass-through to the vLLM Whisper server, so the UI needs one public URL ----

@app.get("/asr/v1/models")
def asr_models():
    r = requests.get(f"{ASR_URL}/v1/models", timeout=10)
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@app.post("/asr/v1/audio/transcriptions")
async def asr_transcribe(request: Request):
    form = await request.form()
    files, data = {}, {}
    for k, v in form.multi_items():
        if hasattr(v, "read"):
            files[k] = (v.filename or "audio.wav", await v.read(), v.content_type or "audio/wav")
        else:
            data[k] = v
    r = await run_in_threadpool(requests.post, f"{ASR_URL}/v1/audio/transcriptions", files=files, data=data, timeout=120)
    return Response(r.content, status_code=r.status_code, media_type="application/json")
