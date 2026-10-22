"""Welsh <-> English translation API (HF transformers).

POST /translate        {"text": "...", "source": "en"|"cy"|"auto", "target": "cy"|"en"}
POST /v1/chat/completions   minimal OpenAI-compatible passthrough
GET  /health
"""
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/cloudadmin/gemma4_31b")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-4-31b-it")
QUANT = os.environ.get("QUANT", "8bit")  # 8bit (near-lossless) | 4bit (smallest) | none (bf16)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

LANGS = {"en": "English", "cy": "Welsh (Cymraeg)"}
SKIP = ["vision_tower", "multi_modal_projector", "embed_vision", "lm_head"]

quant_config = None
if QUANT == "4bit":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # keep the vision tower out of quantization; we only use text
        llm_int8_skip_modules=SKIP,
    )
elif QUANT == "8bit":
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": 0},  # single GPU, pick it with CUDA_VISIBLE_DEVICES
    quantization_config=quant_config,
)
model.eval()
# one generation at a time; HF generate isn't safe to run concurrently on one model
lock = threading.Lock()

app = FastAPI(title="Welsh-English Translator")


def generate(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    with lock, torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
        del inputs
        torch.cuda.empty_cache()
    new = out[0, prompt_len:]
    return tokenizer.decode(new, skip_special_tokens=True).strip(), len(new)


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
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional {LANGS[src]} to {LANGS[tgt]} translator. "
                f"Translate the user's text into natural, fluent, grammatically correct {LANGS[tgt]}, "
                "preserving meaning, tone, formatting, names and numbers. "
                f"Never leave words untranslated from {LANGS[src]}; apply correct mutations. "
                "Output only the translation, with no notes or explanations."
            ),
        },
        {"role": "user", "content": req.text},
    ]
    t0 = time.time()
    translation, n = generate(messages, req.max_new_tokens)
    return {
        "source": src,
        "target": tgt,
        "translation": translation,
        "tokens": n,
        "seconds": round(time.time() - t0, 2),
    }


class ChatReq(BaseModel):
    messages: list[dict]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = 0.0


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    text, n = generate(req.messages, req.max_tokens or MAX_NEW_TOKENS, req.temperature or 0.0)
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
