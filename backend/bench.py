"""Translate a FLORES-200 devtest sample en<->cy with one system; write hyps_<system>.json.

usage: python bench.py <system> <n>       (see SYSTEMS below)
Expects ./flores200_dataset/devtest/{eng,cym}_Latn.devtest (public FLORES-200 release).
"""
import json
import sys
import time

import requests

NAMES = {"en": "English", "cy": "Welsh"}
FLORES = {"en": "eng_Latn", "cy": "cym_Latn"}

# name -> (kind, config)
SYSTEMS = {
    "gemma4": ("http", "http://localhost:10042/translate"),  # transformers server.py, Gemma 4 31B 8-bit
    "tg": ("ollama", "translategemma:27b-it-q8_0"),
    "tg12": ("ollama", "translategemma:12b"),
    "tg4": ("ollama", "translategemma:4b"),
    "gemma4_q4": ("ollama", "gemma4:31b"),
    "qwen36": ("ollama", "qwen3.6:27b"),
    "nllb33": ("nllb", {"en-cy": "facebook/nllb-200-3.3B", "cy-en": "facebook/nllb-200-3.3B"}),
    "techiaith": ("nllb", {"en-cy": "techiaith/nllb-200-1.3B-ft-eng-to-cym-legislation",
                           "cy-en": "techiaith/nllb-200-1.3B-ft-cym-to-eng"}),
}

SYSTEM, N = sys.argv[1], int(sys.argv[2])
KIND, CONF = SYSTEMS[SYSTEM]


def load(lang):
    return open(f"flores200_dataset/devtest/{FLORES[lang]}.devtest", encoding="utf-8").read().splitlines()


en, cy = load("en"), load("cy")
idx = list(range(0, len(en), len(en) // N))[:N]  # evenly spread sample
data = {"en": [en[i] for i in idx], "cy": [cy[i] for i in idx]}


def tg_prompt(text, src, tgt):
    # TranslateGemma's documented prompt; the blank lines before the text matter
    s, t = NAMES[src], NAMES[tgt]
    return (
        f"You are a professional {s} ({src}) to {t} ({tgt}) translator. Your goal is to accurately convey "
        f"the meaning and nuances of the original {s} text while adhering to {t} grammar, vocabulary, "
        f"and cultural sensitivities.\nProduce only the {t} translation, without any additional explanations "
        f"or commentary. Please translate the following {s} text into {t}:\n\n\n{text}"
    )


def llm_messages(text, src, tgt):
    s, t = NAMES[src], NAMES[tgt]
    return [
        {"role": "system", "content": (
            f"You are a professional {s} to {t} translator. Translate the user's text into natural, fluent, "
            f"grammatically correct {t}, preserving meaning, tone, names and numbers. "
            "Output only the translation, with no notes or explanations.")},
        {"role": "user", "content": text},
    ]


def via_http(text, src, tgt):
    r = requests.post(CONF, json={"text": text, "source": src, "target": tgt}, timeout=600)
    return r.json()["translation"]


def via_ollama(text, src, tgt):
    msgs = ([{"role": "user", "content": tg_prompt(text, src, tgt)}]
            if CONF.startswith("translategemma") else llm_messages(text, src, tgt))
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": CONF, "messages": msgs, "stream": False, "think": False, "keep_alive": "5m",
              "options": {"temperature": 0, "num_predict": 512, "num_ctx": 4096}},
        timeout=600,
    )
    return r.json()["message"]["content"].strip()


_nllb = {}


def via_nllb(text, src, tgt):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    repo = CONF[f"{src}-{tgt}"]
    if repo not in _nllb:
        _nllb.clear()
        torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(repo, src_lang=FLORES[src])
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.float16).cuda().eval()
        _nllb[repo] = (tok, model)
    tok, model = _nllb[repo]
    tok.src_lang = FLORES[src]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(FLORES[tgt]),
                             num_beams=4, max_new_tokens=512)
    return tok.decode(out[0], skip_special_tokens=True).strip()


fn = {"http": via_http, "ollama": via_ollama, "nllb": via_nllb}[KIND]
out = {"system": SYSTEM, "kind": KIND, "config": CONF, "idx": idx}
for src, tgt in (("en", "cy"), ("cy", "en")):
    hyps, t0 = [], time.time()
    for i, s in enumerate(data[src]):
        hyps.append(fn(s, src, tgt))
        if i % 10 == 0:
            print(SYSTEM, src, tgt, i, round(time.time() - t0), flush=True)
    out[f"{src}-{tgt}"] = {"src": data[src], "ref": data[tgt], "hyp": hyps, "secs": round(time.time() - t0, 1)}
json.dump(out, open(f"hyps_{SYSTEM}.json", "w"), ensure_ascii=False, indent=1)
print("done", SYSTEM)
