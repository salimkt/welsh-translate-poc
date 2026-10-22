"""Translate FLORES+ devtest en<->cy with each system; write hyps to JSON.

usage: python bench.py <system> <n>     system = gemma4 | tg
"""
import json
import sys
import time

import requests

SYSTEM, N = sys.argv[1], int(sys.argv[2])
NAMES = {"en": "English", "cy": "Welsh"}


def load(lang):
    # public FLORES-200 release, extracted to ./flores200_dataset
    return open(f"flores200_dataset/devtest/{lang}.devtest", encoding="utf-8").read().splitlines()


en, cy = load("eng_Latn"), load("cym_Latn")
idx = list(range(0, len(en), len(en) // N))[:N]  # evenly spread sample
data = {"en": [en[i] for i in idx], "cy": [cy[i] for i in idx]}


def gemma4(text, src, tgt):
    r = requests.post("http://localhost:10042/translate", json={"text": text, "source": src, "target": tgt}, timeout=600)
    return r.json()["translation"]


def tg(text, src, tgt):
    s, t = NAMES[src], NAMES[tgt]
    prompt = (
        f"You are a professional {s} ({src}) to {t} ({tgt}) translator. Your goal is to accurately convey "
        f"the meaning and nuances of the original {s} text while adhering to {t} grammar, vocabulary, "
        f"and cultural sensitivities.\nProduce only the {t} translation, without any additional explanations "
        f"or commentary. Please translate the following {s} text into {t}:\n\n\n{text}"
    )
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "translategemma:27b-it-q8_0",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 512},
        },
        timeout=600,
    )
    return r.json()["message"]["content"].strip()


fn = {"gemma4": gemma4, "tg": tg}[SYSTEM]
out = {"idx": idx}
for src, tgt in (("en", "cy"), ("cy", "en")):
    hyps, t0 = [], time.time()
    for i, s in enumerate(data[src]):
        hyps.append(fn(s, src, tgt))
        if i % 10 == 0:
            print(SYSTEM, src, tgt, i, round(time.time() - t0), flush=True)
    out[f"{src}-{tgt}"] = {"src": data[src], "ref": data[tgt], "hyp": hyps, "secs": round(time.time() - t0, 1)}
json.dump(out, open(f"hyps_{SYSTEM}.json", "w"), ensure_ascii=False, indent=1)
print("done", SYSTEM)
