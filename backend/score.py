"""Score hyps_<system>.json files with BLEU, chrF++ and COMET (wmt22-comet-da).

usage: python score.py <system> [<system> ...]
Prints a table and writes scores.json (system/corpus scores + per-sentence COMET).
"""
import json
import sys

import sacrebleu
from comet import download_model, load_from_checkpoint

comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
rows, detail = [], {}
for system in sys.argv[1:]:
    d = json.load(open(f"hyps_{system}.json"))
    detail[system] = {}
    for direction in ("en-cy", "cy-en"):
        x = d[direction]
        bleu = sacrebleu.corpus_bleu(x["hyp"], [x["ref"]]).score
        chrf = sacrebleu.corpus_chrf(x["hyp"], [x["ref"]], word_order=2).score
        c = comet.predict(
            [{"src": s, "mt": h, "ref": r} for s, h, r in zip(x["src"], x["hyp"], x["ref"])],
            batch_size=16, gpus=1, progress_bar=False,
        )
        secs = x["secs"] / len(x["hyp"])
        rows.append((system, direction, bleu, chrf, c.system_score * 100, secs))
        detail[system][direction] = {"bleu": bleu, "chrf": chrf, "comet": c.system_score * 100,
                                     "secs_per_sent": secs, "comet_sent": [s * 100 for s in c.scores]}

json.dump(detail, open("scores.json", "w"), indent=1)
print(f"{'system':10} {'dir':6} {'BLEU':>6} {'chrF++':>7} {'COMET':>6} {'s/sent':>7}")
for r in rows:
    print(f"{r[0]:10} {r[1]:6} {r[2]:6.1f} {r[3]:7.1f} {r[4]:6.1f} {r[5]:7.2f}")
