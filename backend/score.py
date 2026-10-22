"""Score hyps_*.json with BLEU, chrF++ and COMET (wmt22-comet-da)."""
import json
import sys

import sacrebleu
from comet import download_model, load_from_checkpoint

comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
rows = []
for system in sys.argv[1:]:
    d = json.load(open(f"hyps_{system}.json"))
    for direction in ("en-cy", "cy-en"):
        x = d[direction]
        bleu = sacrebleu.corpus_bleu(x["hyp"], [x["ref"]]).score
        chrf = sacrebleu.corpus_chrf(x["hyp"], [x["ref"]], word_order=2).score
        c = comet.predict(
            [{"src": s, "mt": h, "ref": r} for s, h, r in zip(x["src"], x["hyp"], x["ref"])],
            batch_size=16, gpus=1, progress_bar=False,
        ).system_score
        rows.append((system, direction, bleu, chrf, c * 100, x["secs"] / len(x["hyp"])))

print(f"{'system':8} {'dir':6} {'BLEU':>6} {'chrF++':>7} {'COMET':>6} {'s/sent':>7}")
for r in rows:
    print(f"{r[0]:8} {r[1]:6} {r[2]:6.1f} {r[3]:7.1f} {r[4]:6.1f} {r[5]:7.2f}")
