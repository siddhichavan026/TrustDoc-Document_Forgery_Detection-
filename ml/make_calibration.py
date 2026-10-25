"""Confidence calibration from held-out predictions. Python standard library only (no torch / pandas).

Run from the project root (D:\\TrustDoc):
    python ml\\make_calibration.py
    python ml\\make_calibration.py seg_resnet34 seg_resnet34_syn_test_preds.csv

Reads   results/<tag>_metrics.json     (validated decision threshold)
        results/<preds>.csv            (id,label,score for the held-out synthetic test set)
Writes  weights/<tag>_calibration.json

Why bins instead of a smooth curve: the U-Net's image scores are saturated (almost always ~0 or ~1),
so a logistic curve fits badly. Instead, confidence = the measured fraction of forged images
inside each score range, on data the model never trained on.
(Display-only: no reported metric depends on this file.)
"""
import csv
import json
import os
import sys
from collections import defaultdict


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "seg_resnet34"
    preds_name = sys.argv[2] if len(sys.argv) > 2 else f"{tag}_syn_test_preds.csv"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rdir, wdir = os.path.join(root, "results"), os.path.join(root, "weights")
    os.makedirs(wdir, exist_ok=True)

    thr = float(json.load(open(os.path.join(rdir, f"{tag}_metrics.json")))["threshold"])
    with open(os.path.join(rdir, preds_name), newline="") as f:
        rows = [(r["id"], int(float(r["label"])), float(r["score"])) for r in csv.DictReader(f)]
    print(f"{len(rows)} held-out predictions, decision threshold {thr:.4f}")

    # ---- bins: below threshold | threshold..0.5 | 0.5..0.9 | 0.9..1
    edges = sorted(set([0.0, thr, 0.5, 0.9]))
    edges = [e for e in edges if e < 1.0] + [1.0000001]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = [(i, l, s) for i, l, s in rows if lo <= s < hi]
        nf = sum(l for _, l, _ in inb)
        p = (nf + 1) / (len(inb) + 2)  # Laplace smoothing so an empty/pure bin never says 0% or 100%
        bins.append(dict(lo=lo, hi=hi, n=len(inb), n_forged=nf, p_forged=round(p, 4)))
        print(f"  score {lo:.3f}-{min(hi, 1.0):.3f}: {len(inb):4d} images, {nf:4d} forged -> P(forged)={p:.2f}")

    # ---- overall behaviour at the threshold
    tp = sum(1 for _, l, s in rows if l == 1 and s >= thr)
    fn = sum(1 for _, l, s in rows if l == 1 and s < thr)
    fp = sum(1 for _, l, s in rows if l == 0 and s >= thr)
    tn = sum(1 for _, l, s in rows if l == 0 and s < thr)
    div = lambda a, b: a / b if b else float("nan")
    print(f"\nAt the threshold: TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  recall={div(tp, tp + fn):.2f}  false-positive rate={div(fp, fp + tn):.2f}  "
          f"precision (confidence when 'Forged')={div(tp, tp + fp):.2f}  "
          f"NPV (confidence when 'Authentic')={div(tn, tn + fn):.2f}")

    # ---- error analysis: are errors tied to particular source receipts? (id looks like b0539_f1 / b0513_a1)
    by_base = defaultdict(lambda: {"a": [], "f": []})
    for i, l, s in rows:
        base, kind = i.split("_")[0], ("f" if l == 1 else "a")
        by_base[base][kind].append(s >= thr)
    a_bases = {b: v["a"] for b, v in by_base.items() if v["a"]}
    flagged_any = [b for b, v in a_bases.items() if any(v)]
    flagged_all = [b for b, v in a_bases.items() if all(v)]
    print(f"\nAuthentic false positives: {len(flagged_any)} of {len(a_bases)} source receipts have >=1 variant flagged; "
          f"{len(flagged_all)} have ALL variants flagged")
    print("  (many 'all variants' = the model reacts to a property of that receipt itself, e.g. handwriting/stamps; "
          "report this as a limitation)")
    if flagged_all:
        print("  receipts always flagged:", ", ".join(sorted(flagged_all)[:15]))

    out = os.path.join(wdir, f"{tag}_calibration.json")
    cal = {"threshold": thr, "slope": 20.0,
           "bin_edges": [b["lo"] for b in bins] + [bins[-1]["hi"]],   # format read by infer.py
           "bin_p_forged": [b["p_forged"] for b in bins],
           "bins": bins}                                              # same information, human-readable
    json.dump(cal, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
