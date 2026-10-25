"""Plots and tables for the report, from the prediction files you already downloaded. Runs locally, no GPU.

From the project root (D:\\TrustDoc):   python ml\\report_plots.py
Reads   results/seg_resnet34_{syn,real}_test_preds.csv, results/cls_<name>_{syn,real}_test_preds.csv,
        results/seg_resnet34_metrics.json
Writes  results/report/roc_curves.png, confusion_matrices.png, model_comparison.csv  (and prints the table)
Needs only numpy + matplotlib (both installed with grad-cam).
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "report")
MODELS = [("U-Net (ResNet34)", "seg_resnet34", "score"), ("ResNet50", "cls_resnet50", "prob_forged"),
          ("EfficientNet-B0", "cls_efficientnet_b0", "prob_forged"), ("Small CNN", "cls_small_cnn", "prob_forged")]
SETS = [("syn_test", "Synthetic test (computer-made edits)"), ("real_test", "Real test (Find It Again)")]


def load(tag, split, col):
    p = os.path.join(RES, f"{tag}_{split}_preds.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        return None
    return np.array([float(r["label"]) for r in rows]).astype(int), np.array([float(r[col]) for r in rows])


def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # handle tied scores
    tps, fps = np.cumsum(y)[distinct], (1 + distinct) - np.cumsum(y)[distinct]
    tpr, fpr = np.r_[0, tps / max(y.sum(), 1)], np.r_[0, fps / max((1 - y).sum(), 1)]
    return fpr, tpr, float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    thr = float(json.load(open(os.path.join(RES, "seg_resnet34_metrics.json")))["threshold"])

    # ---- ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    table = {name: {} for name, _, _ in MODELS}
    for ax, (split, title) in zip(axes, SETS):
        for name, tag, col in MODELS:
            d = load(tag, split, col)
            if d is None:
                continue
            fpr, tpr, auc = roc(*d)
            table[name][split] = auc
            ax.plot(fpr, tpr, lw=2.2 if "U-Net" in name else 1.4, label=f"{name} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=180)
    plt.close(fig)

    # ---- confusion matrices of the final model at the validated threshold
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    extra = {}
    for ax, (split, title) in zip(axes, SETS):
        d = load("seg_resnet34", split, "score")
        if d is None:
            ax.axis("off")
            continue
        y, s = d
        p = (s >= thr).astype(int)
        cm = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()],
                       [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black", fontsize=12)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Authentic", "Forged"], yticklabels=["Authentic", "Forged"],
               xlabel="Predicted", ylabel="Actual", title=title, )
        ax.title.set_fontsize(9)
        tp, fn, fp, tn = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
        extra[split] = dict(recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), acc=(tp + tn) / cm.sum(),
                            precision=tp / max(tp + fp, 1), n=int(cm.sum()))
    fig.suptitle(f"U-Net confusion matrices (threshold {thr:.4f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=180)
    plt.close(fig)

    # ---- comparison table
    rows = []
    for name, _, _ in MODELS:
        r = table[name]
        if r:
            rows.append([name, r.get("syn_test", float("nan")), r.get("real_test", float("nan"))])
    with open(os.path.join(OUT, "model_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "synthetic_test_auc", "real_test_auc"])
        w.writerows([[n, f"{a:.3f}", f"{b:.3f}"] for n, a, b in rows])
    print("| Model | AUC synthetic test | AUC real test |\n|---|---|---|")
    for n, a, b in rows:
        print(f"| {n} | {a:.3f} | {b:.3f} |")
    print(f"\nU-Net at threshold {thr:.4f}:")
    for split, e in extra.items():
        print(f"  {split}: n={e['n']} recall={e['recall']:.2f} false-positive rate={e['fpr']:.2f} "
              f"precision={e['precision']:.2f} accuracy={e['acc']:.2f}")
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
