"""Run TrustDoc on every receipt in a folder; save one result picture each and print timing.

From the project root (D:\\TrustDoc):
    python ml\\demo_cli.py demo_samples results\\demo_out
    python ml\\demo_cli.py demo_samples results\\demo_out seg_resnet34
Outputs: <out_dir>/<name>_result.png (original | overlay | heatmap | Grad-CAM) and <out_dir>/summary.csv
"""
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402

EXT = (".jpg", ".jpeg", ".png")


def panel(res, h=700):
    tiles = []
    for key, title in [("original", "original"), ("overlay", "suspicious regions"),
                       ("heatmap", "forgery probability"), ("gradcam", "model attention (Grad-CAM)")]:
        im = res[key] if res[key] is not None else np.full_like(res["original"], 255)
        s = h / float(im.shape[0])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    body = np.hstack(tiles)
    banner = np.full((40, body.shape[1], 3), 245, np.uint8)
    text = f"{res['label'].upper()}   confidence {res['confidence']:.0%}   score {res['score']:.3f}   regions {len(res['boxes'])}"
    cv2.putText(banner, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 200) if res["forged"] else (0, 140, 0), 2, cv2.LINE_AA)
    return np.vstack([banner, body])


def run(an, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(EXT))
    if not files:
        print("no images found in", in_dir)
        return []
    rows = []
    for f in files:
        img = cv2.imread(os.path.join(in_dir, f))
        if img is None:
            print("  skipped (unreadable):", f)
            continue
        t0 = time.time()
        res = an.analyze(img)
        sec = time.time() - t0
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(f)[0] + "_result.png"), panel(res))
        rows.append((f, res["label"], round(res["score"], 4), round(res["confidence"], 3), len(res["boxes"]),
                     round(sec, 2), res["gradcam_error"] or ""))
        print(f"{f:45s} -> {res['label']:9s} confidence {res['confidence']:.0%}  score {res['score']:.3f}  "
              f"regions {len(res['boxes'])}  {sec:.1f}s" + (f"  [gradcam: {res['gradcam_error']}]" if res["gradcam_error"] else ""))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "verdict", "score", "confidence", "regions", "seconds", "gradcam_error"])
        w.writerows(rows)
    if rows:
        print(f"\n{len(rows)} receipts, mean {np.mean([r[5] for r in rows]):.1f}s each. Pictures + summary.csv in {out_dir}")
    return rows


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "seg_resnet34"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    an = infer.TrustDocAnalyzer.from_dir(os.path.join(root, "weights"), tag)
    print(f"model {an.encoder} | threshold {an.threshold:.4f} | device {an.dev}\n")
    run(an, in_dir, out_dir)


if __name__ == "__main__":
    main()
