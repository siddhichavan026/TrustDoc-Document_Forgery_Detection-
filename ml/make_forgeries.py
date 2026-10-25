"""TrustDoc dataset builder.

1. collect_base_pool()  - authentic base receipts, de-duplicated against the Find-It-Again TEST split
2. build_synthetic()    - authentic + forged variants with pixel masks (split by base image)
3. build_real()         - Find-It-Again converted to images + masks (real, human-made forgeries)
4. preview()            - quick visual check
"""
import ast
import glob
import os
import random
import shutil
from multiprocessing import Pool

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

MAX_SIDE = 1024
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONTS = [os.path.join(_FONT_DIR, f) for f in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "DejaVuSerif.ttf")
         if os.path.exists(os.path.join(_FONT_DIR, f))]
OPS = ["replace", "copy_move", "splice", "erase"]
OP_P = [0.4, 0.2, 0.2, 0.2]
IMG_EXT = ("*.jpg", "*.jpeg", "*.png")


# --------------------------------------------------------------------------- helpers
def load_resized(path, max_side=MAX_SIDE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def dhash(path, size=16):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def list_images(root):
    out = []
    for pat in IMG_EXT:
        out += glob.glob(os.path.join(root, "**", pat), recursive=True)
    return sorted(out)


def find_text_boxes(img):
    """Find word/line-sized text blobs with simple morphology (no OCR needed)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(conn, connectivity=8)
    H, W = gray.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if 9 <= h <= 40 and 18 <= w <= 300 and area / float(w * h) > 0.3 \
                and x > 3 and y > 3 and x + w < W - 3 and y + h < H - 3:
            boxes.append((x, y, w, h))
    return boxes


def _pad(box, W, H, p=2):
    x, y, w, h = box
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(W, x + w + p), min(H, y + h + p)
    return (x0, y0, x1 - x0, y1 - y0)


def _colors(patch):
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).reshape(-1)
    px = patch.reshape(-1, 3)
    order = np.argsort(g)
    k = max(1, len(g) // 20)
    ink = np.median(px[order[:k]], axis=0)
    bg = np.median(px[order[len(g) // 2:]], axis=0)
    return ink.astype(np.float32), bg.astype(np.float32)


# --------------------------------------------------------------------------- forgery operations
def _erase(img, box):
    x, y, w, h = box
    patch = img[y:y + h, x:x + w]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    img[y:y + h, x:x + w] = cv2.inpaint(patch, m, 3, cv2.INPAINT_TELEA)


def _random_string(n, rng):
    if rng.random() < 0.55:
        s = "".join(rng.choice(list("0123456789"), size=n))
        if n >= 4 and rng.random() < 0.7:
            p = int(rng.integers(1, n - 1))
            s = s[:p] + "." + s[p + 1:]
        return s
    return "".join(rng.choice(list("ABCDEFGHIJKLMNOPRSTUVWXY"), size=n))


def _replace_text(img, box, rng):
    x, y, w, h = box
    ink, _ = _colors(img[y:y + h, x:x + w])
    _erase(img, box)
    patch = img[y:y + h, x:x + w].astype(np.float32)
    font = ImageFont.truetype(FONTS[int(rng.integers(len(FONTS)))], max(8, int(h * 0.95)))
    n = max(2, int(w / (0.6 * font.size)))
    s = _random_string(n, rng)
    while len(s) > 1 and font.getlength(s) > w:
        n -= 1
        s = _random_string(n, rng)
    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    tb = d.textbbox((0, 0), s, font=font)
    d.text((-tb[0] + 1, (h - (tb[3] - tb[1])) // 2 - tb[1]), s, fill=255, font=font)
    a = cv2.GaussianBlur(np.asarray(canvas, np.float32) / 255.0, (3, 3), 0.5)
    a = np.clip(a * 1.5, 0, 1)[..., None]  # thin strokes: keep ink as dark as real print
    img[y:y + h, x:x + w] = np.clip(patch * (1 - a) + ink * a, 0, 255).astype(np.uint8)


def _paste(img, box, src_patch):
    x, y, w, h = box
    src = cv2.resize(src_patch, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    dst = img[y:y + h, x:x + w].astype(np.float32)
    _, bg_d = _colors(img[y:y + h, x:x + w])
    _, bg_s = _colors(src_patch)
    src = np.clip(src + (bg_d - bg_s), 0, 255)
    a = np.ones((h, w), np.float32)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0.5
    a = cv2.GaussianBlur(a, (3, 3), 0)[..., None]
    img[y:y + h, x:x + w] = np.clip(src * a + dst * (1 - a), 0, 255).astype(np.uint8)


def make_forged(img, boxes, donor, rng):
    """Returns (forged_img, mask, op_names) or None."""
    if len(boxes) < 6:
        return None
    H, W = img.shape[:2]
    out = img.copy()
    mask = np.zeros((H, W), np.uint8)
    k = min(int(rng.integers(1, 5)), len(boxes) // 2)
    idx = rng.permutation(len(boxes))
    targets = [boxes[i] for i in idx[:k]]
    sources = [boxes[i] for i in idx[k:]]
    ops = []
    for t in targets:
        t = _pad(t, W, H)
        x, y, w, h = t
        op = str(rng.choice(OPS, p=OP_P))
        if op == "splice" and not (donor and len(donor[1]) > 0):
            op = "replace"
        if op == "replace":
            _replace_text(out, t, rng)
        elif op == "erase":
            _erase(out, t)
        elif op == "copy_move":
            sx, sy, sw, sh = sources[int(rng.integers(len(sources)))]
            _paste(out, t, img[sy:sy + sh, sx:sx + sw])
        else:  # splice from another document
            dimg, dboxes = donor
            sx, sy, sw, sh = dboxes[int(rng.integers(len(dboxes)))]
            _paste(out, t, dimg[sy:sy + sh, sx:sx + sw])
        mask[y:y + h, x:x + w] = 255
        ops.append(op)
    return out, mask, ops


def finalize(img, rng):
    """Benign 'capture' effects, applied to BOTH classes so the model cannot cheat on them."""
    out = img.astype(np.float32)
    if rng.random() < 0.7:
        out = out * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    if rng.random() < 0.3:
        out = out + rng.normal(0, rng.uniform(1, 4), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.4, 0.9))
    return out


def save_image(img, path_no_ext, rng):
    if rng.random() < 0.25:
        p = path_no_ext + ".png"
        cv2.imwrite(p, img)
        return p, 100
    q = int(rng.integers(55, 96))
    p = path_no_ext + ".jpg"
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return p, q


# --------------------------------------------------------------------------- 1. base pool
def collect_base_pool(findit_root, voxel_root, hash_thresh=20):
    """Authentic base images = Voxel51 + Find-It-Again train/val authentic,
    minus anything that near-duplicates a Find-It-Again TEST image (prevents leakage)."""
    test_df = pd.read_csv(os.path.join(findit_root, "test.txt"))
    test_hashes = np.array([dhash(os.path.join(findit_root, "test", n)) for n in test_df["image"]
                            if os.path.exists(os.path.join(findit_root, "test", n))])
    cands = list_images(voxel_root)
    n_voxel = len(cands)
    for split in ("train", "val"):
        df = pd.read_csv(os.path.join(findit_root, f"{split}.txt"))
        for _, r in df.iterrows():
            if int(r["forged"]) == 0:
                p = os.path.join(findit_root, split, r["image"])
                if os.path.exists(p):
                    cands.append(p)
    kept, kept_h = [], []
    stats = dict(candidates=len(cands), voxel=n_voxel, dropped_test_overlap=0, dropped_duplicate=0, unreadable=0)
    for p in tqdm(cands, desc="hashing"):
        h = dhash(p)
        if h is None:
            stats["unreadable"] += 1
            continue
        if len(test_hashes) and np.count_nonzero(test_hashes != h, axis=1).min() <= hash_thresh:
            stats["dropped_test_overlap"] += 1
            continue
        if kept_h and np.count_nonzero(np.array(kept_h) != h, axis=1).min() <= hash_thresh:
            stats["dropped_duplicate"] += 1
            continue
        kept.append(p)
        kept_h.append(h)
    stats["kept"] = len(kept)
    return kept, stats


# --------------------------------------------------------------------------- 2. synthetic set
def _process_base(args):
    idx, path, donor_path, out_dir, split, k, seed = args
    rng = np.random.default_rng(seed + idx)
    recs = []
    img = load_resized(path)
    if img is None:
        return recs
    boxes = find_text_boxes(img)
    if len(boxes) < 6:
        return recs
    donor = None
    if donor_path:
        dimg = load_resized(donor_path)
        if dimg is not None:
            donor = (dimg, find_text_boxes(dimg))
    stem = f"b{idx:04d}"
    for v in range(k):
        for label in (0, 1):
            if label == 0:
                im, mask, ops = img, np.zeros(img.shape[:2], np.uint8), []
            else:
                res = make_forged(img, boxes, donor, rng)
                if res is None:
                    continue
                im, mask, ops = res
            im = finalize(im, rng)
            name = f"{stem}_{'f' if label else 'a'}{v}"
            p, q = save_image(im, os.path.join(out_dir, "images", split, name), rng)
            cv2.imwrite(os.path.join(out_dir, "masks", split, name + ".png"), mask)
            recs.append(dict(id=name, split=split, label=label, base=stem, ops="|".join(ops),
                             n_regions=len(ops), jpeg_q=q, image=os.path.relpath(p, out_dir),
                             mask=f"masks/{split}/{name}.png"))
    return recs


def build_synthetic(base_paths, out_dir, k=3, seed=42, workers=2, limit=None):
    rnd = random.Random(seed)
    paths = sorted(base_paths)
    rnd.shuffle(paths)
    if limit:
        paths = paths[:limit]
    n = len(paths)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    splits = {"train": paths[:n_tr], "val": paths[n_tr:n_tr + n_va], "test": paths[n_tr + n_va:]}
    jobs, gi = [], 0
    for s, lst in splits.items():
        os.makedirs(os.path.join(out_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks", s), exist_ok=True)
        for j, p in enumerate(lst):
            donor = lst[(j + 1 + rnd.randrange(len(lst) - 1)) % len(lst)] if len(lst) > 1 else None
            jobs.append((gi, p, donor, out_dir, s, k, seed))
            gi += 1
    with Pool(workers) as pool:
        res = list(tqdm(pool.imap_unordered(_process_base, jobs, chunksize=4), total=len(jobs), desc="synthetic"))
    df = pd.DataFrame([r for rs in res for r in rs]).sort_values("id").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return df


# --------------------------------------------------------------------------- 3. real set
def build_real(findit_root, out_dir):
    recs = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(findit_root, f"{split}.txt"))
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks", split), exist_ok=True)
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"real/{split}"):
            src = os.path.join(findit_root, split, r["image"])
            img = cv2.imread(src)
            if img is None:
                continue
            H, W = img.shape[:2]
            mask = np.zeros((H, W), np.uint8)
            n = 0
            if int(r["forged"]) == 1:
                ann = ast.literal_eval(r["forgery annotations"])
                for reg in ann.get("regions", []):
                    if str(reg["region_attributes"].get("Original area", "no")).lower() == "yes":
                        continue  # copied-from area, not a modification
                    sa = reg["shape_attributes"]
                    x, y, w, h = int(sa["x"]), int(sa["y"]), int(sa["width"]), int(sa["height"])
                    mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 255
                    n += 1
            stem = os.path.splitext(r["image"])[0]
            ext = os.path.splitext(r["image"])[1]
            shutil.copy(src, os.path.join(out_dir, "images", split, stem + ext))
            cv2.imwrite(os.path.join(out_dir, "masks", split, stem + ".png"), mask)
            recs.append(dict(id=stem, split=split, label=int(r["forged"]), n_regions=n,
                             image=f"images/{split}/{stem}{ext}", mask=f"masks/{split}/{stem}.png"))
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return out


# --------------------------------------------------------------------------- 4. preview
def preview(out_dir, n=4, split="train", only_forged=True, seed=0):
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(out_dir, "labels.csv"))
    df = df[df.split == split]
    if only_forged:
        df = df[df.label == 1]
    df = df.sample(min(n, len(df)), random_state=seed)
    fig, axes = plt.subplots(2, len(df), figsize=(3.6 * len(df), 9), squeeze=False)
    for i, (_, r) in enumerate(df.iterrows()):
        im = cv2.cvtColor(cv2.imread(os.path.join(out_dir, r["image"])), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(out_dir, r["mask"]), 0)
        ov = im.copy()
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 0, 0), 2)
        axes[0, i].imshow(im); axes[0, i].set_title(f"{r['id']} label={r['label']}", fontsize=8)
        axes[1, i].imshow(ov); axes[1, i].set_title(str(r.get("ops", "")), fontsize=8)
        for a in axes[:, i]:
            a.axis("off")
    plt.tight_layout()
    plt.show()