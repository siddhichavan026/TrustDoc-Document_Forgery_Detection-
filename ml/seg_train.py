"""TrustDoc - localisation-based forgery detection.

A U-Net predicts, for every pixel, the probability that it was edited.
  * localisation   = the predicted mask
  * detection      = image-level score derived from that mask (mean of the top-k pixel probabilities)

Why: a whole-image classifier sees a tiny edit as <1% of the pixels and does not learn from it.
Pixel-level supervision gives the network a direct signal about WHERE the edit is.

Usage (Colab):
    import seg_train as st
    st.run_seg("resnet34", syn_root, real_root, out_dir, epochs=8)
"""
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

import train_cls as tc

SEG_H, SEG_W = 896, 512  # both divisible by 32; close to the native 1024-px scale so small edits survive
TOPK = 100               # image score = mean of the TOPK highest pixel probabilities


# --------------------------------------------------------------------------- data
def augment_photo(img, rng):
    """Photometric only - geometry must stay aligned with the mask."""
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(1, 5), img.shape), 0, 255).astype(np.uint8)
    return img


def letterbox_mask(m, H=SEG_H, W=SEG_W):
    h, w = m.shape[:2]
    s = min(H / float(h), W / float(w))  # identical rounding to tc.letterbox
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.uint8)
    out[:nh, :nw] = m
    return out


class SegDS(Dataset):
    """All images of a split (authentic ones get an empty mask).
    has_mask = forged AND a usable mask exists (only those count for IoU/Dice)."""

    def __init__(self, root, split, train=False, limit=None, forged_only=False):
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        df = df[df.split == split]
        if forged_only:
            df = df[(df.label == 1) & (df.n_regions > 0)]
        df = df.reset_index(drop=True)
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=0).reset_index(drop=True)
        df["has_mask"] = ((df.label == 1) & (df.n_regions > 0)).astype(int)
        self.root, self.df, self.train = root, df, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = tc.load_img(os.path.join(self.root, r["image"]))
        m = cv2.imread(os.path.join(self.root, r["mask"]), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)  # same scale as the image
        if self.train:
            img = augment_photo(img, np.random.default_rng())
        img = tc.letterbox(img, SEG_H, SEG_W)
        m = (letterbox_mask(m) > 0).astype(np.float32)[None]
        return tc.to_tensor(img), torch.from_numpy(m), int(r["label"]), int(r["has_mask"])


# --------------------------------------------------------------------------- model / metrics
def build_unet(encoder="resnet34", pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def iou_dice(pred, gt):
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    s = float(pred.sum() + gt.sum())
    return (inter / union if union else 1.0), (2 * inter / s if s else 1.0)


def best_threshold(scores, labels):
    """Image-score threshold that maximises F1 (chosen on the validation set only)."""
    cands = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(labels, (scores >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


@torch.no_grad()
def predict_all(model, dl, dev, thr_pix=0.5, k=TOPK):
    model.eval()
    scores, labels, ious, dices = [], [], [], []
    for x, y, lab, hm in tqdm(dl, leave=False, mininterval=5):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            logit = model(x.to(dev, non_blocking=True))
        prob = torch.sigmoid(logit.float())
        scores += torch.topk(prob.flatten(1), k, dim=1).values.mean(1).cpu().tolist()
        labels += lab.tolist()
        p = (prob >= thr_pix).cpu().numpy()[:, 0]
        g = y.numpy()[:, 0] > 0.5
        for pi, gi, h in zip(p, g, hm.tolist()):
            if h:
                i, d = iou_dice(pi, gi)
                ious.append(i)
                dices.append(d)
    return np.array(scores), np.array(labels), np.array(ious), np.array(dices)


def summarize(scores, labels, ious, dices, thr):
    out = tc.metrics(scores, labels, thr)  # image-level: auc / acc / precision / recall / f1 / confusion
    out["thr"] = float(thr)
    out["n_pixel_eval"] = int(len(ious))
    out["iou"] = float(np.mean(ious)) if len(ious) else float("nan")
    out["dice"] = float(np.mean(dices)) if len(dices) else float("nan")
    return out


@torch.no_grad()
def save_examples(model, ds, out_png, dev, n=6):
    """Figure for the report: image / ground truth / predicted heat-map (forged samples with masks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    pool = np.where(ds.df["has_mask"].values == 1)[0]
    if len(pool) == 0:
        return
    idx = np.random.default_rng(0).choice(pool, min(n, len(pool)), replace=False)
    fig, ax = plt.subplots(3, len(idx), figsize=(2.6 * len(idx), 11), squeeze=False)
    for j, i in enumerate(idx):
        x, y, _, _ = ds[int(i)]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            p = torch.sigmoid(model(x[None].to(dev)).float())[0, 0].cpu().numpy()
        img = np.clip(x.numpy().transpose(1, 2, 0) * tc.STD + tc.MEAN, 0, 1)
        ax[0, j].imshow(img)
        ax[1, j].imshow(img); ax[1, j].imshow(y[0].numpy(), alpha=0.5, cmap="Reds")
        ax[2, j].imshow(img); ax[2, j].imshow(p, alpha=0.6, cmap="jet", vmin=0, vmax=1)
        for r, t in enumerate(["image", "ground truth", "prediction"]):
            ax[r, j].axis("off")
            if j == 0:
                ax[r, j].set_title(t, fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main entry
def run_seg(encoder, syn_root, real_root, out_dir, epochs=8, bs=8, lr=3e-4, workers=2, limit=None):
    import segmentation_models_pytorch as smp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = encoder.replace("-", "_")
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    train_ds = SegDS(syn_root, "train", train=True, limit=limit)
    val_ds = SegDS(syn_root, "val", limit=limit)
    tests = {"syn_test": SegDS(syn_root, "test", limit=limit), "real_test": SegDS(real_root, "test", limit=limit)}
    kw = dict(num_workers=workers, pin_memory=(dev == "cuda"), persistent_workers=(workers > 0))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=bs, **kw)
    print(f"[unet-{encoder}] train={len(train_ds)} val={len(val_ds)} "
          f"syn_test={len(tests['syn_test'])} real_test={len(tests['real_test'])} device={dev}")

    model = build_unet(encoder).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    ckpt = os.path.join(out_dir, "weights", f"seg_{tag}.pt")
    best, history = -1.0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for x, y, _, _ in tqdm(train_dl, leave=False, mininterval=5):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                logit = model(x)
            logit = logit.float()
            loss = dice_loss(logit, y) + bce(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            cnt += len(y)
        sched.step()
        s, l, ious, dices = predict_all(model, val_dl, dev)
        m = summarize(s, l, ious, dices, best_threshold(s, l))
        history.append(dict(epoch=ep, train_loss=tot / max(cnt, 1), val_auc=m["auc"], val_f1=m["f1"],
                            val_iou=m["iou"], val_dice=m["dice"], seconds=time.time() - t0))
        flag = ""
        if m["auc"] > best or ep == 1:
            best = m["auc"]
            torch.save({"encoder": encoder, "state_dict": model.state_dict(), "in_hw": (SEG_H, SEG_W),
                        "topk": TOPK}, ckpt)
            flag = "  <- saved"
        print(f"[unet-{encoder}] epoch {ep}/{epochs} loss={history[-1]['train_loss']:.4f} "
              f"val_auc={m['auc']:.3f} val_f1={m['f1']:.3f} val_dice={m['dice']:.3f} "
              f"({history[-1]['seconds']:.0f}s){flag}")

    # ---- final: best checkpoint; threshold chosen on SYNTHETIC VAL only, then applied unchanged to both test sets
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["state_dict"])
    s, l, _, _ = predict_all(model, val_dl, dev)
    thr = best_threshold(s, l)
    final = {"model": f"unet-{encoder}", "best_val_auc": best, "threshold": thr, "history": history}
    for tname, ds in tests.items():
        s, l, ious, dices = predict_all(model, DataLoader(ds, batch_size=bs, **kw), dev)
        final[tname] = summarize(s, l, ious, dices, thr)
        pd.DataFrame({"id": ds.df["id"], "label": l, "score": s}).to_csv(
            os.path.join(out_dir, "results", f"seg_{tag}_{tname}_preds.csv"), index=False)
        print(f"[unet-{encoder}] {tname}: " + ", ".join(f"{k}={v:.3f}" for k, v in final[tname].items()
                                                         if isinstance(v, float)))
        save_examples(model, ds, os.path.join(out_dir, "results", f"seg_{tag}_{tname}_examples.png"), dev)
    with open(os.path.join(out_dir, "results", f"seg_{tag}_metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    return final


# --------------------------------------------------------------------------- fine-tune on real forgeries
def make_finetune_data(syn_root, real_root, n_syn=600, forged_weight=6.0):
    """Real Find-It-Again TRAIN split (forged images up-weighted) + a synthetic subset to avoid forgetting."""
    real_tr = SegDS(real_root, "train", train=True)
    keep = ~((real_tr.df.label == 1) & (real_tr.df.n_regions == 0))  # forged but no usable mask -> skip
    real_tr.df = real_tr.df[keep].reset_index(drop=True)
    syn_tr = SegDS(syn_root, "train", train=True, limit=n_syn)
    weights = [forged_weight if h else 1.0 for h in real_tr.df["has_mask"].tolist()] + [1.0] * len(syn_tr)
    return real_tr, syn_tr, weights


def finetune_real(ckpt_name, syn_root, real_root, out_dir, epochs=6, bs=8, lr=1e-4, workers=2,
                  n_syn=600, forged_weight=6.0):
    """Continue training a synthetic-trained U-Net on real forgeries.
    Model selection + threshold use the real VAL split; the real TEST split is only used at the end.
    (Do not report syn_test for this model: real-train authentic receipts also seeded synthetic images.)"""
    import segmentation_models_pytorch as smp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(os.path.join(out_dir, "weights", ckpt_name), map_location=dev)
    encoder = state["encoder"]
    tag = encoder.replace("-", "_") + "_realft"
    model = build_unet(encoder, pretrained=False).to(dev)
    model.load_state_dict(state["state_dict"])

    real_tr, syn_tr, weights = make_finetune_data(syn_root, real_root, n_syn, forged_weight)
    ds = ConcatDataset([real_tr, syn_tr])
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    kw = dict(num_workers=workers, pin_memory=(dev == "cuda"), persistent_workers=(workers > 0))
    train_dl = DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, **kw)
    val_ds, test_ds = SegDS(real_root, "val"), SegDS(real_root, "test")
    val_dl = DataLoader(val_ds, batch_size=bs, **kw)
    print(f"[finetune {encoder}] real_train={len(real_tr)} (forged with mask={int(real_tr.df.has_mask.sum())}) "
          f"syn={len(syn_tr)} real_val={len(val_ds)} real_test={len(test_ds)}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    ckpt = os.path.join(out_dir, "weights", f"seg_{tag}.pt")
    best, history = -1.0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for x, y, _, _ in tqdm(train_dl, leave=False, mininterval=5):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                logit = model(x)
            logit = logit.float()
            loss = dice_loss(logit, y) + bce(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            cnt += len(y)
        sched.step()
        s, l, ious, dices = predict_all(model, val_dl, dev)
        m = summarize(s, l, ious, dices, best_threshold(s, l))
        history.append(dict(epoch=ep, train_loss=tot / max(cnt, 1), val_auc=m["auc"], val_f1=m["f1"],
                            val_iou=m["iou"], val_dice=m["dice"], seconds=time.time() - t0))
        flag = ""
        if m["auc"] > best or ep == 1:
            best = m["auc"]
            torch.save({"encoder": encoder, "state_dict": model.state_dict(), "in_hw": (SEG_H, SEG_W),
                        "topk": TOPK}, ckpt)
            flag = "  <- saved"
        print(f"[finetune {encoder}] epoch {ep}/{epochs} loss={history[-1]['train_loss']:.4f} "
              f"real_val_auc={m['auc']:.3f} f1={m['f1']:.3f} dice={m['dice']:.3f} ({history[-1]['seconds']:.0f}s){flag}")

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    s, l, _, _ = predict_all(model, val_dl, dev)
    thr = best_threshold(s, l)
    final = {"model": f"unet-{encoder}-realft", "best_real_val_auc": best, "threshold": thr, "history": history}
    s, l, ious, dices = predict_all(model, DataLoader(test_ds, batch_size=bs, **kw), dev)
    final["real_test"] = summarize(s, l, ious, dices, thr)
    pd.DataFrame({"id": test_ds.df["id"], "label": l, "score": s}).to_csv(
        os.path.join(out_dir, "results", f"seg_{tag}_real_test_preds.csv"), index=False)
    print(f"[finetune {encoder}] real_test: " + ", ".join(f"{k}={v:.3f}" for k, v in final["real_test"].items()
                                                         if isinstance(v, float)))
    save_examples(model, test_ds, os.path.join(out_dir, "results", f"seg_{tag}_real_test_examples.png"), dev)
    with open(os.path.join(out_dir, "results", f"seg_{tag}_metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    return final
