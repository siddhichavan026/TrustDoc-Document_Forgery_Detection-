"""TrustDoc - image-level classifiers (Authentic vs Forged).

Models: small_cnn (baseline), resnet50, efficientnet_b0   (PyTorch + timm, ImageNet weights for the last two)

Usage (Colab):
    import train_cls as tc
    tc.run("resnet50", syn_root, real_root, out_dir, epochs=8)
"""
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

cv2.setNumThreads(0)  # avoid OpenCV/DataLoader worker deadlocks

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_SIDE = 1024          # same scale the synthetic images were made at
IN_H, IN_W = 640, 384    # network input (receipts are tall); letterboxed, never stretched


# --------------------------------------------------------------------------- image helpers
def load_img(path, max_side=MAX_SIDE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def letterbox(img, H=IN_H, W=IN_W):
    h, w = img.shape[:2]
    s = min(H / float(h), W / float(w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[:nh, :nw] = img
    return canvas


def augment(img, rng):
    """Capture-style augmentation (no flips: mirrored text is not a realistic document)."""
    if rng.random() < 0.5:  # re-compression
        q = int(rng.integers(50, 96))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.5:  # brightness / contrast
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # sensor noise
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(1, 5), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # slight rotation
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img


def to_tensor(img_bgr):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class DocDS(Dataset):
    """Reads <root>/labels.csv (columns: id, split, label, image)."""

    def __init__(self, root, split, train=False, limit=None):
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        df = df[df.split == split].reset_index(drop=True)
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=0).reset_index(drop=True)
        self.root, self.df, self.train = root, df, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_img(os.path.join(self.root, r["image"]))
        if self.train:
            img = augment(img, np.random.default_rng())
        return to_tensor(letterbox(img)), int(r["label"])


# --------------------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """Custom baseline: 5 conv blocks + global average pooling."""

    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96), blk(96, 192), blk(192, 256))
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


def build_model(name, pretrained=True):
    if name == "small_cnn":
        return SmallCNN()
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


DEFAULT_LR = {"small_cnn": 1e-3, "resnet50": 2e-4, "efficientnet_b0": 3e-4}


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    probs, ys = [], []
    for x, y in tqdm(dl, leave=False, mininterval=5):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            logit = model(x.to(dev, non_blocking=True)).view(-1)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metrics(probs, ys, thr=0.5):
    pred = (probs >= thr).astype(int)
    out = dict(n=int(len(ys)), n_forged=int(ys.sum()),
               acc=float(accuracy_score(ys, pred)),
               precision=float(precision_score(ys, pred, zero_division=0)),
               recall=float(recall_score(ys, pred, zero_division=0)),
               f1=float(f1_score(ys, pred, zero_division=0)),
               confusion=confusion_matrix(ys, pred, labels=[0, 1]).tolist())
    try:
        out["auc"] = float(roc_auc_score(ys, probs))
    except ValueError:
        out["auc"] = float("nan")
    return out


# --------------------------------------------------------------------------- main entry
def run(name, syn_root, real_root, out_dir, epochs=8, bs=16, lr=None, workers=2, limit=None):
    """Train on synthetic train, select best epoch on synthetic val,
    then test on synthetic test AND the real Find-It-Again test split."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    lr = lr or DEFAULT_LR[name]

    train_ds = DocDS(syn_root, "train", train=True, limit=limit)
    val_ds = DocDS(syn_root, "val", limit=limit)
    tests = {"syn_test": DocDS(syn_root, "test", limit=limit), "real_test": DocDS(real_root, "test", limit=limit)}
    kw = dict(num_workers=workers, pin_memory=(dev == "cuda"), persistent_workers=(workers > 0))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=bs, **kw)
    print(f"[{name}] train={len(train_ds)} val={len(val_ds)} device={dev} lr={lr}")

    model = build_model(name).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    crit = nn.BCEWithLogitsLoss()
    ckpt = os.path.join(out_dir, "weights", f"cls_{name}.pt")
    best, history = -1.0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for x, y in tqdm(train_dl, leave=False, mininterval=5):
            x, y = x.to(dev, non_blocking=True), y.float().to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                logit = model(x).view(-1)
            loss = crit(logit.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            cnt += len(y)
        sched.step()
        p, y = predict(model, val_dl, dev)
        m = metrics(p, y)
        history.append(dict(epoch=ep, train_loss=tot / max(cnt, 1), val_auc=m["auc"], val_f1=m["f1"], val_acc=m["acc"],
                            seconds=time.time() - t0))
        flag = ""
        if m["auc"] > best or ep == 1:
            best = m["auc"]
            torch.save({"name": name, "state_dict": model.state_dict(), "in_hw": (IN_H, IN_W)}, ckpt)
            flag = "  <- saved"
        print(f"[{name}] epoch {ep}/{epochs} loss={history[-1]['train_loss']:.4f} val_auc={m['auc']:.4f} "
              f"val_f1={m['f1']:.4f} ({history[-1]['seconds']:.0f}s){flag}")

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    final = {"model": name, "best_val_auc": best, "history": history}
    for tname, ds in tests.items():
        dl = DataLoader(ds, batch_size=bs, **kw)
        p, y = predict(model, dl, dev)
        final[tname] = metrics(p, y)
        pd.DataFrame({"id": ds.df["id"], "label": y, "prob_forged": p}).to_csv(
            os.path.join(out_dir, "results", f"cls_{name}_{tname}_preds.csv"), index=False)
        print(f"[{name}] {tname}: " + ", ".join(f"{k}={v:.3f}" for k, v in final[tname].items()
                                                  if isinstance(v, float)))
    with open(os.path.join(out_dir, "results", f"cls_{name}_metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    return final


def summary(out_dir, names=("small_cnn", "resnet50", "efficientnet_b0")):
    rows = []
    for n in names:
        p = os.path.join(out_dir, "results", f"cls_{n}_metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        row = {"model": n}
        for t in ("syn_test", "real_test"):
            for k in ("auc", "acc", "precision", "recall", "f1"):
                row[f"{t}_{k}"] = round(m[t][k], 3)
        rows.append(row)
    return pd.DataFrame(rows)
