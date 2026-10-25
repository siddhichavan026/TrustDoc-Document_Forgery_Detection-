"""TrustDoc inference: one call -> verdict, confidence, edit locations, overlay and Grad-CAM.

Colab:
    import infer
    (run  python ml\\make_calibration.py  once to create weights/seg_resnet34_calibration.json)
    an = infer.TrustDocAnalyzer.from_dir(f"{BASE}/weights", "seg_resnet34")
    res = an.analyze(cv2.imread("receipt.png"))

Web app:  an = TrustDocAnalyzer.from_dir("weights", "seg_resnet34");  json_dict = to_json(an.analyze(img))
Needs: torch, segmentation-models-pytorch, grad-cam, opencv, numpy   (no sklearn / pandas / tqdm)
"""
import base64
import json
import os

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_SIDE = 1024  # same scale the training images were prepared at


# --------------------------------------------------------------------------- pre-processing (identical to training)
def _prep(img_bgr, in_hw):
    h, w = img_bgr.shape[:2]
    s = MAX_SIDE / float(max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    disp = img_bgr
    H, W = in_hw
    h, w = disp.shape[:2]
    s = min(H / float(h), W / float(w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[:nh, :nw] = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return disp, canvas, np.ascontiguousarray(x.transpose(2, 0, 1)), (nh, nw)


# --------------------------------------------------------------------------- drawing helpers
def _boxes(mask, max_boxes=20, min_area=8):
    """Bounding boxes (x, y, w, h) of suspicious regions; nearby fragments are merged."""
    if mask.max() == 0:
        return []
    merged = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in stats[i]]
        if int((mask[y:y + h, x:x + w] > 0).sum()) >= min_area:
            out.append((x, y, w, h))
    out.sort(key=lambda b: -b[2] * b[3])
    return out[:max_boxes]


def _overlay(disp, mask, boxes):
    out = disp.copy()
    red = np.zeros_like(out)
    red[..., 2] = 255
    m = (mask > 0)[..., None]
    out = np.where(m, (0.6 * out + 0.4 * red).astype(np.uint8), out)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
    return out


def _heat(disp, ph):
    heat = cv2.applyColorMap(np.clip(ph * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = np.clip(ph * 1.5, 0, 0.65)[..., None]
    return (disp * (1 - a) + heat * a).astype(np.uint8)


# --------------------------------------------------------------------------- analyzer
class TrustDocAnalyzer:
    def __init__(self, ckpt_path, threshold=0.5, slope=20.0, device=None, bin_edges=None, bin_p_forged=None,
                 bins=None):
        import segmentation_models_pytorch as smp
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        st = torch.load(ckpt_path, map_location=self.dev)
        self.encoder = st["encoder"]
        self.in_hw = tuple(st.get("in_hw", (896, 512)))
        self.topk = int(st.get("topk", 100))
        self.model = smp.Unet(self.encoder, encoder_weights=None, in_channels=3, classes=1).to(self.dev)
        self.model.load_state_dict(st["state_dict"])
        self.model.eval()
        self.threshold, self.slope = float(threshold), float(slope)
        self.bins = bins  # list of {lo, hi, p_forged} from make_calibration.py (preferred)
        self.bin_edges, self.bin_p = bin_edges, bin_p_forged

    @classmethod
    def from_dir(cls, weights_dir, tag="seg_resnet34", **kw):
        cal = json.load(open(os.path.join(weights_dir, f"{tag}_calibration.json")))
        return cls(os.path.join(weights_dir, f"{tag}.pt"), threshold=cal["threshold"], slope=cal["slope"],
                   bin_edges=cal.get("bin_edges"), bin_p_forged=cal.get("bin_p_forged"),
                   bins=cal.get("bins"), **kw)

    def _p_forged(self, score):
        """Measured fraction of forged images in this score range (held-out data); falls back to a slope curve."""
        if self.bins:
            for b in self.bins:
                if b["lo"] <= score < b["hi"]:
                    return float(b["p_forged"])
        return float(1.0 / (1.0 + np.exp(-self.slope * (score - self.threshold))))

    def _gradcam(self, canvas_bgr, x_tensor, prob, nh, nw, disp_hw):
        """Grad-CAM on the last encoder block, explaining the predicted forgery-probability map."""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
        cam = GradCAM(model=self.model, target_layers=[self.model.encoder.layer4[-1]])
        try:
            gc = cam(input_tensor=x_tensor, targets=[SemanticSegmentationTarget(0, prob.astype(np.float32))])[0]
        finally:
            cam.activations_and_grads.release()
        rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        vis = cv2.cvtColor(show_cam_on_image(rgb, gc, use_rgb=True), cv2.COLOR_RGB2BGR)[:nh, :nw]
        return cv2.resize(vis, (disp_hw[1], disp_hw[0]), interpolation=cv2.INTER_AREA)

    def analyze(self, img_bgr, with_gradcam=True):
        disp, canvas, x, (nh, nw) = _prep(img_bgr, self.in_hw)
        xt = torch.from_numpy(x)[None].to(self.dev)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(xt).float())[0, 0].cpu().numpy()
        score = float(np.partition(prob.ravel(), -self.topk)[-self.topk:].mean())
        forged = score >= self.threshold
        if self.bin_edges:  # measured reliability of this score range (preferred)
            b = int(np.clip(np.searchsorted(self.bin_edges, score, side="right") - 1, 0, len(self.bin_p) - 1))
            p_forged = float(self.bin_p[b])
        else:
            p_forged = self._p_forged(score)
        confidence = max(0.5, p_forged if forged else 1.0 - p_forged)

        ph = cv2.resize(prob[:nh, :nw], (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_LINEAR)
        if forged:  # display mask: 0.5 if the model is that sure, otherwise relative to its strongest response
            pmax = float(ph.max())
            mask = (ph >= (0.5 if pmax >= 0.5 else max(0.6 * pmax, 1e-3))).astype(np.uint8) * 255
        else:
            mask = np.zeros(ph.shape, np.uint8)
        boxes = _boxes(mask)

        gradcam, gc_error = None, None
        if with_gradcam:
            try:
                gradcam = self._gradcam(canvas, xt, prob, nh, nw, disp.shape[:2])
            except Exception as e:  # explanation is optional - never block the verdict
                gc_error = f"{type(e).__name__}: {e}"
        return dict(label="Forged" if forged else "Authentic", forged=bool(forged), score=score,
                    confidence=confidence, p_forged=p_forged, threshold=self.threshold, boxes=boxes,
                    image_size=(int(disp.shape[1]), int(disp.shape[0])),
                    original=disp, overlay=_overlay(disp, mask, boxes), heatmap=_heat(disp, ph),
                    gradcam=gradcam, gradcam_error=gc_error, mask=mask)


# --------------------------------------------------------------------------- JSON for the web app
def _b64(img, ext=".jpg"):
    if img is None:
        return None
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 88] if ext == ".jpg" else [])
    return "data:image/%s;base64,%s" % ("jpeg" if ext == ".jpg" else "png", base64.b64encode(buf).decode())


def to_json(res):
    return dict(label=res["label"], forged=res["forged"], score=round(res["score"], 4),
                confidence=round(res["confidence"], 4), p_forged=round(res["p_forged"], 4), threshold=round(res["threshold"], 4),
                boxes=[list(b) for b in res["boxes"]], image_size=list(res["image_size"]),
                images=dict(original=_b64(res["original"]), overlay=_b64(res["overlay"]),
                            heatmap=_b64(res["heatmap"]), gradcam=_b64(res["gradcam"]),
                            mask=_b64(res["mask"], ".png")),
                gradcam_error=res.get("gradcam_error"))
