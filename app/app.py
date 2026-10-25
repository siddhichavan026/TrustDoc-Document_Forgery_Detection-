"""TrustDoc web app.   Run from the project root (D:\\TrustDoc):   python app\\app.py   then open http://127.0.0.1:5000"""
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402
from db import History  # noqa: E402

TAG = os.environ.get("TRUSTDOC_TAG", "seg_resnet34")
DATA = os.environ.get("TRUSTDOC_DATA") or os.path.join(ROOT, "data", "app")
os.makedirs(DATA, exist_ok=True)
ALLOWED = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB
db = History(os.path.join(DATA, "history.db"))
_lock = threading.Lock()  # one inference at a time (CPU model)
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        print("loading model ...", flush=True)
        _analyzer = infer.TrustDocAnalyzer.from_dir(os.path.join(ROOT, "weights"), TAG)
        print(f"model {_analyzer.encoder} ready on {_analyzer.dev}", flush=True)
    return _analyzer


def _urls(rid, folder):
    base = f"/files/{folder}/"
    out = {}
    for k in ("original", "overlay", "heatmap", "gradcam"):
        if os.path.exists(os.path.join(DATA, folder, k + ".jpg")):
            out[k] = base + k + ".jpg"
    return out


def _row_json(r):
    r = dict(r)
    r["images"] = _urls(r["id"], r["folder"])
    return r


@app.route("/")
def index():
    return render_template("index.html", stats=db.stats())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file uploaded."), 400
    name = os.path.basename(f.filename)
    if not name.lower().endswith(ALLOWED):
        return jsonify(error="Please upload a JPG or PNG image."), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 32:
        return jsonify(error="That file could not be read as an image."), 400
    try:
        an = get_analyzer()
        t0 = time.time()
        with _lock:
            res = an.analyze(img)
        sec = time.time() - t0
    except Exception as e:  # never show a stack trace to the user
        app.logger.exception("analysis failed")
        return jsonify(error=f"Analysis failed: {type(e).__name__}"), 500

    folder = time.strftime("%Y%m%d_%H%M%S_") + os.urandom(3).hex()
    os.makedirs(os.path.join(DATA, folder), exist_ok=True)
    for k in ("original", "overlay", "heatmap", "gradcam"):
        if res.get(k) is not None:
            cv2.imwrite(os.path.join(DATA, folder, k + ".jpg"), res[k], [cv2.IMWRITE_JPEG_QUALITY, 90])
    rid = db.add(name, res["label"], res["forged"], round(res["score"], 4), round(res["confidence"], 4),
                 len(res["boxes"]), round(sec, 2), folder)
    out = _row_json(db.get(rid))
    out["boxes"] = [list(b) for b in res["boxes"]]
    out["image_size"] = list(res["image_size"])
    out["threshold"] = round(res["threshold"], 4)
    out["gradcam_error"] = res.get("gradcam_error")
    return jsonify(out)


@app.route("/api/history")
def history():
    return jsonify(items=[_row_json(r) for r in db.list()], stats=db.stats())


@app.route("/api/result/<int:rid>")
def result(rid):
    r = db.get(rid)
    if not r:
        abort(404)
    return jsonify(_row_json(r))


@app.route("/api/history/<int:rid>", methods=["DELETE"])
def delete(rid):
    r = db.get(rid)
    if r:
        d = os.path.join(DATA, r["folder"])
        if os.path.isdir(d):
            for fn in os.listdir(d):
                os.remove(os.path.join(d, fn))
            os.rmdir(d)
        db.delete(rid)
    return jsonify(ok=True)


@app.route("/files/<folder>/<name>")
def files(folder, name):
    if not folder.replace("_", "").isalnum():
        abort(404)
    return send_from_directory(os.path.join(DATA, folder), name)  # send_from_directory blocks path traversal


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File is too large (max 12 MB)."), 413


if __name__ == "__main__":
    get_analyzer()  # load once at start so the first upload is fast
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
