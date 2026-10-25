"""Automatic checks for the web app. From the project root (D:\\TrustDoc):   python tests\\test_app.py
Uses the real model and the sample receipts in results\\demo_samples. Prints PASS/FAIL per check."""
import io
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import tempfile  # tests use a throw-away data folder, so your real history is untouched
os.environ["TRUSTDOC_DATA"] = tempfile.mkdtemp(prefix="trustdoc_test_")
import app as webapp  # noqa: E402

SAMPLES = os.path.join(ROOT, "results", "demo_samples")
client = webapp.app.test_client()
results = []


def check(name, ok, info=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{info}]" if info else ""))


def upload(name, data):
    return client.post("/api/analyze", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


# ---- 1. pages load
check("home page loads", client.get("/").status_code == 200)
check("css and js load", client.get("/static/css/style.css").status_code == 200
      and client.get("/static/js/app.js").status_code == 200)

# ---- 2. bad input is refused politely (no crash)
r = client.post("/api/analyze", data={}, content_type="multipart/form-data")
check("no file -> 400", r.status_code == 400, r.json.get("error", ""))
r = upload("notes.txt", b"hello")
check("wrong extension -> 400", r.status_code == 400, r.json.get("error", ""))
r = upload("fake.png", b"this is not an image")
check("corrupt image -> 400", r.status_code == 400, r.json.get("error", ""))
r = upload("tiny.png", png_bytes(np.full((10, 10, 3), 255, np.uint8)))
check("tiny image -> 400", r.status_code == 400)
r = upload("big.png", b"0" * (13 * 1024 * 1024))
check("file over 12 MB -> 413", r.status_code == 413, r.json.get("error", "") if r.is_json else "")

# ---- 3. unusual but valid images still work
blank = upload("blank_white.png", png_bytes(np.full((600, 400, 3), 255, np.uint8)))
check("blank white page analyzed", blank.status_code == 200 and "label" in blank.json,
      blank.json.get("label", ""))
wide = upload("wide.png", png_bytes(np.random.randint(0, 255, (300, 1800, 3), np.uint8)))
check("very wide noisy image analyzed", wide.status_code == 200)
gray = cv2.cvtColor(np.full((500, 300, 3), 200, np.uint8), cv2.COLOR_BGR2GRAY)
check("grayscale file analyzed", upload("gray.png", png_bytes(gray)).status_code == 200)

# ---- 4. sample receipts: synthetic forgeries should be flagged, authentic ones should not (expected, not guaranteed)
if os.path.isdir(SAMPLES):
    flagged = {}
    for f in sorted(os.listdir(SAMPLES)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        d = open(os.path.join(SAMPLES, f), "rb").read()
        r = upload(f, d)
        ok = r.status_code == 200
        check(f"sample {f} analyzed", ok, f"{r.json.get('label')} {r.json.get('confidence')}" if ok else str(r.status_code))
        if ok:
            flagged[f] = r.json["forged"]
            imgs = r.json["images"]
            check(f"  result images exist for {f}", all(client.get(u).status_code == 200 for u in imgs.values()))
            if r.json["forged"]:
                check(f"  flagged {f} has >=1 region", r.json["regions"] >= 1, f"regions={r.json['regions']}")
    print("\nInformation only (the model is not perfect, so these are not pass/fail):")
    for f, v in flagged.items():
        print(f"   {f:45s} -> {'Forged' if v else 'Authentic'}")

# ---- 5. history, reopen, delete
h = client.get("/api/history").json
check("history lists analyses", len(h["items"]) >= 1, f"{h['stats']}")
rid = h["items"][0]["id"]
check("single result can be reopened", client.get(f"/api/result/{rid}").status_code == 200)
check("unknown id -> 404", client.get("/api/result/999999").status_code == 404)
before = h["stats"]["total"]
client.delete(f"/api/history/{rid}")
after = client.get("/api/history").json["stats"]["total"]
check("delete removes one row", after == before - 1, f"{before} -> {after}")

# ---- 6. security: cannot read other files through /files
check("path traversal blocked", client.get("/files/../history.db").status_code in (404, 308, 400))
check("path traversal blocked (encoded)", client.get("/files/%2e%2e/history.db").status_code in (404, 308, 400))

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
