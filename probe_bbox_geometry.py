#!/usr/bin/env python
# probe_bbox_geometry.py -- how much class information is in the annotation box alone?
"""
PURPOSE
-------
Run this BEFORE the B4.1 background-only training. It measures how much of the class
label is recoverable from the EGG ANNOTATION GEOMETRY ALONE -- box size, box position,
image size. No pixel content whatsoever.

WHY IT MUST RUN FIRST
---------------------
B4.1 masks the annotated box and trains on what is left. The mask outline is therefore
a deterministic function of the annotation. If box geometry is class-predictive, a
model trained on egg-masked images can read the HOLE instead of the BACKGROUND, and
the resulting accuracy would be misattributed to background content -- the same class
of error as the withdrawn 44.3% frame-geometry claim (B2.1).

    "everything" accuracy ~= 0.09 (chance)  -> naive per-box mask is safe, run B4.1 as designed
    "everything" accuracy >> 0.09           -> mask must be FIXED SIZE, not per-box

ANSWERS TWO OTHER OPEN ITEMS FOR FREE
-------------------------------------
  * PART G item 4 -- apparent egg scale per class (area-fraction table at the bottom)
  * the H. diminuta resolution-signature hypothesis (B3 residuals) -- the
    "image size only" row is exactly that test

SAMPLING
--------
Uses ALL boxes. No files[:N]. See PART H.

SPLIT
-----
Grouped by source image so multiple boxes from one image never straddle train/test,
and stratified by class. Repeated R times with different seeds; mean +/- sd reported.
This is NOT the canonical seed-1337 split -- it does not need to be, this measures a
property of the annotations, not a model.

USAGE
-----
    python -u probe_bbox_geometry.py --ann /path/to/labels.json
    python -u probe_bbox_geometry.py --ann boxes.csv --format csv

If your annotation format is neither COCO JSON nor a CSV with the expected columns,
edit load_annotations() -- it is the only format-dependent function in the file.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------------- config

COLS = ["w", "h", "area", "logarea", "aspect",
        "cx", "cy",
        "W", "H", "imgar", "logimgpix",
        "wfrac", "hfrac", "areafrac"]

FEATURE_GROUPS = {
    "box size (pixels)":   ["w", "h", "area", "logarea", "aspect"],
    "box position":        ["cx", "cy"],
    "image size only":     ["W", "H", "imgar", "logimgpix"],
    "box size (relative)": ["wfrac", "hfrac", "areafrac"],
    "everything":          COLS,
}


# ------------------------------------------------------------------------ annotations

def load_annotations(path, fmt):
    """Return a list of dicts: cls, x, y, w, h, W, H, key.

    THE ONLY FORMAT-DEPENDENT FUNCTION IN THIS FILE.
    x, y, w, h are in original-image pixels; W, H are the original image dimensions;
    key is a per-image identifier used for the grouped split.
    """
    if fmt == "coco":
        with open(path) as f:
            d = json.load(f)
        imgs = {im["id"]: im for im in d["images"]}
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        rows = []
        for a in d["annotations"]:
            im = imgs[a["image_id"]]
            x, y, w, h = a["bbox"]
            rows.append(dict(
                cls=cats.get(a["category_id"], str(a["category_id"])),
                x=float(x), y=float(y), w=float(w), h=float(h),
                W=float(im["width"]), H=float(im["height"]),
                key=str(im.get("file_name", im["id"])),
            ))
        return rows

    if fmt == "csv":
        alias = {
            "cls": ["cls", "class", "label", "category", "species"],
            "x":   ["x", "xmin", "x1", "bbox_x", "left"],
            "y":   ["y", "ymin", "y1", "bbox_y", "top"],
            "w":   ["w", "width", "bbox_w", "box_w"],
            "h":   ["h", "height", "bbox_h", "box_h"],
            "W":   ["W", "img_w", "image_width", "img_width"],
            "H":   ["H", "img_h", "image_height", "img_height"],
            "key": ["key", "file", "filename", "file_name", "image", "path"],
        }
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            hdr = rd.fieldnames or []
            pick = {}
            for want, cands in alias.items():
                for c in cands:
                    if c in hdr:
                        pick[want] = c
                        break
                else:
                    sys.exit(f"CSV is missing a column for '{want}'. Header was: {hdr}")
            rows = []
            for r in rd:
                rows.append(dict(
                    cls=r[pick["cls"]], key=r[pick["key"]],
                    x=float(r[pick["x"]]), y=float(r[pick["y"]]),
                    w=float(r[pick["w"]]), h=float(r[pick["h"]]),
                    W=float(r[pick["W"]]), H=float(r[pick["H"]]),
                ))
        return rows

    sys.exit(f"unknown --format {fmt}")


# --------------------------------------------------------------------------- features

def build(rows):
    X, y, g, dropped = [], [], [], 0
    for r in rows:
        w, h, W, H = r["w"], r["h"], r["W"], r["H"]
        if min(w, h, W, H) <= 0:
            dropped += 1
            continue
        cx = (r["x"] + w / 2.0) / W
        cy = (r["y"] + h / 2.0) / H
        X.append([w, h, w * h, np.log(w * h), w / h,
                  cx, cy,
                  W, H, W / H, np.log(W * H),
                  w / W, h / H, (w * h) / (W * H)])
        y.append(r["cls"])
        g.append(r["key"])
    if dropped:
        print(f"  dropped {dropped} degenerate boxes")
    return np.asarray(X, float), np.asarray(y), np.asarray(g)


# ---------------------------------------------------------------------- split + model

def grouped_split(g, y, seed, test_frac=0.30):
    """Split on image key, stratified by class. Returns boolean train / test masks."""
    rng = np.random.default_rng(seed)
    key_cls = {}
    for k, c in zip(g, y):
        key_cls.setdefault(k, c)
    test_keys = set()
    for c in np.unique(y):
        ck = np.array([k for k, kc in key_cls.items() if kc == c])
        rng.shuffle(ck)
        n = max(1, int(round(test_frac * len(ck))))
        test_keys.update(ck[:n].tolist())
    te = np.fromiter((k in test_keys for k in g), bool, len(g))
    return ~te, te


def predict(Xtr, ytr, Xte, backend):
    if backend == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9      # nearest centroid, standardised
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    classes = np.unique(ytr)
    C = np.stack([Ztr[ytr == c].mean(0) for c in classes])
    d = ((Zte[:, None, :] - C[None]) ** 2).sum(-1)
    return classes[d.argmin(1)]


# -------------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="annotation file")
    ap.add_argument("--format", default="coco", choices=["coco", "csv"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.30)
    a = ap.parse_args()

    rows = load_annotations(a.ann, a.format)
    X, y, g = build(rows)
    classes = np.unique(y)
    chance = 1.0 / len(classes)
    print(f"\n{len(X)} boxes  |  {len(np.unique(g))} images  |  "
          f"{len(classes)} classes  |  chance = {chance:.4f}")

    try:
        import sklearn  # noqa: F401
        backend = "gbm"
    except ImportError:
        backend = "centroid"
    print(f"classifier: {backend}"
          f"{'  (install scikit-learn for the stronger test)' if backend == 'centroid' else ''}")

    idx = {c: i for i, c in enumerate(COLS)}
    splits = [grouped_split(g, y, s, a.test_frac) for s in range(a.repeats)]

    print(f"\n{'feature set':<22} {'accuracy':>16}   {'x chance':>8}")
    print("-" * 52)
    results = {}
    for name, feats in FEATURE_GROUPS.items():
        cols = [idx[f] for f in feats]
        accs = []
        for tr, te in splits:
            p = predict(X[tr][:, cols], y[tr], X[te][:, cols], backend)
            accs.append((p == y[te]).mean())
        accs = np.asarray(accs)
        results[name] = accs
        print(f"{name:<22} {accs.mean():>8.4f} +/- {accs.std():<5.4f} "
              f"{accs.mean() / chance:>7.2f}x")

    # per-class recall for the full model, first split only
    tr, te = splits[0]
    p = predict(X[tr], y[tr], X[te], backend)
    print("\nper-class recall, 'everything' (split 0)")
    print("-" * 52)
    for c in classes:
        m = y[te] == c
        print(f"  {c:<28} {(p[m] == c).mean():.4f}   n={m.sum()}")

    # PART G item 4 -- apparent egg scale per class
    print("\napparent egg scale per class  (PART G item 4)")
    print(f"{'class':<28} {'area frac %':>18}  {'median box px':>14}  {'img sizes':>9}")
    print("-" * 76)
    af = X[:, idx["areafrac"]] * 100.0
    for c in classes:
        m = y == c
        q1, q2, q3 = np.percentile(af[m], [25, 50, 75])
        bw = np.median(X[m][:, idx["w"]])
        bh = np.median(X[m][:, idx["h"]])
        nsz = len(set(map(tuple, X[m][:, [idx["W"], idx["H"]]].tolist())))
        print(f"  {c:<26} {q2:>7.2f} [{q1:.2f}-{q3:.2f}]  {bw:>6.0f}x{bh:<6.0f}  {nsz:>7}")
    lo, hi = np.percentile(af, [1, 99])
    print(f"\n  1st-99th pct of egg area fraction: {lo:.2f}% - {hi:.2f}%")
    frac = np.percentile(np.maximum(X[:, idx["wfrac"]], X[:, idx["hfrac"]]), 99)
    print(f"  a square of side {frac:.3f} x min(W,H) covers 99% of boxes "
          f"-> use --fixed-frac {frac:.2f} if a fixed-size mask is needed")

    # verdict
    e = results["everything"].mean()
    print("\n" + "=" * 52)
    print(f"VERDICT   geometry alone -> {e:.4f}  ({e / chance:.2f}x chance)")
    if e < 1.5 * chance:
        print("  LEAK NEGLIGIBLE. Run B4.1 with the naive per-box mask.")
    elif e < 3.0 * chance:
        print("  LEAK MODERATE. Use a fixed-size mask; report this number as the")
        print("  floor that background-only accuracy must be compared against.")
    else:
        print("  LEAK SEVERE. A per-box mask would make B4.1 uninterpretable.")
        print("  Fixed-size mask is mandatory; re-run this probe on the fixed mask")
        print("  geometry (position + image size only) to confirm it drops to chance.")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()