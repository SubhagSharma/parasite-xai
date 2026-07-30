#!/usr/bin/env python
# probe_scale_decomposition.py -- is apparent egg size real morphology, or acquisition scale?
"""
WHY THIS EXISTS
---------------
probe_bbox_geometry.py found that box size in pixels alone predicts the species at
0.5428 (5.97x chance), and everything together at 0.8069 (8.88x). Before building the
unified ROI dataset we have to know WHICH of two very different things that is:

  (1) TRUE MORPHOLOGY.  Fasciolopsis eggs really are ~135 um and Opisthorchis ~28 um.
      Size is a textbook diagnostic feature. A model using it is not cheating.
  (2) ACQUISITION SCALE. Every class holds 9-13 distinct image sizes. If um/pixel
      varies by source, apparent size is partly a camera fingerprint, and any resize
      to a uniform output either bakes it in or scrambles it.

This decides the ROI design, so it must run before make_unified_roi:

  mostly (1)  ->  a fixed NATIVE-PIXEL window preserves a constant field of view.
                  Crop, never per-image resize.
  mostly (2)  ->  each source group must be resampled to a common um/pixel FIRST,
                  then the fixed window is cropped from the resampled image.

METHOD
------
Fit  log(max(w,h))  ~  mu + a_group + b_class  over all boxes, where group is the
exact native (W,H). a_group is the acquisition scale factor in log space; b_class is
the species size factor. Variance decomposition says which dominates. exp(a_g) then
gives the resample factor each group needs to reach a common scale.

The last block re-runs the geometry classifier with box size divided by the fitted
group factor. The drop from 0.5428 is the part of the size cue that was acquisition
rather than biology.

USAGE
-----
    cd /workspace/XAI/MTP/Project/parasite_xai
    python -u probe_scale_decomposition.py \
        --ann ../Data/Chula-ParasiteEgg-11/labels.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def load_coco(path):
    with open(path) as f:
        d = json.load(f)
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    rows = []
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        if min(w, h) <= 0:
            continue
        rows.append(dict(cls=cats.get(a["category_id"], str(a["category_id"])),
                         x=float(x), y=float(y), w=float(w), h=float(h),
                         W=float(im["width"]), H=float(im["height"]),
                         key=str(im.get("file_name", im["id"]))))
    return rows


def dummies(labels):
    u = sorted(set(labels))
    idx = {v: i for i, v in enumerate(u)}
    M = np.zeros((len(labels), len(u)))
    M[np.arange(len(labels)), [idx[v] for v in labels]] = 1.0
    return M, u


def r2(L, D):
    """R^2 of least squares fit of L on design D (intercept assumed inside D)."""
    beta, *_ = np.linalg.lstsq(D, L, rcond=None)
    resid = L - D @ beta
    return 1.0 - resid.var() / L.var(), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--min-group", type=int, default=30,
                    help="native sizes with fewer boxes are pooled into 'other'")
    ap.add_argument("--ref", default="min",
                    help="reference group: 'min' = coarsest scale (every other group is "
                         "then DOWNsampled, never upsampled -- recommended), "
                         "'common' = most populous, or an explicit '1280x960'")
    ap.add_argument("--emit", default=None,
                    help="write the per-group resample factors to this JSON, for "
                         "make_unified_roi to consume")
    a = ap.parse_args()

    rows = load_coco(a.ann)
    L = np.log(np.array([max(r["w"], r["h"]) for r in rows]))
    cls = [r["cls"] for r in rows]
    raw_grp = [f"{int(r['W'])}x{int(r['H'])}" for r in rows]

    n_by_grp = defaultdict(int)
    for g in raw_grp:
        n_by_grp[g] += 1
    grp = [g if n_by_grp[g] >= a.min_group else "other" for g in raw_grp]

    classes = sorted(set(cls))
    groups = sorted(set(grp), key=lambda g: -n_by_grp.get(g, 0))
    print(f"\n{len(rows)} boxes | {len(classes)} classes | "
          f"{len(set(raw_grp))} native sizes -> {len(groups)} groups "
          f"(min {a.min_group} boxes)")

    # ---------------------------------------------------------- identifiability check
    # a_group and b_class are separable ONLY if the design is crossed. If a class were
    # shot on one camera alone, "big eggs" and "high magnification" are the same
    # parameter and the fit is meaningless.
    cls_per_grp = defaultdict(set)
    grp_per_cls = defaultdict(set)
    for c, g in zip(cls, grp):
        cls_per_grp[g].add(c)
        grp_per_cls[c].add(g)
    worst_g = min(cls_per_grp.items(), key=lambda t: len(t[1]))
    worst_c = min(grp_per_cls.items(), key=lambda t: len(t[1]))
    print(f"design crossing: every group holds >= {len(worst_g[1])} classes "
          f"(worst: {worst_g[0]}); every class spans >= {len(worst_c[1])} groups "
          f"(worst: {worst_c[0]})")
    if len(worst_g[1]) < 3 or len(worst_c[1]) < 3:
        print("  ** WARNING: design is close to nested. Species size and acquisition")
        print("     scale are then partly the same parameter and the split below is")
        print("     not trustworthy. Pool groups harder with --min-group.")

    # ---------------------------------------------------------------- variance split
    Dc, uc = dummies(cls)
    Dg, ug = dummies(grp)
    one = np.ones((len(L), 1))
    r2c, _ = r2(L, np.hstack([one, Dc[:, 1:]]))
    r2g, _ = r2(L, np.hstack([one, Dg[:, 1:]]))
    r2b, beta = r2(L, np.hstack([one, Dc[:, 1:], Dg[:, 1:]]))

    print("\nvariance of log(apparent egg size) explained")
    print("-" * 58)
    print(f"  species only                       R^2 = {r2c:.4f}")
    print(f"  native image size only             R^2 = {r2g:.4f}")
    print(f"  both                               R^2 = {r2b:.4f}")
    print(f"  unique to acquisition (both-species) = {r2b - r2c:.4f}")
    print(f"  unique to species (both-size)        = {r2b - r2g:.4f}")

    # group effects -> resample factors
    ag = np.zeros(len(ug))
    ag[1:] = beta[1 + (len(uc) - 1):]
    ag -= ag.mean()
    if a.ref == "min":
        ref = ug[int(np.argmin(ag))]          # coarsest scale -> only downsampling
    elif a.ref == "common":
        ref = ug[int(np.argmax([n_by_grp.get(g, 0) for g in ug]))]
    else:
        if a.ref not in ug:
            raise SystemExit(f"--ref {a.ref} is not one of {ug}")
        ref = a.ref
    aref = ag[ug.index(ref)]

    print(f"\nacquisition scale by native image size   (reference = {ref})")
    print(f"{'native size':<14} {'boxes':>7} {'rel scale':>10} {'resample':>9}")
    print("-" * 58)
    for g, v in sorted(zip(ug, ag), key=lambda t: -n_by_grp.get(t[0], 0)):
        rel = float(np.exp(v - aref))
        print(f"  {g:<12} {n_by_grp.get(g, len(L) - sum(n_by_grp[x] for x in ug if x != 'other')):>7} "
              f"{rel:>9.3f}x {1.0 / rel:>8.3f}x")
    spread = float(np.exp(ag.max() - ag.min()))
    print(f"\n  scale spread across sources: {spread:.2f}x")
    up = [g for g, v in zip(ug, ag) if float(np.exp(v - aref)) < 0.999]
    if up:
        print(f"  ** {len(up)} group(s) would be UPSAMPLED: {up}")
        print("     upsampling invents no detail and leaves those images blurrier than")
        print("     the rest -- a new source fingerprint. Prefer --ref min.")

    if a.emit:
        factors = {g: {"resample": float(np.exp(aref - v)),
                       "boxes": int(n_by_grp.get(g, 0))}
                   for g, v in zip(ug, ag)}
        with open(a.emit, "w") as f:
            json.dump({"reference_group": ref,
                       "note": "multiply image dimensions by 'resample' to reach "
                               "the reference scale; key 'other' covers all native "
                               f"sizes with fewer than {a.min_group} boxes",
                       "groups": factors}, f, indent=2)
        print(f"\n  wrote resample factors -> {a.emit}")

    # species effects -> implied relative physical size
    bc = np.zeros(len(uc))
    bc[1:] = beta[1:len(uc)]
    bc -= bc.min()
    print("\nimplied relative egg size, acquisition removed  (smallest = 1.00)")
    print("-" * 58)
    order = np.argsort(bc)
    for i in order:
        print(f"  {uc[i]:<30} {np.exp(bc[i]):>6.2f}x")
    print("  (sanity check: this ordering should match textbook egg dimensions)")

    # ------------------------------------------------- residual leak after correction
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        print("\nscikit-learn missing; skipping the residual-leak block")
        return

    gi = {g: i for i, g in enumerate(ug)}
    k = np.exp(np.array([ag[gi[g]] - aref for g in grp]))          # per-box scale factor
    W = np.array([r["W"] for r in rows]); H = np.array([r["H"] for r in rows])
    w = np.array([r["w"] for r in rows]); h = np.array([r["h"] for r in rows])
    cx = np.array([(r["x"] + r["w"] / 2) / r["W"] for r in rows])
    cy = np.array([(r["y"] + r["h"] / 2) / r["H"] for r in rows])
    keys = np.array([r["key"] for r in rows])
    y = np.array(cls)

    raw = np.column_stack([w, h, w * h, np.log(w * h), w / h])
    cor = np.column_stack([w / k, h / k, (w * h) / k**2, np.log(w * h / k**2), w / h])
    full_raw = np.column_stack([raw, cx, cy, W, H])
    full_cor = np.column_stack([cor, cx, cy, np.ones_like(W), np.ones_like(H)])

    rng = np.random.default_rng(0)
    kk = np.unique(keys)
    te_keys = set(rng.permutation(kk)[: int(0.3 * len(kk))].tolist())
    te = np.fromiter((s in te_keys for s in keys), bool, len(keys))

    print("\nresidual class information after scale correction")
    print(f"{'feature set':<34} {'accuracy':>9} {'x chance':>9}")
    print("-" * 58)
    for name, X in [("box size, raw pixels", raw),
                    ("box size, scale-corrected", cor),
                    ("+ position + image size, raw", full_raw),
                    ("+ position, scale-corrected", full_cor)]:
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        m.fit(X[~te], y[~te])
        acc = (m.predict(X[te]) == y[te]).mean()
        print(f"  {name:<32} {acc:>8.4f} {acc * len(classes):>8.2f}x")

    # ------------------------------------------------------------------- ROI recipe
    print("\nROI recipe implied by the above")
    print("-" * 58)
    Wr = np.array([r["W"] for r in rows]); Hr = np.array([r["H"] for r in rows])
    short_ref = np.minimum(Wr, Hr) / k                # short side at common scale
    for q in (1, 5, 10, 25, 50):
        print(f"  {q:>3}th pct of short side at common scale: {np.percentile(short_ref, q):>8.0f} px")
    print(f"\n  a window of {np.percentile(short_ref, 5):.0f} px at the {ref} scale fits "
          f"95% of images")
    print(f"  99th pct egg long axis at common scale: "
          f"{np.percentile(np.maximum(w, h) / k, 99):.0f} px")
    print()


if __name__ == "__main__":
    main()
