r"""
diagnose_frame_geometry.py — is the confound visible in the RAW images, per class?

Measures, per class, the acquisition artifacts a classifier could read instead of the
parasite: black-border fraction, fill ratio (~1.00 = square frame, ~0.785 = circular
field of view), and colour cast. Samples RANDOMLY -- taking the first N files sorted
by name samples one acquisition source when a class contains several.
"""
from __future__ import annotations
import argparse, os
from collections import defaultdict
import numpy as np
from PIL import Image


def analyse(path, black_thresh=12):
    a = np.asarray(Image.open(path).convert("RGB"))
    H, W = a.shape[:2]
    mask = a.mean(2) > black_thresh
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    bbox = (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())
    content = int(mask.sum())
    mr, mg, mb = (a[..., c][mask].mean() for c in range(3))
    return {"black_frac": 1.0 - content / (H * W),
            "fill_ratio": content / max(bbox, 1),
            "content_frac": content / (H * W),
            "mean_r": mr, "mean_g": mg, "mean_b": mb,
            "w": W, "h": H, "cast": float(mb - mr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--black-thresh", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)) and not d.startswith("."))
    stats = defaultdict(list)
    for c in classes:
        d = os.path.join(args.root, c)
        files = sorted(f for f in os.listdir(d)
                       if not f.startswith(".") and f.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.per_class and len(files) > args.per_class:
            rng = np.random.RandomState(args.seed)
            files = [files[i] for i in rng.choice(len(files), args.per_class, replace=False)]
        for f in files:
            r = analyse(os.path.join(d, f), args.black_thresh)
            if r:
                stats[c].append(r)
        print(f"  {c[:26]:>26}  {len(stats[c])} images", flush=True)

    keys = ["black_frac", "fill_ratio", "content_frac", "cast", "mean_r", "mean_g", "mean_b"]
    print(f"\n=== per-class frame geometry and colour ===")
    print(f"{'class':>26} {'black%':>7} {'+-sd':>6} {'fill':>6} {'cast':>7} "
          f"{'meanR':>6} {'meanB':>6} {'size (n distinct)':>22}")
    table = {}
    for c in classes:
        v = stats[c]
        if not v:
            continue
        m = {k: float(np.mean([x[k] for x in v])) for k in keys}
        table[c] = m
        sd = float(np.std([x["black_frac"] for x in v])) * 100
        sizes = {(x["w"], x["h"]) for x in v}
        sz = f"{int(np.median([x['w'] for x in v]))}x{int(np.median([x['h'] for x in v]))}"
        if len(sizes) > 1:
            sz += f"  ({len(sizes)} sizes)"
        print(f"{c[:26]:>26} {m['black_frac']*100:>7.1f} {sd:>6.1f} {m['fill_ratio']:>6.3f} "
              f"{m['cast']:>7.1f} {m['mean_r']:>6.1f} {m['mean_b']:>6.1f} {sz:>22}")

    print(f"\n=== spread ACROSS classes ===")
    for k in keys:
        vals = np.array([table[c][k] for c in table])
        sc = 100 if k.endswith("frac") else 1
        lo = classes[int(np.argmin(vals))][:20]; hi = classes[int(np.argmax(vals))][:20]
        print(f"  {k:>13}: {vals.min()*sc:>8.2f} ({lo})  ->  {vals.max()*sc:>8.2f} ({hi})"
              f"   range {(vals.max()-vals.min())*sc:.2f}")

    X, y = [], []
    for ci, c in enumerate(classes):
        for x in stats[c]:
            X.append([x[k] for k in keys]); y.append(ci)
    X = np.array(X); y = np.array(y)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cent = np.stack([Xs[y == i].mean(0) for i in range(len(classes))])
    acc = float((np.argmin(((Xs[:, None] - cent[None]) ** 2).sum(-1), 1) == y).mean())
    print(f"\n  nearest-centroid on these {len(keys)} stats ONLY: {acc:.4f}  "
          f"(chance {1/len(classes):.4f})")
    print("  NOTE: features are standardised, so this amplifies even tiny residual")
    print("  differences. The RANGES above are the meaningful measure.")


if __name__ == "__main__":
    main()
