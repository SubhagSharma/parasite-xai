#!/usr/bin/env python
# make_unified_roi_v2.py -- scale-normalised, isotropic, fixed-field-of-view dataset
r"""
WHAT CHANGED FROM v1
--------------------
v1 removed border geometry and colour cast -- both measured at only ~1.7x chance. It
left the two cues that are actually large, and one of them it made worse.

  1. SCALE.  v1 resized each ROI to --size, but ROI side varies 477-2270 px across
     sources, so the same physical egg landed at up to 1.74x different pixel size.
     v2 resamples every image to a common scale FIRST (factors from
     probe_scale_decomposition.py), then crops a FIXED window. Field of view and
     magnification are then constant, and apparent egg size is true egg size.

  2. ANISOTROPY.  pxai/data.py uses transforms.Resize((S,S)), which forces a square by
     STRETCHING. A circular egg comes out elliptical, with eccentricity set by the
     source aspect ratio -- a 4.84x spread across the 13 native sizes. Extracting a
     square makes the final resize isotropic and the distortion disappears.
     (v1 already did this; it was never credited as the main benefit.)

  3. COLOUR TARGET SAMPLING.  v1 took files[:per] after sorted(listdir), i.e. one
     acquisition source per class -- the exact failure that produced the withdrawn
     44.3% frame-geometry claim. v2 samples randomly.

PIPELINE
--------
    resample to common scale (Lanczos, downsample only)
      -> content mask
      -> place a FIXED window that lies inside the content AND contains the egg
      -> Reinhard colour transfer in l-alpha-beta
      -> one uniform resize to --size, identical factor for every image

USAGE
-----
Step 1, choose the window. Costs nothing, writes nothing:

    python -u make_unified_roi_v2.py --dry-run \
        --root ../Data/Chula-ParasiteEgg-11/data \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json \
        --scale-factors runs/scale_factors.json

Read the per-class retention table. A window is only usable if the images it drops
are NOT concentrated in a few classes -- otherwise you have traded one bias for
another. Then:

    python -u make_unified_roi_v2.py \
        --root ../Data/Chula-ParasiteEgg-11/data \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json \
        --scale-factors runs/scale_factors.json \
        --out ../Data/chula_roi2 --window 640 --size 384

Runtime is roughly 30-60 min for 11k images: the resample and the distance transform
both run on full-resolution arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image

# ------------------------------------------------------- Reinhard l-alpha-beta (v1)
_RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_LMS2LAB = np.diag([1 / np.sqrt(3.0), 1 / np.sqrt(6.0), 1 / np.sqrt(2.0)]) @ \
           np.array([[1., 1., 1.], [1., 1., -2.], [1., -1., 0.]])
_LAB2LMS = np.linalg.inv(_LMS2LAB)


def rgb_to_lab(rgb):
    lms = np.log10(np.clip(rgb.reshape(-1, 3) @ _RGB2LMS.T, 1e-6, None))
    return (lms @ _LMS2LAB.T).reshape(rgb.shape)


def lab_to_rgb(lab):
    lms = np.power(10.0, lab.reshape(-1, 3) @ _LAB2LMS.T)
    return np.clip(lms @ _LMS2RGB.T, 0.0, 1.0).reshape(lab.shape)


def lab_stats(rgb, mask=None):
    lab = rgb_to_lab(rgb)
    sel = (lab if mask is None else lab[mask]).reshape(-1, 3)
    return sel.mean(0), sel.std(0) + 1e-6


def reinhard(rgb, tgt_mean, tgt_std, mask=None):
    m, s = lab_stats(rgb, mask)
    return lab_to_rgb((rgb_to_lab(rgb) - m) / s * tgt_std + tgt_mean)


# ------------------------------------------------------------------------ geometry
def content_mask(arr255, thresh=12):
    return arr255.mean(2) > thresh


def half_available(mask):
    """h_available[y,x] = largest half-side of a square centred at (y,x) inside mask.

    Chessboard distance transform. The False pad makes the image edge a boundary --
    without it an all-content mask has no background pixel and the transform is
    undefined, which is exactly the rectangular-frame classes.
    """
    from scipy import ndimage
    D = ndimage.distance_transform_cdt(
        np.pad(mask, 1, constant_values=False), metric="chessboard")[1:-1, 1:-1]
    return D.astype(np.int32) - 1


def half_required(shape, must):
    """h_required[y,x] = smallest half-side of a square at (y,x) that covers `must`."""
    H, W = shape
    mx0, my0, mx1, my1 = must
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    return np.maximum(np.maximum(my1 - yy, yy - my0),
                      np.maximum(mx1 - xx, xx - mx0))


def max_window(h_avail, h_req):
    """Largest window side that is both inside the content and covers the egg."""
    usable = h_avail >= h_req
    if not usable.any():
        return 0
    return 2 * int(np.where(usable, h_avail, -1).max()) + 1


def place_window(h_avail, h_req, side, target):
    """Position a window of exactly `side` px, as close to `target` as feasible."""
    H, W = h_avail.shape
    h = side // 2
    ok = (h_avail >= h) & (h_req <= h)
    if not ok.any():
        return None
    ty, tx = target
    d2 = (np.arange(H)[:, None] - ty) ** 2 + (np.arange(W)[None, :] - tx) ** 2
    cy, cx = np.unravel_index(np.argmin(np.where(ok, d2, np.inf)), d2.shape)
    s = 2 * h + 1
    return int(np.clip(cx - h, 0, W - s)), int(np.clip(cy - h, 0, H - s)), s


# --------------------------------------------------------------------- annotations
def load_index(labels_path):
    with open(labels_path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: os.path.basename(im["file_name"]) for im in coco["images"]}
    boxes = defaultdict(list)
    for a in coco["annotations"]:
        if a["image_id"] in imgs:
            boxes[imgs[a["image_id"]]].append(tuple(a["bbox"]))
    return dict(boxes)


def load_scale_factors(path):
    if path is None:
        return None, None
    with open(path) as f:
        d = json.load(f)
    return {k: v["resample"] for k, v in d["groups"].items()}, d.get("reference_group")


def resample_factor(factors, W, H):
    if factors is None:
        return 1.0, "none"
    key = f"{W}x{H}"
    if key in factors:
        return float(factors[key]), key
    return float(factors.get("other", 1.0)), "other"


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--scale-factors", default=None,
                    help="JSON from probe_scale_decomposition.py --emit. Without it "
                         "no scale normalisation happens and v2 degenerates to v1.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=None,
                    help="window side in COMMON-SCALE pixels. Run --dry-run to pick.")
    ap.add_argument("--size", type=int, default=384, help="final output side")
    ap.add_argument("--black-thresh", type=int, default=12)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--target-sample", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the per-class retention table and write nothing")
    ap.add_argument("--dry-per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ann = load_index(a.labels) if a.labels else {}
    factors, ref = load_scale_factors(a.scale_factors)
    if factors:
        print(f"scale factors loaded, reference group {ref}, "
              f"{len(factors)} groups, range "
              f"{min(factors.values()):.3f}-{max(factors.values()):.3f}")
    else:
        print("** no --scale-factors: NO scale normalisation. This is v1 behaviour.")

    classes = sorted(d for d in os.listdir(a.root)
                     if os.path.isdir(os.path.join(a.root, d)) and not d.startswith("."))
    by_class = {}
    for c in classes:
        d = os.path.join(a.root, c)
        by_class[c] = [(f, os.path.join(d, f)) for f in sorted(os.listdir(d))
                       if not f.startswith(".")
                       and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total = sum(len(v) for v in by_class.values())
    print(f"{total} images across {len(classes)} classes")

    def prepare(path, fname):
        """-> (resampled float array 0..255, must-box in resampled px, factor) or None"""
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            return None
        W, H = im.size
        r, _ = resample_factor(factors, W, H)
        if abs(r - 1.0) > 1e-6:
            im = im.resize((max(8, round(W * r)), max(8, round(H * r))), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
        must = None
        if fname in ann:
            b = ann[fname]
            must = (int(min(x for x, _, _, _ in b) * r),
                    int(min(y for _, y, _, _ in b) * r),
                    int(np.ceil(max(x + w for x, _, w, _ in b) * r)),
                    int(np.ceil(max(y + h for _, y, _, h in b) * r)))
        return arr, must, r

    # ------------------------------------------------------------------- dry run
    if a.dry_run:
        avail = defaultdict(list)
        for c in classes:
            sample = rng.sample(by_class[c], min(a.dry_per_class, len(by_class[c])))
            for fname, path in sample:
                got = prepare(path, fname)
                if got is None:
                    continue
                arr, must, _ = got
                m = content_mask(arr, a.black_thresh)
                if m.sum() < 100:
                    continue
                ha = half_available(m)
                hr = half_required(m.shape, must) if must is not None \
                    else np.zeros_like(ha)
                avail[c].append(max_window(ha, hr))
            print(f"  {c[:26]:<28} n={len(avail[c])} "
                  f"median {int(np.median(avail[c])) if avail[c] else 0}", flush=True)

        cands = [int(np.percentile(np.concatenate([avail[c] for c in classes]), p))
                 for p in (1, 5, 10, 25)]
        cands = sorted(set(cands))
        print(f"\nper-class retention  (% of images that can supply the window)")
        print(f"{'class':<28}" + "".join(f"{w:>9}" for w in cands))
        print("-" * (28 + 9 * len(cands)))
        for c in classes:
            v = np.array(avail[c])
            print(f"  {c[:26]:<26}" +
                  "".join(f"{(v >= w).mean() * 100:>8.1f}%" for w in cands))
        print(f"\n  {'ALL':<26}" + "".join(
            f"{np.mean(np.concatenate([np.array(avail[c]) >= w for c in classes])) * 100:>8.1f}%"
            for w in cands))
        print("\nPick the largest window whose retention row is FLAT across classes.")
        print("A window that drops 40% of one species and 2% of another has replaced")
        print("the shortcut with a sampling bias.")
        return

    # -------------------------------------------------------------------- extract
    if not a.out or not a.window:
        raise SystemExit("--out and --window are required unless --dry-run")

    tgt_mean = tgt_std = None
    if not a.no_color:
        per = max(1, a.target_sample // len(classes))
        means, stds = [], []
        for c in classes:                      # RANDOM, not files[:per]  (PART H)
            for fname, path in rng.sample(by_class[c], min(per, len(by_class[c]))):
                got = prepare(path, fname)
                if got is None:
                    continue
                arr = got[0] / 255.0
                m = content_mask(got[0], a.black_thresh)
                if m.sum() < 100:
                    continue
                mu, sd = lab_stats(arr, m)
                means.append(mu)
                stds.append(sd)
        tgt_mean, tgt_std = np.mean(means, 0), np.mean(stds, 0)
        print(f"colour target from {len(means)} RANDOM images: "
              f"mean {np.round(tgt_mean, 3)} std {np.round(tgt_std, 3)}")

    made = 0
    dropped = defaultdict(int)
    kept = defaultdict(int)
    egg_px = defaultdict(list)
    # Remapped annotations in OUTPUT pixel coordinates. Without this every downstream
    # probe (occlusion, pointing game, bbox geometry) silently reads boxes that belong
    # to a different coordinate system -- the same failure mode as the crop-name
    # ambiguity in pxai/eval/cropgeom.py.
    out_imgs, out_anns, cat_id = [], [], {c: i for i, c in enumerate(classes)}
    for c in classes:
        od = os.path.join(a.out, c)
        os.makedirs(od, exist_ok=True)
        for fname, path in by_class[c]:
            got = prepare(path, fname)
            if got is None:
                dropped[c] += 1
                continue
            arr, must, r = got
            m = content_mask(arr, a.black_thresh)
            if m.sum() < 100:
                dropped[c] += 1
                continue
            ha = half_available(m)
            hr = half_required(m.shape, must) if must is not None else np.zeros_like(ha)
            tgt = ((must[1] + must[3]) / 2, (must[0] + must[2]) / 2) if must is not None \
                else (m.shape[0] / 2, m.shape[1] / 2)
            pos = place_window(ha, hr, a.window, tgt)
            if pos is None:
                dropped[c] += 1
                continue
            x0, y0, s = pos
            crop = arr[y0:y0 + s, x0:x0 + s] / 255.0
            if not a.no_color:
                crop = reinhard(crop, tgt_mean, tgt_std)
            outname = os.path.splitext(fname)[0] + ".jpg"
            Image.fromarray((crop * 255.0).astype(np.uint8)).resize(
                (a.size, a.size), Image.LANCZOS).save(
                os.path.join(od, outname), quality=95)
            kept[c] += 1
            made += 1

            # original px -> resampled px -> window-relative px -> output px
            k = a.size / s
            img_id = len(out_imgs)
            out_imgs.append({"id": img_id, "file_name": outname,
                             "width": a.size, "height": a.size})
            for bx, by, bw, bh in ann.get(fname, []):
                out_anns.append({"image_id": img_id, "category_id": cat_id[c],
                                 "bbox": [(bx * r - x0) * k, (by * r - y0) * k,
                                          bw * r * k, bh * r * k]})
            if must is not None:
                egg_px[c].append(max(must[2] - must[0], must[3] - must[1]) * k)
            if made % 500 == 0:
                print(f"  {made} written...", flush=True)

    lp = os.path.join(a.out, "labels.json")
    with open(lp, "w") as f:
        json.dump({"images": out_imgs, "annotations": out_anns,
                   "categories": [{"id": i, "name": c} for c, i in cat_id.items()]}, f)
    print(f"\nwrote {len(out_anns)} remapped boxes -> {lp}")
    print("  point probe_occlusion.py / probe_bbox_geometry.py at THIS labels.json,")
    print("  never the original -- the coordinate systems differ.")

    print(f"\nwrote {made} images to {a.out}")
    print(f"{'class':<28}{'kept':>7}{'dropped':>9}{'keep %':>9}{'egg px @out':>13}")
    print("-" * 66)
    for c in classes:
        n = kept[c] + dropped[c]
        e = f"{np.median(egg_px[c]):.0f}" if egg_px[c] else "-"
        print(f"  {c[:26]:<26}{kept[c]:>7}{dropped[c]:>9}"
              f"{kept[c] / max(n, 1) * 100:>8.1f}%{e:>13}")
    print("\n'egg px @out' is the median egg long axis in the OUTPUT image. With scale")
    print("normalisation working, this column is now pure morphology -- it should")
    print("track textbook egg dimensions and nothing else.")
    print("Verify with: python -u probe_bbox_geometry.py, and re-run")
    print("diagnose_frame_geometry.py on the output.")


if __name__ == "__main__":
    main()