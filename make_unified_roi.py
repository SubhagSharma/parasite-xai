r"""
make_unified_roi.py — build a format-uniform, colour-normalised copy of the dataset.

THE PROBLEM. Frame geometry differs by class: some species are square frames with no
border, others are a circular field of view inside a black square. A classifier can
read the border and the colour cast instead of the parasite -- measured: 0.5700
accuracy on whole images with the egg blacked out, and 89% of ProtoPNet's prototypes
sitting on background, many in the letterbox corners.

WHY NOT JUST CROP TO THE EGG. A 20% margin crop (make_crops.py) does remove the
border, but it also changes the task: "name this tightly cropped egg" rather than
"name the parasite in this field of view". This script instead keeps a realistic
field of view and removes the ARTIFACT:

  1. detect the actual content region (exclude black border / vignette)
  2. extract the LARGEST axis-aligned square fully inside that region -- the maximum
     usable image, not a tight crop
  3. if an annotation is available, constrain that square to CONTAIN the egg, so no
     parasite is lost
  4. resize every ROI to one uniform size
  5. colour-normalise with Reinhard transfer in l-alpha-beta space (Reinhard et al.
     2001). NOT grayscale: grayscale discards shell tint and stain information that
     may be diagnostic. Reinhard equalises cast while keeping colour structure.

Output is a plain ImageFolder tree, so nothing downstream changes:

    chula_roi/<class_name>/<original_stem>.jpg

    python make_unified_roi.py \
        --root   ../Data/Chula-ParasiteEgg-11/data \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json \
        --out    ../Data/chula_roi --size 384
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- Reinhard l-a-b
_RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_D = np.diag([1 / np.sqrt(3.0), 1 / np.sqrt(6.0), 1 / np.sqrt(2.0)])
_M = np.array([[1., 1., 1.], [1., 1., -2.], [1., -1., 0.]])
_LMS2LAB = _D @ _M
_LAB2LMS = np.linalg.inv(_LMS2LAB)


def rgb_to_lab(rgb):
    """rgb float in [0,1] -> Reinhard l-alpha-beta."""
    lms = rgb.reshape(-1, 3) @ _RGB2LMS.T
    lms = np.log10(np.clip(lms, 1e-6, None))
    return (lms @ _LMS2LAB.T).reshape(rgb.shape)


def lab_to_rgb(lab):
    lms = lab.reshape(-1, 3) @ _LAB2LMS.T
    lms = np.power(10.0, lms)
    rgb = lms @ _LMS2RGB.T
    return np.clip(rgb, 0.0, 1.0).reshape(lab.shape)


def lab_stats(rgb, mask=None):
    lab = rgb_to_lab(rgb)
    sel = lab if mask is None else lab[mask]
    sel = sel.reshape(-1, 3)
    return sel.mean(0), sel.std(0) + 1e-6


def reinhard(rgb, tgt_mean, tgt_std, mask=None):
    lab = rgb_to_lab(rgb)
    m, s = lab_stats(rgb, mask)
    lab = (lab - m) / s * tgt_std + tgt_mean
    return lab_to_rgb(lab)


# ---------------------------------------------------------------- geometry
def content_mask(arr, thresh=12):
    return arr.mean(2) > thresh


def largest_square(mask, must=None, max_side=None):
    """Largest axis-aligned square fully inside `mask`, computed EXACTLY.

    Uses a chessboard (L-infinity) distance transform: D[y,x] is the distance from
    (y,x) to the nearest non-content pixel, so the largest square centred there with
    half-side h fits iff h <= D[y,x]-1. Taking the maximum over all centres gives the
    true optimum -- a grid scan over candidate corners misses it whenever the optimal
    position falls between scan steps.

    must: (x0, y0, x1, y1) that the square is required to contain (the egg box). For a
    centre (cy,cx) the half-side needed to cover the box is
        h_req = max(y1-cy, cy-y0, x1-cx, cx-x0)
    so the feasible centres are those with D-1 >= h_req, and among them we take the
    largest available square. This guarantees the parasite is never cropped away.

    Returns (x0, y0, side) or None.
    """
    from scipy import ndimage
    H, W = mask.shape
    # Pad with a False ring so the IMAGE EDGE counts as a boundary. Without this an
    # all-content mask (a square frame with no vignette) has no background pixel and
    # the distance transform is undefined -- exactly the rectangular-frame classes.
    D = ndimage.distance_transform_cdt(
        np.pad(mask, 1, constant_values=False), metric="chessboard")[1:-1, 1:-1]
    D = D.astype(np.int32)
    h_avail = D - 1                                  # max half-side at each centre
    if max_side is not None:
        h_avail = np.minimum(h_avail, (max_side - 1) // 2)
    if h_avail.max() < 4:
        return None

    if must is None:
        cy, cx = np.unravel_index(np.argmax(h_avail), h_avail.shape)
        h = int(h_avail[cy, cx])
    else:
        mx0, my0, mx1, my1 = must
        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        # (H,1) and (1,W) terms broadcast to (H,W) only via pairwise maximum
        h_req = np.maximum(np.maximum(my1 - yy, yy - my0),
                           np.maximum(mx1 - xx, xx - mx0))
        feasible = h_avail >= np.maximum(h_req, 0)
        if not feasible.any():
            return None
        scored = np.where(feasible, h_avail, -1)
        cy, cx = np.unravel_index(np.argmax(scored), scored.shape)
        h = int(scored[cy, cx])
        if h < 4:
            return None

    side = 2 * h + 1
    x0 = int(np.clip(cx - h, 0, W - side))
    y0 = int(np.clip(cy - h, 0, H - side))
    return x0, y0, side


# ---------------------------------------------------------------- annotations
def load_index(labels_path):
    with open(labels_path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: os.path.basename(im["file_name"]) for im in coco["images"]}
    boxes, cls = defaultdict(list), {}
    for a in coco["annotations"]:
        if a["image_id"] not in imgs:
            continue
        n = imgs[a["image_id"]]
        boxes[n].append(tuple(a["bbox"]))
        cls[n] = cats.get(a["category_id"], str(a["category_id"]))
    return {n: (cls[n], boxes[n]) for n in boxes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default=None,
                    help="optional; if given, ROIs are constrained to contain the egg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--black-thresh", type=int, default=12)
    ap.add_argument("--min-roi", type=int, default=256,
                    help="reject ROIs whose side is below this many ORIGINAL pixels. "
                         "An 11px square blown up to 384 is noise, not an image.")
    ap.add_argument("--require-egg", action="store_true",
                    help="skip images where no square can cover the annotation, "
                         "instead of silently falling back and possibly losing the egg")
    ap.add_argument("--no-color", action="store_true", help="skip Reinhard normalisation")
    ap.add_argument("--target-sample", type=int, default=400,
                    help="images sampled to compute the colour target")
    args = ap.parse_args()

    ann = load_index(args.labels) if args.labels else {}
    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d))
                     and not d.startswith("."))
    files = []
    for c in classes:
        d = os.path.join(args.root, c)
        for f in sorted(os.listdir(d)):
            if f.startswith("."):
                continue
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append((c, f, os.path.join(d, f)))
    print(f"{len(files)} images across {len(classes)} classes")

    # ---- pass 1: colour target from the CONTENT of a stratified sample ----
    tgt_mean = tgt_std = None
    if not args.no_color:
        per = max(1, args.target_sample // max(1, len(classes)))
        means, stds = [], []
        for c in classes:
            sub = [f for f in files if f[0] == c][:per]
            for _, _, p in sub:
                a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                m = content_mask(a * 255.0, args.black_thresh)
                if m.sum() < 100:
                    continue
                mu, sd = lab_stats(a, m)
                means.append(mu)
                stds.append(sd)
        tgt_mean = np.mean(means, 0)
        tgt_std = np.mean(stds, 0)
        print(f"colour target from {len(means)} images (content only): "
              f"mean {np.round(tgt_mean,3)}  std {np.round(tgt_std,3)}")

    # ---- pass 2: extract, normalise, save ----
    made = skipped = no_egg = egg_kept = too_small = 0
    sides, black_before = [], defaultdict(list)
    per_class = defaultdict(int)
    for i, (c, fname, path) in enumerate(files):
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        a255 = np.asarray(im, dtype=np.float32)
        mask = content_mask(a255, args.black_thresh)
        if mask.sum() < 100:
            skipped += 1
            continue
        black_before[c].append(1.0 - mask.mean())

        must = None
        if fname in ann:
            bxs = ann[fname][1]
            xs0 = min(b[0] for b in bxs); ys0 = min(b[1] for b in bxs)
            xs1 = max(b[0] + b[2] for b in bxs); ys1 = max(b[1] + b[3] for b in bxs)
            must = (int(xs0), int(ys0), int(np.ceil(xs1)), int(np.ceil(ys1)))

        sq = largest_square(mask, must)
        if sq is None and must is not None:
            if args.require_egg:
                no_egg += 1
                skipped += 1
                continue
            sq = largest_square(mask, None)      # fall back; the egg may be lost
            no_egg += 1
        if sq is None:
            skipped += 1
            continue
        x0, y0, s = sq
        if s < args.min_roi:
            too_small += 1
            skipped += 1
            continue
        sides.append(s)
        if must is not None and x0 <= must[0] and y0 <= must[1] \
           and x0 + s >= must[2] and y0 + s >= must[3]:
            egg_kept += 1

        crop = a255[y0:y0 + s, x0:x0 + s] / 255.0
        if not args.no_color:
            crop = reinhard(crop, tgt_mean, tgt_std)
        out = Image.fromarray((crop * 255.0).astype(np.uint8)).resize(
            (args.size, args.size), Image.BICUBIC)
        od = os.path.join(args.out, c)
        os.makedirs(od, exist_ok=True)
        out.save(os.path.join(od, os.path.splitext(fname)[0] + ".jpg"), quality=95)
        made += 1
        per_class[c] += 1
        if made % 1000 == 0:
            print(f"  {made} written...", flush=True)

    print(f"\nwrote {made} ROIs to {args.out}   (skipped {skipped})")
    if too_small:
        print(f"  {too_small} rejected: ROI side below --min-roi={args.min_roi}px "
              f"(these were the pathological 11px cases)")
    if ann:
        print(f"  egg fully inside the ROI: {egg_kept}/{made} "
              f"({egg_kept/max(made,1)*100:.1f}%)")
        if no_egg:
            print(f"  {no_egg} images where no square could cover the egg "
                  f"(fell back to unconstrained)")
    if sides:
        sides = np.array(sides)
        print(f"  ROI side (original px): median {int(np.median(sides))}, "
              f"min {sides.min()}, max {sides.max()}  -> all resized to {args.size}")
    print(f"\n{'class':>26} {'ROIs':>6} {'black% BEFORE':>14}")
    for c in classes:
        bb = np.mean(black_before[c]) * 100 if black_before[c] else float("nan")
        print(f"{c[:26]:>26} {per_class[c]:>6} {bb:>13.1f}%")
    print("\nblack% BEFORE is the per-class letterbox fraction in the ORIGINALS -- the")
    print("artifact this extraction removes. After extraction it is 0 for every class")
    print("by construction. Re-run diagnose_frame_geometry.py on the output to confirm.")


if __name__ == "__main__":
    main()