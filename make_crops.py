r"""
make_crops.py — build an ROI-cropped copy of Chula-ParasiteEgg-11.

Why: the whole-image model reaches 0.9921 but classifies 5 of 11 species with the
parasite blacked out (0.5700 with the egg masked; Hookworm 1.000). 89% of its
prototypes sit on background, and the evidence map hits the annotated egg 0.4% of
the time against a 1.9% baseline. The model is reading acquisition context.

IPI-CVx (the base paper) crops to the ROI for exactly this reason -- "it was
essential to minimize background noise", D_egg <- D_train (*) M_bbox. This script
does the same, producing a normal ImageFolder tree so the rest of the pipeline is
unchanged:

    chula_crops/<class_name>/<original_stem>_<k>.jpg

Design choices, each deliberate:
  --margin 0.20   expand the box 20% before cropping. A tight crop clips the shell
                  boundary and operculum -- exactly the morphology a clinician uses.
  --square        expand the shorter side to match the longer BEFORE resizing, so
                  the egg is not distorted. Shape is diagnostic for eggs; a
                  non-aspect-preserving resize would destroy it.
  one crop per box, so images with 2 eggs yield 2 crops (11,031 boxes / 11,000 imgs).

NOTE ON SPLITTING: crops from the same source image must not straddle train/test.
The stem is preserved in the filename so a grouped split can be recovered later;
with ~31 multi-box images the risk is small but real.

    python make_crops.py \
        --root   ../Data/Chula-ParasiteEgg-11/data \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json \
        --out    ../Data/chula_crops --margin 0.20 --square
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from PIL import Image


def load_index(labels_path):
    """-> {basename: (class_name, [(x,y,w,h), ...])}"""
    with open(labels_path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: os.path.basename(im["file_name"]) for im in coco["images"]}
    boxes = defaultdict(list)
    cls = {}
    for a in coco["annotations"]:
        if a["image_id"] not in imgs:
            continue
        name = imgs[a["image_id"]]
        boxes[name].append(tuple(a["bbox"]))
        cls[name] = cats.get(a["category_id"], str(a["category_id"]))
    return {n: (cls[n], boxes[n]) for n in boxes}


def crop_box(im, box, margin, square):
    W, H = im.size
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    w *= (1 + margin)
    h *= (1 + margin)
    if square:
        w = h = max(w, h)
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    x1, y1 = cx + w / 2.0, cy + h / 2.0
    # clip to the frame, keeping the box centred where possible
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return im.crop((x0, y0, x1, y1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ImageFolder root of the originals")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--min-side", type=int, default=32,
                    help="skip boxes whose crop is smaller than this")
    args = ap.parse_args()

    idx = load_index(args.labels)
    print(f"annotations: {len(idx)} images, "
          f"{sum(len(v[1]) for v in idx.values())} boxes")

    # walk the source tree so we use the actual files on disk
    files = {}
    for dirpath, _, names in os.walk(args.root):
        for n in names:
            if n.lower().endswith((".jpg", ".jpeg", ".png")):
                files.setdefault(n, os.path.join(dirpath, n))
    print(f"source tree: {len(files)} image files under {args.root}")

    made, skipped, missing, small = 0, 0, 0, 0
    per_class = defaultdict(int)
    sizes = []
    for name, (cname, boxes) in idx.items():
        path = files.get(name)
        if path is None:
            missing += 1
            continue
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        outdir = os.path.join(args.out, cname)
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.splitext(name)[0]
        for k, b in enumerate(boxes):
            c = crop_box(im, b, args.margin, args.square)
            if c is None or min(c.size) < args.min_side:
                small += 1
                continue
            c.save(os.path.join(outdir, f"{stem}_{k}.jpg"), quality=95)
            sizes.append(c.size)
            per_class[cname] += 1
            made += 1
        if made and made % 2000 == 0:
            print(f"  {made} crops written...", flush=True)

    print(f"\nwrote {made} crops to {args.out}")
    if missing:
        print(f"  {missing} annotated images not found in the source tree")
    if small:
        print(f"  {small} boxes skipped (crop smaller than {args.min_side}px)")
    if sizes:
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"  crop size: median {sorted(ws)[len(ws)//2]}x{sorted(hs)[len(hs)//2]}, "
              f"min {min(ws)}x{min(hs)}, max {max(ws)}x{max(hs)}")
    print(f"\n{'class':>26} {'crops':>7}")
    for c, n in sorted(per_class.items()):
        print(f"{c[:26]:>26} {n:>7}")

    print("\nnext: point a config at this root and train.")
    print("  NOTE crops are much smaller than the originals; consider img_size 128")
    print("  and re-checking that the model still fits the sub-10MB target.")


if __name__ == "__main__":
    main()
