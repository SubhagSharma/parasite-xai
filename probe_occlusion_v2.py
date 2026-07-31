r"""
probe_occlusion_v2.py — does the model use the egg, measured without the mask leak?

WHY v2
------
v1 blanked the ANNOTATION BOX. probe_bbox_geometry.py then measured that box geometry
alone classifies at 0.8069 (8.88x chance), and probe_scale_decomposition.py showed
scale normalisation makes box size a CLEANER species signal (0.5458 raw -> 0.7338
corrected). So on the ROI datasets the "egg masked" number has a floor set by the
shape of the hole, and normalising scale RAISED that floor. Every v1 number on
chula_roi2_* is contaminated by an amount nobody has measured.

v2 masks a FIXED-SIZE square centred on the egg instead. Side length is identical for
every image and every class, so the hole carries no size information. What survives is
mask POSITION, which is reported so you can see whether it is class-correlated.

DEFAULT MODE IS `both`. It runs the v1 box mask and the v2 fixed mask on the same
checkpoint in one pass and prints them side by side. The delta between them IS the
measurement of how much of last night's result was mask leakage.

SPEED
-----
v1 decoded the test set three times (once per mode) and ran on CPU. v2 decodes each
image ONCE, builds all five variants from it, and pushes them through the model as a
single batch of B*5 under autocast. On an A100 slice that is roughly 15-20x faster
than `--device cpu`; the whole test set takes well under a minute per checkpoint.

    python -u probe_occlusion_v2.py \
        --config configs/generated/roi477_blackbox_120ep.yaml \
        --ckpt   runs/roi477_blackbox_120ep/best.pt \
        --labels ../Data/chula_roi2_w477/labels.json \
        --device cuda

ALWAYS pass the labels.json that matches the dataset's coordinate system. For
chula_roi2_* that is the remapped file inside the dataset directory, never the
original Chula labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.eval.cropgeom import load_coco, box_in_crop

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

VARIANTS = ("none", "egg_box", "bg_box", "egg_fixed", "bg_fixed")


def centre_of(mask):
    ys, xs = np.nonzero(mask)
    return (int(ys.min() + ys.max()) // 2, int(xs.min() + xs.max()) // 2)


def long_axis(mask):
    ys, xs = np.nonzero(mask)
    return max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)


def fixed_square(centre, side, S):
    """Constant-area square centred as close to `centre` as the frame allows."""
    cy, cx = centre
    y0 = int(np.clip(cy - side // 2, 0, max(0, S - side)))
    x0 = int(np.clip(cx - side // 2, 0, max(0, S - side)))
    m = np.zeros((S, S), dtype=bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m


class VariantSet(Dataset):
    """One decode per image; returns all five variants stacked as (5,3,S,S)."""

    def __init__(self, samples, box_masks, fixed_masks, img_size, fill):
        self.samples = samples
        self.box_masks = box_masks
        self.fixed_masks = fixed_masks
        self.size = img_size
        self.fill = fill
        self.pre = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.samples)

    def _blank(self, x, region):
        x = x.clone()
        if region is None or not region.any():
            return x
        r = torch.from_numpy(region)
        if self.fill == "black":
            x[:, r] = 0.0
        elif self.fill == "noise":
            x[:, r] = torch.rand(3, int(r.sum()))
        else:
            for c, mv in enumerate(MEAN):
                x[c][r] = mv
        return x

    def __getitem__(self, i):
        path, label = self.samples[i]
        x = self.pre(Image.open(path).convert("RGB"))
        b, f = self.box_masks[i], self.fixed_masks[i]
        out = [x,
               self._blank(x, b),
               self._blank(x, None if b is None else ~b),
               self._blank(x, f),
               self._blank(x, None if f is None else ~f)]
        return torch.stack([self.norm(v) for v in out]), label


def test_samples(loaders):
    ds = loaders.test.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [base.samples[i] for i in ds.indices]
    return list(ds.samples)


@torch.no_grad()
def score_all(model, ds, device, bs, nw):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device.type == "cuda"))
    hits = {v: 0 for v in VARIANTS}
    per_class = {v: defaultdict(lambda: [0, 0]) for v in VARIANTS}
    total = 0
    amp = torch.autocast(device_type=device.type,
                         enabled=(device.type == "cuda"), dtype=torch.float16)
    for xb, y in loader:
        B = y.numel()
        flat = xb.reshape(B * len(VARIANTS), *xb.shape[2:]).to(device, non_blocking=True)
        with amp:
            pred = model(flat).float().argmax(1).cpu().reshape(B, len(VARIANTS))
        for k, v in enumerate(VARIANTS):
            p = pred[:, k]
            hits[v] += (p == y).sum().item()
            for pi, ti in zip(p.tolist(), y.tolist()):
                per_class[v][ti][0] += int(pi == ti)
                per_class[v][ti][1] += 1
        total += B
    return {v: hits[v] / max(total, 1) for v in VARIANTS}, per_class, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill", default="mean", choices=["mean", "black", "noise"])
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--no-square", action="store_true")
    ap.add_argument("--fixed-side", type=int, default=0,
                    help="mask side in model-input px; 0 = auto (99th pct egg long "
                         "axis, so the fixed mask covers 99%% of eggs whole)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = config value")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device
    dev = pick_device(cfg["device"])
    S = cfg["data"]["img_size"]
    margin, square = args.margin, not args.no_square

    ann = load_coco(args.labels)
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    samples = test_samples(loaders)
    if args.n:
        rng = np.random.default_rng(1337)          # random, never samples[:n]
        samples = [samples[i] for i in
                   rng.choice(len(samples), min(args.n, len(samples)), replace=False)]

    # one box_in_crop call per sample, reused everywhere
    box_masks = [box_in_crop(p, ann, S, margin, square) for p, _ in samples]
    ok = [m is not None and m.any() for m in box_masks]
    box_masks = [m if k else None for m, k in zip(box_masks, ok)]
    matched = sum(ok)

    axes = [long_axis(m) for m in box_masks if m is not None]
    side = args.fixed_side or int(np.ceil(np.percentile(axes, 99)))
    side = min(side, S)
    centres = [centre_of(m) if m is not None else None for m in box_masks]
    fixed_masks = [fixed_square(c, side, S) if c is not None else None for c in centres]

    print(f"test set: {len(samples)} images, {matched} with boxes "
          f"({matched / max(len(samples), 1) * 100:.1f}%)")
    print(f"box mask : egg long axis {np.percentile(axes, 5):.0f}-"
          f"{np.percentile(axes, 95):.0f} px (5th-95th), "
          f"area {np.mean([m.mean() for m in box_masks if m is not None]) * 100:.1f}% "
          f"of frame -- VARIES BY CLASS, this is the leak")
    print(f"fixed mask: side {side} px for every image, "
          f"area {side * side / (S * S) * 100:.1f}% of frame -- constant")
    cys = np.array([c[0] for c in centres if c]) / S
    cxs = np.array([c[1] for c in centres if c]) / S
    print(f"mask centre spread: y {cys.std():.3f}, x {cxs.std():.3f} "
          f"(0 = perfectly centred, so no positional leak)\n")

    bs = args.batch_size or cfg["data"]["batch_size"]
    ds = VariantSet(samples, box_masks, fixed_masks, S, args.fill)
    acc, per_class, n = score_all(model, ds, dev, bs, args.workers)
    chance = 1.0 / len(loaders.classes)

    print(f"{'':<22}{'BOX mask (v1)':>16}{'FIXED mask (v2)':>18}")
    print("-" * 56)
    print(f"{'unmodified':<22}{acc['none']:>16.4f}{acc['none']:>18.4f}")
    print(f"{'egg masked out':<22}{acc['egg_box']:>16.4f}{acc['egg_fixed']:>18.4f}")
    print(f"{'egg only kept':<22}{acc['bg_box']:>16.4f}{acc['bg_fixed']:>18.4f}")
    print("-" * 56)
    print(f"{'drop, egg removed':<22}"
          f"{(acc['none'] - acc['egg_box']) * 100:>15.2f}p"
          f"{(acc['none'] - acc['egg_fixed']) * 100:>17.2f}p")
    print(f"\n  chance level: {chance:.4f}")

    leak = acc["egg_box"] - acc["egg_fixed"]
    print("\n=== reading ===")
    print(f"  mask-geometry leak: {leak * 100:+.2f} points "
          f"({acc['egg_box']:.4f} box vs {acc['egg_fixed']:.4f} fixed)")
    if leak > 0.03:
        print("  The BOX number was inflated by the shape of the hole. Use the FIXED")
        print("  column; the v1 figure overstates background reliance by that much.")
    elif leak < -0.03:
        print("  The FIXED mask scores HIGHER -- it covers more of the frame, so it is")
        print("  removing context too. Treat it as an upper bound, not a clean read.")
    else:
        print("  Negligible. The v1 numbers stand as measured.")

    e, b = acc["egg_fixed"], acc["bg_fixed"]
    if e > acc["none"] - 0.05:
        print(f"  Removing the egg costs almost nothing ({e:.4f}). The model is reading")
        print("  context, not morphology.")
    else:
        print(f"  Removing the egg hurts ({e:.4f} vs {acc['none']:.4f}) -- the egg matters.")
    if b > acc["none"] - 0.05:
        print(f"  The egg alone is sufficient ({b:.4f}) -- morphology carries the signal.")
    else:
        print(f"  The egg alone is NOT sufficient ({b:.4f}); context does part of the work.")

    print(f"\n{'class':>24} {'normal':>8} {'noegg_box':>10} {'noegg_fix':>10} "
          f"{'eggonly':>8}")
    for ci, cname in enumerate(loaders.classes):
        r = []
        for v in ("none", "egg_box", "egg_fixed", "bg_fixed"):
            h, t = per_class[v][ci]
            r.append(h / max(t, 1))
        print(f"{cname[:24]:>24} {r[0]:>8.3f} {r[1]:>10.3f} {r[2]:>10.3f} {r[3]:>8.3f}")


if __name__ == "__main__":
    main()
