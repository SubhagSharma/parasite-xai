r"""
probe_occlusion_v2.py — egg-masked accuracy as a CURVE over mask size, not a number.

WHY A SWEEP
-----------
Two confounds make any single egg-masked figure uninterpretable.

  1. BOX-GEOMETRY LEAK. Masking the annotation box leaves a hole whose SIZE encodes
     the species (box geometry alone classifies at 0.8069). Measured leak on your
     checkpoints: +8 to +34 points.

  2. MASK-AREA MISMATCH. The v1 box mask covered 2.0% of the frame on whole images,
     11.5% on roi477 and 53.8% on crops. "0.5700 whole vs 0.2494 crops" therefore
     compared removing a fiftieth of the picture against removing half of it. The
     first fixed-mask run inherited this: auto-sizing to the 99th percentile egg
     gave 27.8% of frame on whole images but 79.7% on crops.

A constant-size square fixes (1). Only a SWEEP fixes (2): run several sizes, then
compare models at EQUAL masked fraction. The curve is the measurement.

THE COLUMN THAT MAKES IT READABLE
---------------------------------
`egg cov` is the fraction of the annotated egg actually inside the mask. A row is
only evidence about background reliance where egg coverage is high AND frame area
is low. On crops that region may not exist — a crop is mostly egg, so you cannot
remove the egg without removing the image. That is itself the finding, and this
column is what shows it.

    python -u probe_occlusion_v2.py \
        --config configs/generated/roi477_blackbox_120ep.yaml \
        --ckpt   runs/roi477_blackbox_120ep/best.pt \
        --labels ../Data/chula_roi2_w477/labels.json \
        --device cuda --emit-tsv runs/occlusion_sweep.tsv

ALWAYS pass the labels.json matching that dataset's coordinate system. For
chula_roi2_* that is the remapped file inside the dataset directory.
"""
from __future__ import annotations

import argparse
import os
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


def centre_of(mask):
    ys, xs = np.nonzero(mask)
    return (int(ys.min() + ys.max()) // 2, int(xs.min() + xs.max()) // 2)


def long_axis(mask):
    ys, xs = np.nonzero(mask)
    return max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)


def fixed_square(centre, side, S):
    cy, cx = centre
    y0 = int(np.clip(cy - side // 2, 0, max(0, S - side)))
    x0 = int(np.clip(cx - side // 2, 0, max(0, S - side)))
    m = np.zeros((S, S), dtype=bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m


class SweepSet(Dataset):
    """One decode per image; emits every mask variant built from that decode."""

    def __init__(self, samples, masks, img_size, fill):
        self.samples, self.masks = samples, masks     # masks: list of list-of-(name,arr)
        self.size, self.fill = img_size, fill
        self.pre = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor()])
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
        out = [self.norm(x)]
        for _, arr in self.masks[i]:
            out.append(self.norm(self._blank(x, arr)))
        return torch.stack(out), label


def test_samples(loaders):
    ds = loaders.test.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [base.samples[i] for i in ds.indices]
    return list(ds.samples)


@torch.no_grad()
def score(model, ds, device, bs, nw, nvar):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device.type == "cuda"))
    hits = np.zeros(nvar)
    per_class = [defaultdict(lambda: [0, 0]) for _ in range(nvar)]
    total = 0
    amp = torch.autocast(device_type=device.type,
                         enabled=(device.type == "cuda"), dtype=torch.float16)
    for xb, y in loader:
        B = y.numel()
        flat = xb.reshape(B * nvar, *xb.shape[2:]).to(device, non_blocking=True)
        with amp:
            pred = model(flat).float().argmax(1).cpu().reshape(B, nvar)
        for k in range(nvar):
            p = pred[:, k]
            hits[k] += (p == y).sum().item()
            for pi, ti in zip(p.tolist(), y.tolist()):
                per_class[k][ti][0] += int(pi == ti)
                per_class[k][ti][1] += 1
        total += B
    return hits / max(total, 1), per_class, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill", default="mean", choices=["mean", "black", "noise"])
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--no-square", action="store_true")
    ap.add_argument("--sides", default="32,64,96,128,160",
                    help="comma list of mask sides in MODEL-INPUT px. Identical "
                         "across datasets, so area fractions are comparable.")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--emit-tsv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device
    dev = pick_device(cfg["device"])
    S = cfg["data"]["img_size"]
    margin, square = args.margin, not args.no_square
    sides = [int(s) for s in args.sides.split(",") if int(s) <= S]

    ann = load_coco(args.labels)
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    samples = test_samples(loaders)
    if args.n:
        rng = np.random.default_rng(1337)               # random, never samples[:n]
        samples = [samples[i] for i in
                   rng.choice(len(samples), min(args.n, len(samples)), replace=False)]

    boxes = [box_in_crop(p, ann, S, margin, square) for p, _ in samples]
    boxes = [b if (b is not None and b.any()) else None for b in boxes]
    matched = sum(b is not None for b in boxes)

    # variant order: [box_egg, box_bg] then per side [egg, bg]
    names, per_sample = ["box_egg", "box_bg"], []
    for s in sides:
        names += [f"egg{s}", f"bg{s}"]
    cover = {s: [] for s in sides}
    for b in boxes:
        v = [("box_egg", b), ("box_bg", None if b is None else ~b)]
        for s in sides:
            f = fixed_square(centre_of(b), s, S) if b is not None else None
            v += [(f"egg{s}", f), (f"bg{s}", None if f is None else ~f)]
            if b is not None:
                cover[s].append(float((b & f).sum()) / float(b.sum()))
        per_sample.append(v)
    nvar = 1 + len(names)

    axes = [long_axis(b) for b in boxes if b is not None]
    box_area = float(np.mean([b.mean() for b in boxes if b is not None])) * 100
    print(f"test set: {len(samples)} images, {matched} with boxes "
          f"({matched / max(len(samples), 1) * 100:.1f}%)   frame {S}x{S}")
    print(f"egg long axis: {np.percentile(axes, 5):.0f}-{np.percentile(axes, 95):.0f} px "
          f"(5th-95th), box mask covers {box_area:.1f}% of frame")

    bs = args.batch_size or cfg["data"]["batch_size"]
    acc, per_class, n = score(model, SweepSet(samples, per_sample, S, args.fill),
                              dev, bs, args.workers, nvar)
    chance = 1.0 / len(loaders.classes)
    base = acc[0]
    idx = {nm: i + 1 for i, nm in enumerate(names)}
    tag = os.path.basename(os.path.dirname(args.ckpt))

    print(f"\nunmodified accuracy {base:.4f}   chance {chance:.4f}\n")
    print(f"{'mask':>8}{'frame %':>10}{'egg cov %':>11}{'egg masked':>13}"
          f"{'egg only':>11}{'x chance':>10}")
    print("-" * 63)
    ebox = acc[idx["box_egg"]]
    print(f"{'box':>8}{box_area:>9.1f}{100.0:>11.1f}{ebox:>13.4f}"
          f"{acc[idx['box_bg']]:>11.4f}{ebox / chance:>10.2f}")
    rows = []
    for s in sides:
        e, g = acc[idx[f"egg{s}"]], acc[idx[f"bg{s}"]]
        area, cv = s * s / (S * S) * 100, np.mean(cover[s]) * 100
        print(f"{s:>8}{area:>9.1f}{cv:>11.1f}{e:>13.4f}{g:>11.4f}{e / chance:>10.2f}")
        rows.append((s, area, cv, e, g))

    print("\n=== reading ===")
    good = [r for r in rows if r[2] >= 90.0]
    if good:
        s, area, cv, e, g = min(good, key=lambda r: r[1])
        print(f"  Cleanest row: side {s}, covering {cv:.0f}% of the egg while blanking")
        print(f"  only {area:.0f}% of the frame -> egg-masked {e:.4f} ({e/chance:.2f}x chance).")
        print("  Compare models at THIS row, not at the box row.")
    else:
        print("  No mask reaches 90% egg coverage below full-frame. The egg fills too")
        print("  much of this dataset to be removed without destroying the image, so")
        print("  'background reliance' is not separately measurable here. Report that.")
    # Compare the box against the fixed side of closest FRAME AREA. The two cannot
    # also be matched on egg coverage -- the box reaches 100% at low area precisely
    # because it is egg-shaped, which is the leak. So the delta below is an UPPER
    # BOUND on the leak, inflated by whatever coverage the fixed mask gives up.
    m_s, m_area, m_cov, m_e, _ = min(rows, key=lambda r: abs(r[1] - box_area))
    print(f"\n  matched-area leak estimate (box {box_area:.1f}% vs side {m_s} "
          f"{m_area:.1f}% of frame):")
    print(f"    box {ebox:.4f} at 100% egg coverage  vs  "
          f"side {m_s} {m_e:.4f} at {m_cov:.0f}%")
    print(f"    delta {(ebox - m_e) * 100:+.2f} points -- UPPER BOUND on the geometry")
    print(f"    leak; {100 - m_cov:.0f} points of that gap is lost egg coverage, not leak.")

    if args.emit_tsv:
        new = not os.path.exists(args.emit_tsv)
        with open(args.emit_tsv, "a") as f:
            if new:
                f.write("run\tside\tframe_pct\tegg_cov_pct\tacc_none\t"
                        "acc_eggmask\tacc_eggonly\tchance\n")
            f.write(f"{tag}\tbox\t"
                    f"{np.mean([b.mean() for b in boxes if b is not None])*100:.2f}\t"
                    f"100.00\t{base:.4f}\t{ebox:.4f}\t{acc[idx['box_bg']]:.4f}\t"
                    f"{chance:.4f}\n")
            for s, area, cv, e, g in rows:
                f.write(f"{tag}\t{s}\t{area:.2f}\t{cv:.2f}\t{base:.4f}\t"
                        f"{e:.4f}\t{g:.4f}\t{chance:.4f}\n")
        print(f"\n  appended {len(rows) + 1} rows -> {args.emit_tsv}")

    ref = min(good, key=lambda r: r[1])[0] if good else sides[-1]
    print(f"\n{'class':>24} {'normal':>8} {'noegg_box':>10} {'noegg_' + str(ref):>10}")
    for ci, cname in enumerate(loaders.classes):
        r = []
        for k in (0, idx["box_egg"], idx[f"egg{ref}"]):
            h, t = per_class[k][ci]
            r.append(h / max(t, 1))
        print(f"{cname[:24]:>24} {r[0]:>8.3f} {r[1]:>10.3f} {r[2]:>10.3f}")


if __name__ == "__main__":
    main()
