r"""
probe_displaced_control.py — is the drop egg-SPECIFIC, or just damage?

THE QUESTION
------------
Masking the egg drops accuracy. Two things could cause that:

  (a) the model needed the egg          -> egg-specific, the interpretation we want
  (b) the model dislikes a grey rectangle anywhere -> information loss + out-of-
      distribution input, nothing to do with the egg

These are indistinguishable from the egg-masked number alone. This probe separates
them by masking a region of the SAME SIZE AND SHAPE, moved OFF the egg.

    displaced ~= unmodified   -> the drop is entirely egg-specific. Clean result.
    displaced ~= egg-masked   -> the drop is generic occlusion damage. The
                                 egg-masked figure means much less than claimed.
    in between                -> report the egg-specific FRACTION, printed below.

WHY IT MATTERS MOST FOR CROPS
-----------------------------
The crop box mask blanks 53.8% of the frame against 2.0% on whole images. If large
flat patches hurt regardless of position, the crop's low egg-masked score is partly
an artefact of mask size. That would weaken B3 in a way no amount of area-matching
can detect -- area-matching unmatches egg coverage, which is the variable that has
to stay pinned at 100%.

Expect a high residual overlap on crops: the egg spans 186-188 px in a 224 px frame,
so a same-size box CANNOT be placed clear of it. The achieved overlap is reported
per dataset and is itself a result -- on crops the control is weak by construction,
and that limit belongs in the writeup.

    python -u probe_displaced_control.py \
        --config configs/generated/roi477_protopnet_120ep.yaml \
        --ckpt   runs/roi477_protopnet_120ep/best.pt \
        --labels ../Data/chula_roi2_w477/labels.json \
        --device cuda --emit-tsv runs/displaced_control.tsv
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


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def displaced_candidates(mask, S, n_out, stride=8):
    """Same-size boxes placed to MINIMISE overlap with the egg box.

    Returns up to n_out (x0, y0, w, h, overlap_fraction), drawn from distinct
    quadrants so the controls are not all clustered in one corner. Deterministic.
    """
    x0, y0, x1, y1 = bbox_of(mask)
    w, h = x1 - x0, y1 - y0
    if w >= S or h >= S:
        return []
    cand = []
    for ny in range(0, S - h + 1, stride):
        for nx in range(0, S - w + 1, stride):
            ox = max(0, min(x1, nx + w) - max(x0, nx))
            oy = max(0, min(y1, ny + h) - max(y0, ny))
            cand.append((nx, ny, w, h, (ox * oy) / float(w * h)))
    if not cand:
        return []
    half = S / 2.0
    picked, used = [], set()
    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):           # one per quadrant first
        pool = [c for c in cand
                if ((c[0] + c[2] / 2) >= half) == bool(q[0])
                and ((c[1] + c[3] / 2) >= half) == bool(q[1])]
        if pool:
            best = min(pool, key=lambda c: c[4])
            picked.append(best)
            used.add(best[:2])
    picked.sort(key=lambda c: c[4])
    for c in sorted(cand, key=lambda c: c[4]):           # top up if quadrants ran out
        if len(picked) >= n_out:
            break
        if c[:2] not in used:
            picked.append(c)
            used.add(c[:2])
    return picked[:n_out]


class ControlSet(Dataset):
    """One decode; emits [unmodified, egg-masked, displaced_1..displaced_N].

    The displaced box is chosen at load time from the precomputed candidates: the
    lowest-overlap one whose footprint is at least `min_content` non-black. Without
    that check a displaced mask can land on the letterbox border of the rectangular-
    frame classes, blanking pixels that were already blank and trivially reproducing
    the unmodified score.
    """

    def __init__(self, samples, boxes, cands, img_size, fill, n_disp, min_content=0.7):
        self.samples, self.boxes, self.cands = samples, boxes, cands
        self.size, self.fill, self.n_disp = img_size, fill, n_disp
        self.min_content = min_content
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
        S = self.size
        b = self.boxes[i]
        out = [self.norm(x), self.norm(self._blank(x, b))]

        content = (x.mean(0) > 0.05).numpy()
        chosen, ov = [], []
        for (nx, ny, w, h, o) in self.cands[i]:
            if content[ny:ny + h, nx:nx + w].mean() >= self.min_content:
                m = np.zeros((S, S), dtype=bool)
                m[ny:ny + h, nx:nx + w] = True
                chosen.append(m)
                ov.append(o)
            if len(chosen) >= self.n_disp:
                break
        while len(chosen) < self.n_disp:                  # pad with the unmasked image
            chosen.append(None)
            ov.append(float("nan"))
        for m in chosen:
            out.append(self.norm(self._blank(x, m)))
        return torch.stack(out), label, torch.tensor(ov, dtype=torch.float32)


def test_samples(loaders):
    ds = loaders.test.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [base.samples[i] for i in ds.indices]
    return list(ds.samples)


@torch.no_grad()
def run(model, ds, device, bs, nw, nvar):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=nw > 0, prefetch_factor=4 if nw else None)
    hits = np.zeros(nvar)
    per_class = [defaultdict(lambda: [0, 0]) for _ in range(nvar)]
    ovs, total = [], 0
    amp = torch.autocast(device_type=device.type,
                         enabled=(device.type == "cuda"), dtype=torch.float16)
    for xb, y, ov in loader:
        B = y.numel()
        flat = xb.reshape(B * nvar, *xb.shape[2:]).to(device, non_blocking=True)
        if device.type == "cuda":
            flat = flat.contiguous(memory_format=torch.channels_last)
        with amp:
            pred = model(flat).float().argmax(1).cpu().reshape(B, nvar)
        for k in range(nvar):
            p = pred[:, k]
            hits[k] += (p == y).sum().item()
            for pi, ti in zip(p.tolist(), y.tolist()):
                per_class[k][ti][0] += int(pi == ti)
                per_class[k][ti][1] += 1
        ovs.append(ov.numpy())
        total += B
    return hits / max(total, 1), per_class, np.concatenate(ovs, 0), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill", default="mean", choices=["mean", "black", "noise"])
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--no-square", action="store_true")
    ap.add_argument("--n-displace", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--emit-tsv", default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    samples = test_samples(loaders)
    if args.n:
        rng = np.random.default_rng(1337)
        samples = [samples[i] for i in
                   rng.choice(len(samples), min(args.n, len(samples)), replace=False)]

    boxes = [box_in_crop(p, ann, S, margin, square) for p, _ in samples]
    keep = [i for i, b in enumerate(boxes) if b is not None and b.any()]
    samples = [samples[i] for i in keep]
    boxes = [boxes[i] for i in keep]
    cands = [displaced_candidates(b, S, args.n_displace) for b in boxes]

    box_area = float(np.mean([b.mean() for b in boxes])) * 100
    print(f"test set: {len(samples)} images with boxes   frame {S}x{S}")
    print(f"egg box covers {box_area:.1f}% of frame, 100% egg coverage "
          f"(minimum-area full removal)")
    print(f"displaced controls: {args.n_displace} per image, same size and shape\n")

    nvar = 2 + args.n_displace
    acc, per_class, ov, n = run(
        model, ControlSet(samples, boxes, cands, S, args.fill, args.n_displace),
        dev, args.batch_size, args.workers, nvar)
    chance = 1.0 / len(loaders.classes)
    base, egg = acc[0], acc[1]
    disp = acc[2:]
    dmean = float(np.nanmean(disp))
    ovm = float(np.nanmean(ov)) * 100
    tag = os.path.basename(os.path.dirname(args.ckpt))

    print(f"{'variant':<22}{'accuracy':>10}{'x chance':>10}{'egg overlap':>13}")
    print("-" * 55)
    print(f"{'unmodified':<22}{base:>10.4f}{base / chance:>10.2f}{'-':>13}")
    print(f"{'egg masked':<22}{egg:>10.4f}{egg / chance:>10.2f}{'100%':>13}")
    for k, a in enumerate(disp):
        print(f"{'displaced #' + str(k + 1):<22}{a:>10.4f}{a / chance:>10.2f}"
              f"{np.nanmean(ov[:, k]) * 100:>12.0f}%")
    print("-" * 55)
    print(f"{'displaced mean':<22}{dmean:>10.4f}{dmean / chance:>10.2f}{ovm:>12.0f}%")

    print("\n=== reading ===")
    total_drop = base - egg
    if total_drop <= 0.01:
        print("  Masking the egg costs nothing; the control cannot say anything.")
    else:
        frac = (dmean - egg) / total_drop
        print(f"  total drop from masking the egg : {total_drop * 100:.2f} points")
        print(f"  drop from an equal mask elsewhere: {(base - dmean) * 100:.2f} points")
        print(f"  EGG-SPECIFIC FRACTION OF THE DROP: {frac * 100:.1f}%")
        if frac > 0.85:
            print("  Clean. The loss is because the EGG went, not because a hole")
            print("  appeared. The egg-masked figure means what it claims.")
        elif frac > 0.5:
            print("  Mostly egg-specific, but a real share is generic occlusion")
            print("  damage. Quote the egg-specific fraction alongside the raw number.")
        else:
            print("  WEAK. Most of the drop is generic damage from occluding anything")
            print("  of this size. The egg-masked figure overstates egg reliance and")
            print("  should not be compared against datasets with smaller masks.")
    if ovm > 25:
        print(f"\n  CAVEAT: displaced masks still overlap the egg by {ovm:.0f}% on")
        print("  average -- the box is too large relative to the frame to be placed")
        print("  clear of it. The control is weak by construction on this dataset.")

    if args.emit_tsv:
        new = not os.path.exists(args.emit_tsv)
        with open(args.emit_tsv, "a") as f:
            if new:
                f.write("run\tbox_area_pct\tacc_none\tacc_eggmask\tacc_disp_mean\t"
                        "mean_overlap_pct\tegg_specific_frac\tchance\n")
            fr = (dmean - egg) / total_drop if total_drop > 0.01 else float("nan")
            f.write(f"{tag}\t{box_area:.2f}\t{base:.4f}\t{egg:.4f}\t{dmean:.4f}\t"
                    f"{ovm:.1f}\t{fr:.4f}\t{chance:.4f}\n")
        print(f"\n  appended -> {args.emit_tsv}")

    print(f"\n{'class':>24} {'normal':>8} {'egg mask':>9} {'displaced':>10}")
    for ci, cname in enumerate(loaders.classes):
        r = []
        for k in (0, 1):
            h, t = per_class[k][ci]
            r.append(h / max(t, 1))
        dd = np.mean([per_class[k][ci][0] / max(per_class[k][ci][1], 1)
                      for k in range(2, nvar)])
        print(f"{cname[:24]:>24} {r[0]:>8.3f} {r[1]:>9.3f} {dd:>10.3f}")


if __name__ == "__main__":
    main()
