r"""
probe_occlusion.py — does the model actually use the egg?

The prototypes sit on background 89% of the time and the evidence map points at the
annotated egg 0.4% of the time (chance 1.9%). This tests the consequence directly:
re-score the TRAINED model on masked copies of the test set.

    normal            unmodified                      -> reference (0.9921)
    egg masked        the annotated box blanked out    -> is the egg NEEDED?
    background masked everything EXCEPT the box blanked -> is the egg SUFFICIENT?

Reading:
    egg-masked stays high     -> the egg is not needed; prediction rides on context
    background-masked crashes -> the egg alone is not sufficient
    both of the above         -> the model is classifying acquisition conditions,
                                 not parasite morphology. Dataset/model confound.
    egg-masked crashes AND
    background-masked holds   -> the model genuinely uses the egg, and the
                                 prototype/attribution problem is a display bug only

Masking happens in ORIGINAL image coordinates, before the resize, so boxes line up
exactly. Fill defaults to the ImageNet mean (~0 after normalisation), which is the
least out-of-distribution choice; --fill black is the harsher variant.

    python probe_occlusion.py \
        --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \
        --ckpt   runs/A2_protopnet_mobilevit_120ep/best.pt \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json --device cpu
"""
from __future__ import annotations

import argparse
import json
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




class MaskedSet(Dataset):
    """ImageFolder samples with the egg region (or its complement) blanked out.

    Masking happens AFTER the resize, in the same coordinate frame the model sees,
    using cropgeom.box_in_crop. That handles both whole images (boxes scaled from
    original coords) and crops (the box mapped into the reproduced crop rectangle),
    which the earlier original-coordinate version could not do for crops.
    """

    def __init__(self, samples, ann, img_size, mode="none", fill="mean",
                 margin=0.20, square=True):
        self.samples = samples
        self.ann = ann
        self.mode = mode                # none | egg | background
        self.fill = fill
        self.size = img_size
        self.margin, self.square = margin, square
        self.pre = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize(MEAN, STD)
        self.n_masked = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        x = self.pre(Image.open(path).convert("RGB"))        # (3,S,S) in [0,1]

        if self.mode != "none":
            gt = box_in_crop(path, self.ann, self.size, self.margin, self.square)
            if gt is not None and gt.any():
                region = torch.from_numpy(gt if self.mode == "egg" else ~gt)
                if self.fill == "black":
                    x[:, region] = 0.0
                elif self.fill == "noise":
                    x[:, region] = torch.rand(3, int(region.sum()))
                else:                                        # dataset mean
                    for c, mv in enumerate(MEAN):
                        x[c][region] = mv
                self.n_masked += 1

        return self.norm(x), label


def test_samples(loaders):
    ds = loaders.test.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [base.samples[i] for i in ds.indices]
    return list(ds.samples)


@torch.no_grad()
def score(model, ds, device, bs, nw):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    correct = total = 0
    per_class = defaultdict(lambda: [0, 0])
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for p, t in zip(pred.tolist(), y.tolist()):
            per_class[t][0] += int(p == t)
            per_class[t][1] += 1
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fill", default="mean", choices=["mean", "black", "noise"])
    ap.add_argument("--margin", type=float, default=0.20,
                    help="margin used by make_crops (to map boxes into crops)")
    ap.add_argument("--no-square", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="0 = whole test set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device
    dev = pick_device(cfg["device"])
    size = cfg["data"]["img_size"]

    ann = load_coco(args.labels)
    margin, square = args.margin, not args.no_square
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    samples = test_samples(loaders)
    if args.n:
        samples = samples[: args.n]
    matched = sum(1 for p, _ in samples
                  if box_in_crop(p, ann, size, margin, square) is not None)
    print(f"test set: {len(samples)} images, {matched} with boxes "
          f"({matched/max(len(samples),1)*100:.1f}%)")
    cov = [float(box_in_crop(p, ann, size, margin, square).mean())
           for p, _ in samples[:200]
           if box_in_crop(p, ann, size, margin, square) is not None]
    import numpy as _np
    print(f"fill: {args.fill}   mean egg coverage of the frame: "
          f"{_np.mean(cov)*100:.1f}%\n")

    bs, nw = cfg["data"]["batch_size"], 2
    results = {}
    for mode, desc in (("none", "unmodified"),
                       ("egg", "EGG masked out"),
                       ("background", "BACKGROUND masked out")):
        ds = MaskedSet(samples, ann, size, mode, args.fill, margin, square)
        acc, per_class = score(model, ds, dev, bs, nw)
        results[mode] = (acc, per_class)
        print(f"  {desc:>24}: accuracy {acc:.4f}")

    base = results["none"][0]
    egg = results["egg"][0]
    bg = results["background"][0]
    chance = 1.0 / len(loaders.classes)

    print(f"\n  chance level: {chance:.4f}")
    print(f"  drop when the egg is REMOVED     : {(base-egg)*100:+.2f} points")
    print(f"  drop when only the egg is KEPT   : {(base-bg)*100:+.2f} points")

    print("\n=== reading ===")
    if egg > base - 0.05:
        print(f"  Removing the egg costs almost nothing ({egg:.4f} vs {base:.4f}).")
        print("  The parasite is NOT needed for the prediction -- the model is reading")
        print("  context/acquisition, not morphology. This is a confound.")
    else:
        print(f"  Removing the egg hurts ({egg:.4f} vs {base:.4f}) -- the egg matters.")
    if bg > base - 0.05:
        print(f"  The egg alone is sufficient ({bg:.4f}) -- morphology carries the signal.")
    else:
        print(f"  The egg alone is NOT sufficient ({bg:.4f} vs {base:.4f}); context is")
        print("  doing part of the work.")

    print(f"\n{'class':>24} {'normal':>8} {'no egg':>8} {'egg only':>9}")
    for ci, cname in enumerate(loaders.classes):
        r = []
        for m in ("none", "egg", "background"):
            h, t = results[m][1][ci]
            r.append(h / max(t, 1))
        print(f"{cname[:24]:>24} {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>9.3f}")


if __name__ == "__main__":
    main()