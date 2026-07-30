"""
measure_prototype_purity.py — item 4's measurement.

The CURRENT best.pt was trained with an UNFILTERED push: each prototype was
projected onto its nearest patch across ALL classes, so a prototype that votes for
class A may actually sit on a class-B patch. This script quantifies that, WITHOUT
retraining, so you know whether the existing prototype figure is trustworthy.

For each prototype, it finds the nearest training patch (exactly as the old push
did — all classes eligible) and reports whether that patch's IMAGE label matches the
prototype's assigned class (p // ppc). Output: per-class and overall same-class rate.

Interpretation:
  - high same-class rate (say >90%): the unfiltered push mostly landed correctly;
    the current figure is fine, and the class-filter fix (train.py) is tidiness.
  - low rate: prototypes are displaying wrong-species patches; regenerate the figure
    after retraining with the filtered push before using it in the paper.

CPU-only, reads best.pt, changes nothing.

    python measure_prototype_purity.py \
        --config configs/generated/A2_protopnet_mobilevit.yaml \
        --ckpt   runs/A2_protopnet_mobilevit/best.pt \
        --max-batches 0        # 0 = full training set (recommended); N = first N batches
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = full training set; N = first N batches (faster, approximate)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = "cpu"
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    head = model.head
    P, D = head.num_protos, head.proto_dim
    ppc = head.ppc
    proto_class = np.array([p // ppc for p in range(P)])         # assigned class per prototype

    best_dist = torch.full((P,), float("inf"))
    best_patch_class = torch.full((P,), -1, dtype=torch.long)    # class of the winning patch

    cap = args.max_batches if args.max_batches > 0 else None
    n_scanned = 0
    for bi, (x, y) in enumerate(loaders.train):
        if cap is not None and bi >= cap:
            break
        n_scanned += x.size(0)
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(B, H * W, D)          # (B,HW,D)
        # patch -> its image's class, flattened to match zf.reshape(-1,D)
        patch_cls = y.view(B, 1).expand(B, H * W).reshape(-1)    # (B*HW,)
        zf_flat = zf.reshape(-1, D)
        for p in range(P):
            pv = head.prototypes[p].view(1, D)
            d = ((zf_flat - pv) ** 2).sum(1)
            md, mi = d.min(0)
            if md < best_dist[p]:
                best_dist[p] = md
                best_patch_class[p] = patch_cls[mi]

    same = (best_patch_class.numpy() == proto_class)
    overall = same.mean()

    print(f"\n=== prototype purity on best.pt  (scanned {n_scanned} images"
          f"{' — FULL train set' if cap is None else f', first {cap} batches'}) ===")
    print(f"  prototypes: {P}  ({ppc} per class × {head.num_classes} classes)\n")

    by_class = defaultdict(lambda: [0, 0])
    for p in range(P):
        c = int(proto_class[p])
        by_class[c][0] += int(same[p])
        by_class[c][1] += 1
    print(f"{'class':>22} {'same-class protos':>18}")
    for c in range(head.num_classes):
        ok, tot = by_class[c]
        name = loaders.classes[c][:20]
        flag = "" if ok == tot else "  <-- has wrong-class prototypes"
        print(f"{name:>22} {ok:>8}/{tot:<8}{flag}")

    print(f"\n  OVERALL same-class rate: {overall*100:.1f}%  ({same.sum()}/{P})")
    if overall > 0.90:
        print("  -> unfiltered push mostly landed correctly; current figure is defensible.")
        print("     The class-filter fix in train.py is tidiness / paper-faithfulness.")
    else:
        print("  -> prototypes display wrong-species patches. Regenerate the prototype")
        print("     figure AFTER retraining with the filtered push before using it.")
    print()


if __name__ == "__main__":
    main()