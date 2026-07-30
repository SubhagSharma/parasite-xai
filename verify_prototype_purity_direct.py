"""
verify_prototype_purity_direct.py — settle the 94.5% ambiguity.

measure_prototype_purity.py infers purity via a nearest-patch search across ALL
classes, so two near-identical patches from different classes can make a correctly
placed prototype LOOK wrong (an embedding tie, not a real error). This script asks
the direct question instead:

    Does prototype p sit EXACTLY on some patch from its own class (p // ppc)?

The filtered push sets each prototype equal to a specific same-class patch, so a
correctly-pushed prototype must have an (almost) exact match among that class's
patches. We measure the minimum L2 distance from each prototype to (a) the nearest
SAME-CLASS patch and (b) the nearest patch of ANY class. Interpretation:

  - same-class distance ~0  -> prototype IS on a same-class patch (correct), even if
    some other-class patch is marginally closer (that's the tie that fooled the NN
    metric). This confirms the push worked.
  - same-class distance >> 0 -> prototype is genuinely NOT on any same-class patch;
    the filter has a real gap and needs fixing.

For the 3 flagged classes especially, this says which case we're in.

CPU-only, reads best.pt, changes nothing.

    python verify_prototype_purity_direct.py \
        --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \
        --ckpt   runs/A2_protopnet_mobilevit_120ep/best.pt
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="L2 distance below this counts as 'exactly on a patch'")
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
    proto_class = np.array([p // ppc for p in range(P)])

    # collect ALL training patches, tagged by class, over the full set
    patches_by_class = {c: [] for c in range(head.num_classes)}
    for x, y in loaders.train:
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(B, H * W, D)          # (B,HW,D)
        for i in range(B):
            patches_by_class[int(y[i])].append(zf[i])            # (HW,D)
    for c in patches_by_class:
        patches_by_class[c] = (torch.cat(patches_by_class[c], 0)
                               if patches_by_class[c] else torch.empty(0, D))

    all_patches = torch.cat([patches_by_class[c] for c in range(head.num_classes)
                             if patches_by_class[c].numel()], 0)

    print(f"\n=== DIRECT prototype purity on best.pt (full train set) ===")
    print(f"  prototypes: {P}  | tol for 'on a patch' = {args.tol}\n")
    print(f"{'proto':>6} {'class':>18} {'d(same-class)':>14} {'d(any-class)':>13} {'verdict':>22}")

    same_ok = 0
    genuinely_wrong = []
    for p in range(P):
        c = int(proto_class[p])
        pv = head.prototypes[p].view(1, D)
        same_patches = patches_by_class[c]
        d_same = (((same_patches - pv) ** 2).sum(1).min().sqrt().item()
                  if same_patches.numel() else float("inf"))
        d_any = ((all_patches - pv) ** 2).sum(1).min().sqrt().item()

        on_same = d_same <= args.tol
        if on_same:
            same_ok += 1
            verdict = "on same-class patch"
        else:
            genuinely_wrong.append(p)
            verdict = "NOT on same-class!"
        # only print flagged/interesting rows to keep it readable
        if not on_same or d_same > 1e-4:
            print(f"{p:>6} {loaders.classes[c][:18]:>18} {d_same:>14.5f} {d_any:>13.5f} {verdict:>22}")

    print(f"\n  prototypes exactly on a same-class patch: {same_ok}/{P} "
          f"({same_ok/P*100:.1f}%)")

    if same_ok == P:
        print("\n  VERDICT: TRUE purity = 100%. Every prototype sits on a same-class")
        print("  patch. The 94.5% from the NN metric was embedding ties (other-class")
        print("  patches marginally closer), NOT real errors. Filter works; report 100%.")
    else:
        print(f"\n  VERDICT: {len(genuinely_wrong)} prototype(s) genuinely NOT on any")
        print(f"  same-class patch: {genuinely_wrong}. The filter has a real gap —")
        print("  do NOT report 100%; send this output back for a fix.")
    print()


if __name__ == "__main__":
    main()