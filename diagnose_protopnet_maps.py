"""
diagnose_protopnet_maps.py — Option-B part 1: is ProtoPNet's explanation SPATIAL?

The sanity_check = 1.0 on A2 is a degeneracy artifact: MPRT correlates the
trained-model explanation with the randomised-model one, and if the map is
near-constant the correlation is meaningless (returns ~1.0). This script settles
WHY the map is near-constant by measuring, on the TRAINED model, how much spatial
contrast sim_maps actually has — and renders a few so you can look.

Decision rule:
  - If contrast is healthy (the winning prototype's map has a clear peak, and
    std/|mean| is well above ~1e-2), the flatness only appears under randomisation
    -> harness fix (Option A) is correct and honest.
  - If contrast is poor even on the trained model, the sigmoid add-on is flattening
    the evidence -> the "this looks like that" heatmap is low-contrast for real,
    which is a model issue (Option C) and a C3 finding.

CPU-only. Reads runs/<name>/best.pt. Writes protopnet_map_diagnostic.png +
prints a contrast table. Does NOT touch the GPU or the running eval.

    python diagnose_protopnet_maps.py \
        --config configs/generated/A2_protopnet_mobilevit.yaml \
        --ckpt   runs/A2_protopnet_mobilevit/best.pt \
        --n 6
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model


def target_attr(ev, target):
    """The exact attribution ante_hoc_attr builds for ProtoPNet, per sample."""
    maps = ev["sim_maps"]                                    # (B,P,h,w)
    pc = ev["proto_class"]                                   # (P,C)
    sel = pc[:, target].t().view(target.size(0), -1, 1, 1)   # (B,P,1,1)
    return (maps * sel).sum(1, keepdim=True)                 # (B,1,h,w)


def contrast_stats(a):
    """Per-sample spatial-contrast measures on a (B,1,h,w) attribution."""
    flat = a.reshape(a.shape[0], -1).cpu().numpy().astype(np.float64)
    std = flat.std(axis=1)
    mean = np.abs(flat.mean(axis=1))
    rel = std / np.maximum(mean, 1e-12)                      # scale-free spread
    # peak-to-median ratio: how much the hottest pixel stands out
    peak = flat.max(axis=1)
    med = np.median(flat, axis=1)
    p2m = peak / np.maximum(np.abs(med), 1e-12)
    return rel, p2m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6, help="how many test images to render")
    ap.add_argument("--out", default="protopnet_map_diagnostic.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = "cpu"                                    # keep off the GPU
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)

    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # grab one batch of test images
    x, y = next(iter(loaders.test))
    x, y = x[: args.n], y[: args.n]

    with torch.no_grad():
        ev = model.explain(x)
        a = target_attr(ev, y)                               # (n,1,h,w)
        a_up = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)

    rel, p2m = contrast_stats(a)

    print("\n=== ProtoPNet sim_map spatial contrast on the TRAINED model ===")
    print(f"{'sample':>6} {'class':>6} {'std/|mean|':>12} {'peak/median':>13}")
    for i in range(len(y)):
        print(f"{i:>6} {loaders.classes[y[i]][:6]:>6} {rel[i]:>12.4f} {p2m[i]:>13.3f}")
    print(f"\n  mean std/|mean|  = {rel.mean():.4f}   (>~0.05 = usable spatial contrast)")
    print(f"  mean peak/median = {p2m.mean():.3f}   (>~1.5  = a real hotspot exists)")

    verdict = ("HEALTHY spatial contrast -> flatness is randomisation-only -> Option A is honest"
               if rel.mean() > 0.05 and p2m.mean() > 1.5
               else "LOW spatial contrast on the trained model -> sigmoid add-on flattens evidence -> Option C")
    print(f"\n  VERDICT: {verdict}\n")

    # ---- render ----
    inv = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imgs = (x * inv + mu).clamp(0, 1)

    n = len(y)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"{loaders.classes[y[i]][:10]}", fontsize=9)
        axes[0, i].axis("off")
        hm = a_up[i, 0].cpu().numpy()
        axes[1, i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[1, i].imshow(hm, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"rel={rel[i]:.3f}", fontsize=9)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("sim_map", fontsize=10)
    fig.suptitle("ProtoPNet evidence maps (trained model) — is the explanation spatial?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()