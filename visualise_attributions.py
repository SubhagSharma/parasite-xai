"""
visualise_attributions.py — look at where each method says the model is looking.

WHY
---
Everything in this project so far is aggregate: egg-masked accuracy, deletion,
Spearman correlations. Aggregates hide the failure that matters. Three specific
questions are unanswerable from numbers alone:

  * HOOKWORM scores 1.000 egg-masked under every model, dataset and mask tested,
    and the displaced control says it is genuine. WHAT is the model reading?
  * ProtoPNet's deletion is 0.0038, 5x better than any post-hoc method. Is the
    explanation on the egg, or is it sharply concentrated somewhere irrelevant?
    (A tight, confident, WRONG attribution scores well on deletion.)
  * LIME's sanity_check is exactly 0.0000 on all four models. Its attributions
    presumably collapse on a randomised model -- but do they look sane on the
    trained one?

Renders a grid: one row per image, one column per method, attribution overlaid on
the image with the annotation box drawn in. The `frac` figure printed under each
panel is the share of total attribution mass falling inside the box -- the pointing
game, per panel, so the picture and the number are visible together.

    # a stratified look at everything
    python -u visualise_attributions.py \
        --config configs/generated/roi477_protopnet_120ep.yaml \
        --ckpt runs/roi477_protopnet_120ep/best.pt \
        --labels ../Data/chula_roi2_w477/labels.json \
        --out figs/attr_protopnet.png

    # the Hookworm question, 8 images of one class
    python -u visualise_attributions.py --config ... --ckpt ... --labels ... \
        --only "Hookworm egg" --n 8 --out figs/attr_hookworm.png

    # what does the model see with the egg blanked? (the shortcut, visualised)
    python -u visualise_attributions.py --config ... --ckpt ... --labels ... \
        --only "Hookworm egg" --mask-egg --out figs/attr_hookworm_masked.png

--mask-egg is the important one for Hookworm: it blanks the annotation box before
explaining, so the attribution shows what the model falls back on. That is the
shortcut itself, rendered.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.evaluate import ante_hoc_attr
from pxai.explainers.posthoc import explain_posthoc
from pxai.eval.cropgeom import load_coco, box_in_crop

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(x):
    return np.clip(x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


def norm01(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def mass_in_box(attr, mask):
    """Pointing-game fraction: share of |attribution| inside the annotation box."""
    a = np.abs(np.asarray(attr, dtype=np.float64))
    tot = a.sum()
    return float(a[mask].sum() / tot) if tot > 0 and mask is not None else float("nan")


def test_samples(loaders):
    ds = loaders.test.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [(base.samples[i], i) for i in ds.indices]
    return [(s, i) for i, s in enumerate(ds.samples)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="figs/attributions.png")
    ap.add_argument("--n", type=int, default=6, help="images to render")
    ap.add_argument("--only", default=None, help="restrict to one class name")
    ap.add_argument("--methods", default="gradcam,hirescam,integrated_gradients,lime,kernelshap")
    ap.add_argument("--mask-egg", action="store_true",
                    help="blank the annotation box before explaining: shows the shortcut")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    S = cfg["data"]["img_size"]
    kind = cfg["model"]["kind"]

    ann = load_coco(a.labels)
    loaders = build_loaders(cfg)
    classes = loaders.classes
    cfg["model"]["num_classes"] = len(classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.eval()

    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))

    if a.only:
        if a.only not in classes:
            raise SystemExit(f"--only {a.only!r} not in {classes}")
        ci = classes.index(a.only)
        idxs = [i for i in idxs if base.samples[i][1] == ci]
        if not idxs:
            raise SystemExit(f"no test images for {a.only!r}")

    rng = np.random.default_rng(a.seed)          # random, never idxs[:n]
    pick = rng.choice(len(idxs), min(a.n, len(idxs)), replace=False)
    chosen = [idxs[i] for i in pick]

    methods = [m for m in a.methods.split(",") if m]
    cols = (["ours:" + kind] if kind != "blackbox" else []) + methods
    ncol = 1 + len(cols)

    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(2.3 * ncol, 2.55 * len(chosen)), squeeze=False)

    for r, gi in enumerate(chosen):
        path, label = base.samples[gi]
        x, y = base[gi]
        x = x.unsqueeze(0).to(dev)
        t = torch.tensor([label], device=dev)

        box = box_in_crop(path, ann, S, a.margin, True)
        if a.mask_egg and box is not None:
            for c, mv in enumerate(MEAN):
                x[0, c][torch.from_numpy(box).to(dev)] = float(mv)

        with torch.no_grad():
            pred = model(x).argmax(1).item()
        img = denorm(x[0])

        ax = axes[r][0]
        ax.imshow(img)
        if box is not None:
            ys, xs = np.nonzero(box)
            ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                   fill=False, ec="lime", lw=1.6))
        ok = "OK" if pred == label else f"-> {classes[pred][:12]}"
        ax.set_title(f"{classes[label][:16]}\n{ok}", fontsize=7)
        ax.axis("off")
        if r == 0:
            ax.text(0.5, 1.28, "input" + (" (egg masked)" if a.mask_egg else ""),
                    transform=ax.transAxes, ha="center", fontsize=8, weight="bold")

        for c, name in enumerate(cols):
            ax = axes[r][c + 1]
            try:
                if name.startswith("ours:"):
                    with torch.enable_grad():
                        at = ante_hoc_attr(kind)(model, x, t)
                else:
                    with torch.enable_grad():
                        at = explain_posthoc(name, model, x, t)[0]
                m = at.detach().float().cpu().numpy()
                m = m[0, 0] if m.ndim == 4 else m.squeeze()
                ax.imshow(img)
                ax.imshow(norm01(m), cmap="jet", alpha=0.45)
                f = mass_in_box(m, box)
                ax.set_title(f"frac {f:.2f}" if f == f else "frac n/a", fontsize=7)
            except Exception as e:
                ax.imshow(img)
                ax.set_title(f"FAILED\n{type(e).__name__}", fontsize=6, color="red")
            if box is not None:
                ys, xs = np.nonzero(box)
                ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                       fill=False, ec="lime", lw=1.2))
            ax.axis("off")
            if r == 0:
                ax.text(0.5, 1.28, name.replace("integrated_gradients", "IG"),
                        transform=ax.transAxes, ha="center", fontsize=8, weight="bold")

    ttl = os.path.basename(os.path.dirname(a.ckpt))
    fig.suptitle(f"{ttl}   green = annotation box   "
                 f"frac = attribution mass inside it"
                 + ("   [EGG MASKED]" if a.mask_egg else ""),
                 fontsize=9, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=135, bbox_inches="tight")
    print(f"wrote {a.out}   {len(chosen)} images x {ncol} panels")
    print("\nWHAT TO LOOK FOR")
    print("  frac near 1.0  -> attribution is on the egg")
    print("  frac near 0.0  -> it is not, whatever the deletion score says")
    print("  a tight blob OFF the egg with a good deletion score is the case the")
    print("  aggregates cannot show you: confidently faithful to the wrong feature.")
    if a.mask_egg:
        print("  With --mask-egg, every frac SHOULD be low (the egg is gone). What")
        print("  matters is WHERE the mass went instead -- that location is the shortcut.")


if __name__ == "__main__":
    main()
